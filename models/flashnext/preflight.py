"""Offline preflight for a Flash-Next checkpoint directory.

Answers "is this directory a complete checkpoint the Flash-Next runtime can
load, with the precision and routing the project validated?" without loading
the model and, by default, without opening a single weight shard. It reads only
``config.json``, the safetensors index, the tokenizer files and the directory
listing, and calls ``stat`` on the shards. That is safe to run while a download
is still writing the shards.

Two checks look inside shards and are opt-in: ``headers`` (safetensors headers
only, a few hundred KB per shard) and ``identity`` (the repo's content identity,
which hashes those headers). Both refuse to run while the download is
incomplete, so they never compete with it.

Verdicts: ``READY``, ``INCOMPLETE`` (a download problem: missing, partial or
still-downloading files), ``INCOMPATIBLE`` (the metadata is wrong for the
runtime or would change precision or routing), ``MISSING`` (no checkpoint).

    python -m models.flashnext.preflight ~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP
    python -m models.flashnext.preflight vontra-mtp --json

Exit status: 0 READY, 1 INCOMPLETE, 2 INCOMPATIBLE or MISSING.

Nothing here is imported by the runtime, changes how a checkpoint loads, or
changes any weight, precision or routing decision.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .checkpoint_compat import MTP_PREFIX, MTP_SIDECAR_NAME, validate_mtp_structure

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
READY, INCOMPLETE, INCOMPATIBLE, MISSING = "READY", "INCOMPLETE", "INCOMPATIBLE", "MISSING"

MANIFEST_DIR = Path(__file__).with_name("preflight_manifests")
DEFAULT_MANIFEST = "qwen3.8-flash-next-mlx-4bit-mtp.json"
HEADER_LIMIT = 100 * 1024 * 1024        # the repo's own bound for a safetensors header

# The validated checkpoint (Vontra/TensorFold Qwen3.8-Flash-Next-MLX-4bit-MTP).
# A mismatch in any of these is reported as FAIL: the runtime might still load a
# different checkpoint, but it is not what the memory plans, ring sizes and
# recorded results were made for.
PROFILE = {
    "model_type": "qwen4_exp", "text_model_type": "qwen4_exp_text",
    "hidden_size": 2560, "num_hidden_layers": 48, "num_experts": 512,
    "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640, "vocab_size": 248320,
    "mtp_num_hidden_layers": 1,
}
QUANT = {"bits": 4, "group_size": 32, "mode": "affine"}
LAYER_TYPE_COUNTS = {"linear_attention": 36, "full_attention": 12}
RESIDENT_EXPERTS_POLICY = 8             # applies only if the content identity matches

# Tensor families, derived from the real index (36 linear-attention and 12
# full-attention layers have identical sets). "WSB" is a quantized triplet
# (weight, scales, biases), "W" a plain weight, "" a bare tensor name.
_HYPER = {f"{kind}_hyper_connection.{part}": spec for kind in ("attn", "mlp") for part, spec in (
    ("block_inject_weight", "WSB"), ("hc_norm", "W"),
    ("input_mix_weight_down", "WSB"), ("input_mix_weight_up", "WSB"))}
MLP = {
    **_HYPER,
    "mlp.gate": "W",                    # the router: stored unquantized
    "mlp.shared_expert_gate": "WSB",
    **{f"mlp.shared_expert.{p}_proj": "WSB" for p in ("gate", "up", "down")},
    **{f"mlp.switch_mlp.{p}_proj": "WSB" for p in ("gate", "up", "down")},
}
LINEAR_ATTENTION = {
    "linear_attn.A_log": "", "linear_attn.dt_bias": "", "linear_attn.conv1d": "W",
    "linear_attn.norm": "W", "linear_attn.out_proj": "WSB",
    **{f"linear_attn.in_proj_{p}": "WSB" for p in ("a", "b", "qkv", "z")},
}
FULL_ATTENTION = {
    **{f"self_attn.{p}_proj": "WSB" for p in ("q", "k", "v", "o")},
    "self_attn.q_norm": "W", "self_attn.k_norm": "W",
    "self_attn.indexer.index_qk_proj": "WSB",
    "self_attn.indexer.k_layernorm": "W", "self_attn.indexer.q_layernorm": "W",
}
MODEL_LEVEL = {
    "language_model.model.embed_tokens": "WSB", "language_model.lm_head": "WSB",
    "language_model.model.hyper_connection_mixer.hc_norm": "W",
    "language_model.model.hyper_connection_mixer.input_mix_weight_down": "WSB",
    "language_model.model.hyper_connection_mixer.input_mix_weight_up": "WSB",
}
MTP_LEVEL = {
    f"{MTP_PREFIX}{name}": spec for name, spec in {
        "fc_embedding": "WSB", "fc_hidden": "WSB",
        "pre_fc_norm_embedding": "W", "pre_fc_norm_hidden": "W",
        "hyper_connection_mixer.hc_norm": "W",
        "hyper_connection_mixer.input_mix_weight_down": "WSB",
        "hyper_connection_mixer.input_mix_weight_up": "WSB",
    }.items()
}
ROUTER_BASES = ("mlp.gate",)
_SUFFIX = {"W": ".weight", "S": ".scales", "B": ".biases"}
SHARD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*\.safetensors$")


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Report:
    path: str
    checks: list = field(default_factory=list)
    verdict: str = MISSING

    def add(self, name, status, detail="", **data) -> Check:
        check = Check(name, status, detail, data)
        self.checks.append(check)
        return check

    def by_status(self, status):
        return [c for c in self.checks if c.status == status]

    def as_dict(self) -> dict:
        return {"path": self.path, "verdict": self.verdict,
                "checks": [{"name": c.name, "status": c.status, "detail": c.detail, **c.data}
                           for c in self.checks]}

    def text(self) -> str:
        lines = [f"Flash-Next checkpoint preflight: {self.path}"]
        for check in self.checks:
            lines.append(f"  [{check.status:4s}] {check.name}: {check.detail}")
        lines.append(f"VERDICT: {self.verdict}")
        return "\n".join(lines)


def _json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def load_manifest(name_or_path=None) -> dict | None:
    path = Path(name_or_path) if name_or_path else MANIFEST_DIR / DEFAULT_MANIFEST
    if not path.is_file() and name_or_path:
        path = MANIFEST_DIR / str(name_or_path)
    return _json(path)


def expected_families(layer_types: list[str]) -> dict[str, dict]:
    """tensor base name -> spec for every layer, the model level and the MTP block."""
    out: dict[str, str] = dict(MODEL_LEVEL)
    out.update(MTP_LEVEL)
    for index, kind in enumerate(layer_types):
        prefix = f"language_model.model.layers.{index}."
        own = LINEAR_ATTENTION if kind == "linear_attention" else FULL_ATTENTION
        for base, spec in {**MLP, **own}.items():
            out[prefix + base] = spec
    for base, spec in {**MLP, **FULL_ATTENTION}.items():   # the MTP layer is full attention
        out[f"{MTP_PREFIX}layers.0.{base}"] = spec
    return out


def tensor_names(families: dict[str, str]) -> set[str]:
    names = set()
    for base, spec in families.items():
        if spec == "":
            names.add(base)
        else:
            names.update(base + _SUFFIX[ch] for ch in spec)
    return names


def expert_record_bytes(hidden: int, intermediate: int, bits: int, group: int) -> int:
    """Bytes of one expert's nine tensors (gate, up, down; weight, scales, biases)."""
    def projection(rows, cols):
        weight = rows * (cols * bits // 32) * 4          # uint32 packs 32/bits values
        scales = rows * (cols // group) * 2              # bf16
        return weight + 2 * scales                       # biases match scales
    return 2 * projection(intermediate, hidden) + projection(hidden, intermediate)


# -- individual checks -----------------------------------------------------------

def _check_files(report: Report, root: Path) -> dict:
    needed = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
    missing = [n for n in needed if not (root / n).is_file()]
    template = (root / "chat_template.jinja").is_file() or bool(
        (_json(root / "tokenizer_config.json") or {}).get("chat_template"))
    report.add("required files", FAIL if missing else PASS,
               f"missing {missing}" if missing else "config, index, tokenizer present")
    report.add("chat template", PASS if template else FAIL,
               "present" if template else "no chat_template.jinja or chat_template in tokenizer_config.json")
    report.add("generation_config.json", PASS if (root / "generation_config.json").is_file() else WARN,
               "present" if (root / "generation_config.json").is_file() else "absent")
    return {"config": _json(root / "config.json"), "index": _json(root / "model.safetensors.index.json")}


def _check_config(report: Report, config: dict | None, profile: dict | None) -> dict:
    if config is None:
        report.add("config", FAIL, "config.json is missing or not valid JSON")
        return {}
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    caps = describe_checkpoint_dict(config)
    report.add("model type", PASS if caps["compatible"] else FAIL,
               f"{config.get('model_type')} / {text.get('model_type')}"
               + ("" if caps["compatible"] else " (runtime needs qwen4_exp / qwen4_exp_text)"))
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        report.add("quantization", FAIL, "config has no quantization block")
    else:
        got = {k: quant.get(k) for k in QUANT}
        overrides = sorted(k for k, v in quant.items() if isinstance(v, dict))
        bad = {k: (got[k], QUANT[k]) for k in QUANT if got[k] != QUANT[k]} if profile else {}
        if bad:
            report.add("quantization", FAIL, "differs from the validated Q4: "
                       + ", ".join(f"{k}={g} (expected {e})" for k, (g, e) in bad.items()), **got)
        elif overrides:
            report.add("quantization", FAIL, f"per-module overrides present ({len(overrides)}); the "
                       "validated checkpoint is uniform", overrides=overrides[:5], **got)
        else:
            report.add("quantization", PASS, f"{got['bits']}-bit, group {got['group_size']}, {got['mode']}, uniform", **got)
    if profile:
        wrong = []
        for key, want in profile.items():
            if key == "model_type":
                have = config.get(key)
            elif key == "text_model_type":
                have = text.get("model_type")
            else:
                have = text.get(key)
            if have != want:
                wrong.append(f"{key}={have} (expected {want})")
        report.add("architecture profile", FAIL if wrong else PASS,
                   "; ".join(wrong) if wrong else
                   "hidden 2560, 48 layers, 512 experts, top-10, expert width 640, vocab 248320, 1 MTP layer")
    layer_types = text.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        counts = {k: layer_types.count(k) for k in set(layer_types)}
        consistent = len(layer_types) == text.get("num_hidden_layers") and set(counts) <= set(LAYER_TYPE_COUNTS)
        wrong = profile is not None and counts != LAYER_TYPE_COUNTS
        report.add("layer types", FAIL if (not consistent or wrong) else PASS,
                   f"{counts}" + ("" if consistent and not wrong else f" (expected {LAYER_TYPE_COUNTS})"))
    else:
        report.add("layer types", FAIL, "config text_config has no layer_types")
        layer_types = []
    if profile and isinstance(quant, dict):
        per_expert = expert_record_bytes(text.get("hidden_size", 0), text.get("moe_intermediate_size", 0),
                                         quant.get("bits", 0), quant.get("group_size", 0))
        stride = _slab_stride(quant.get("group_size", 0))
        ok = stride is None or stride == per_expert
        report.add("expert record size", PASS if ok else FAIL,
                   f"{per_expert:,} bytes per expert from config"
                   + ("" if stride is None else f"; slab layout stride {stride:,}"
                      + ("" if ok else " (MISMATCH: the pool, ring and prefetch sizes assume the stride)")),
                   per_expert=per_expert, slab_stride=stride)
    return {"text": text, "layer_types": layer_types, "quant": quant if isinstance(quant, dict) else {}}


def describe_checkpoint_dict(config: dict) -> dict:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    types = {config.get("model_type"), text.get("model_type")}
    return {"compatible": bool({"qwen4_exp", "qwen4_exp_text"} & {t for t in types if t})}


def _slab_stride(group_size):
    try:
        from .slab_pack import get_slab_layout

        return int(get_slab_layout(int(group_size)).record_stride)
    except Exception:
        return None


def _check_index(report: Report, root: Path, index: dict | None, layer_types: list[str],
                 profile: dict | None) -> dict:
    if index is None or not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        report.add("tensor index", FAIL, "model.safetensors.index.json is missing, invalid or empty")
        return {}
    weight_map = index["weight_map"]
    shards = sorted({v for v in weight_map.values()})
    insane = [s for s in shards if not (isinstance(s, str) and SHARD_NAME.match(s))]
    report.add("shard names", FAIL if insane else PASS,
               f"unsafe names {insane[:3]}" if insane else f"{len(shards)} shards, plain file names")
    report.add("tensor index", PASS, f"{len(weight_map):,} tensors in {len(shards)} shards",
               tensors=len(weight_map), shards=len(shards))
    if profile and layer_types:
        families = expected_families(layer_types)
        wanted_all = tensor_names(families)
        wanted = {n for n in wanted_all if not n.startswith(MTP_PREFIX)}   # MTP is optional (disabled in production)
        have = set(weight_map)
        missing = sorted(wanted - have)
        report.add("required tensors", FAIL if missing else PASS,
                   f"{len(missing)} of {len(wanted):,} missing, e.g. {missing[:3]}" if missing
                   else f"all {len(wanted):,} backbone and expert tensors named in the index",
                   missing=len(missing))
        orphans = []
        for name in have:
            for suffix, partner in ((".scales", (".weight", ".biases")), (".biases", (".weight", ".scales"))):
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    if not all(base + p in have for p in partner):
                        orphans.append(base)
        report.add("quantized triplets", FAIL if orphans else PASS,
                   f"{len(set(orphans))} tensors lack weight/scales/biases partners, e.g. {sorted(set(orphans))[:3]}"
                   if orphans else "every scales/biases tensor has its weight and partner")
        quantized_routers = sorted(
            name for name in have for base in ROUTER_BASES
            if re.search(rf"(^|\.){re.escape(base)}\.(scales|biases)$", name))
        report.add("router precision", FAIL if quantized_routers else PASS,
                   f"routers are quantized ({quantized_routers[:2]}): routing would not be exact"
                   if quantized_routers else "routers (mlp.gate) are plain weights, not quantized")
        unknown = sorted(have - wanted_all)
        vision = [n for n in unknown if n.startswith("vision_tower.")]
        ple = [n for n in unknown if ".ple." in n]
        other = [n for n in unknown if n not in set(vision) and n not in set(ple)]
        report.add("unrecognised tensors", WARN if other else INFO,
                   f"{len(other)} outside the known families, e.g. {other[:3]}" if other
                   else f"none beyond the PLE n-gram table ({len(ple)}) and the unused vision tower ({len(vision)})",
                   other=len(other))
    mtp_keys = sorted(k for k in weight_map if k.startswith(MTP_PREFIX))
    sidecar = (root / MTP_SIDECAR_NAME).is_file()
    if sidecar and mtp_keys:
        report.add("MTP", FAIL, f"both {MTP_SIDECAR_NAME} and {len(mtp_keys)} indexed MTP tensors: the loader refuses this")
    elif not mtp_keys and not sidecar:
        report.add("MTP", WARN, "no MTP tensors (MTP stays disabled in production, so this still loads)")
    else:
        missing = validate_mtp_structure(mtp_keys) if mtp_keys else []
        if mtp_keys and profile and layer_types:       # present, so it must be whole
            wanted_mtp = {n for n in tensor_names(expected_families(layer_types)) if n.startswith(MTP_PREFIX)}
            missing = missing + sorted(wanted_mtp - set(mtp_keys))
        report.add("MTP", FAIL if missing else PASS,
                   f"missing {len(missing)}, e.g. {missing[:3]}" if missing else
                   (f"{len(mtp_keys)} indexed MTP tensors, structure complete (disabled unless FLASHNEXT_NATIVE_MTP=on)"
                    if mtp_keys else "sidecar present"))
    total = (index.get("metadata") or {}).get("total_size")
    return {"weight_map": weight_map, "shards": shards, "total_size": total}


def _incomplete_files(root: Path) -> list[Path]:
    folder = root / ".cache" / "huggingface" / "download"
    return sorted(folder.glob("*.incomplete")) if folder.is_dir() else []


def _check_download(report: Report, root: Path, index_info: dict, manifest: dict | None) -> dict:
    shards = index_info.get("shards") or []
    pending = _incomplete_files(root)
    state: dict[str, dict] = {}
    by_sha = {}
    if manifest:
        for name, meta in manifest["files"].items():
            if meta.get("sha256"):
                by_sha[meta["sha256"]] = name
        expected = {n: m["size"] for n, m in manifest["files"].items()}
    else:
        expected = {}
    writing = {}
    for path in pending:
        parts = path.name.split(".")
        sha = parts[1] if len(parts) >= 4 else None
        writing[by_sha.get(sha, path.name)] = path.stat().st_size
    names = sorted(set(shards) | {n for n in expected if n.endswith(".safetensors")})
    for name in names:
        file = root / name
        size = file.stat().st_size if file.is_file() else None
        want = expected.get(name)
        if size is None:
            status = "downloading" if name in writing else "missing"
        elif want is not None and size < want:
            status = "partial"
        elif want is not None and size > want:
            status = "oversize"
        elif size == 0:
            status = "partial"
        else:
            status = "complete"
        state[name] = {"status": status, "size": size, "expected": want, "on_disk_incomplete": writing.get(name)}
    counts = {}
    for item in state.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    bad = {n: s for n, s in state.items() if s["status"] != "complete"}
    if bad:
        listing = ", ".join(f"{n}={s['status']}" for n, s in list(bad.items())[:6])
        report.add("shards", FAIL if any(s["status"] == "oversize" for s in bad.values()) else WARN,
                   f"{len(names) - len(bad)} of {len(names)} complete; {counts}; {listing}"
                   + (" ..." if len(bad) > 6 else ""), counts=counts, incomplete=sorted(bad))
    else:
        report.add("shards", PASS, f"all {len(names)} shards present at the expected size"
                   if expected else f"all {len(names)} indexed shards present", counts=counts)
    if pending:
        report.add("download in progress", WARN,
                   f"{len(pending)} .incomplete files ({sum(p.stat().st_size for p in pending) / 1e9:.1f} GB on disk, "
                   "may be pre-allocated); wait for the download to finish", files=len(pending))
    if expected:
        extra_missing = [n for n, w in expected.items() if not n.endswith(".safetensors")
                         and not (root / n).is_file() and n not in (".gitattributes",)]
        if extra_missing:
            report.add("other files", INFO, f"not present (not needed to load): {extra_missing[:5]}")
    total = index_info.get("total_size")
    present = [s["size"] for s in state.values() if s["size"] is not None]
    if total and not bad and not expected:
        slack = len(names) * 4 * 1024 * 1024
        ok = total <= sum(present) <= total + slack
        report.add("index total size", PASS if ok else WARN,
                   f"index says {total:,} tensor bytes; shards total {sum(present):,}")
    needed = sum((s["expected"] or 0) - (s["size"] or 0) for s in state.values()
                 if s["status"] in ("missing", "partial", "downloading"))
    if bad:
        free = shutil.disk_usage(root).free
        report.add("disk space", PASS if free > needed * 1.02 + 2e9 else FAIL,
                   f"{free / 1e9:.0f} GB free; about {needed / 1e9:.0f} GB still to download"
                   if needed else f"{free / 1e9:.0f} GB free", free=free, needed=needed)
    return {"state": state, "complete": not bad and not pending}


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("truncated")
        length = struct.unpack("<Q", raw)[0]
        if length > HEADER_LIMIT:
            raise ValueError("invalid header length")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def _check_headers(report: Report, root: Path, index_info: dict, text: dict, quant: dict) -> None:
    """Opt-in. Reads each shard's header only (never tensor data)."""
    shards, weight_map = index_info["shards"], index_info["weight_map"]
    found, problems, size_bad = {}, [], []
    for name in shards:
        try:
            header, start = _read_header(root / name)
        except (OSError, ValueError) as error:
            problems.append(f"{name}: {error}")
            continue
        end = max((v["data_offsets"][1] for v in header.values()), default=0)
        if start + end != (root / name).stat().st_size:
            size_bad.append(name)
        for tensor, entry in header.items():
            found[tensor] = (name, entry)
    wrong_place = [t for t, shard in weight_map.items() if t in found and found[t][0] != shard]
    missing = [t for t in weight_map if t not in found]
    extra = [t for t in found if t not in weight_map]
    report.add("shard headers", FAIL if (problems or size_bad or wrong_place or missing or extra) else PASS,
               f"{len(found):,} tensors in headers" if not (problems or size_bad or wrong_place or missing or extra)
               else f"unreadable {problems[:2]}, size mismatch {size_bad[:2]}, wrong shard {len(wrong_place)}, "
                    f"not in shard {len(missing)}, not in index {len(extra)}")
    total = sum(e["data_offsets"][1] - e["data_offsets"][0] for _, e in found.values())
    expected_total = index_info.get("total_size")
    report.add("tensor bytes", PASS if expected_total in (None, total) else FAIL,
               f"{total:,} bytes; index total_size {expected_total:,}" if expected_total else f"{total:,} bytes")
    bits, group = quant.get("bits", 4), quant.get("group_size", 32)
    bad, checked = [], 0
    for name, (_, entry) in found.items():
        if not name.endswith(".scales"):
            continue
        base = name[:-len(".scales")]
        weight = found.get(base + ".weight")
        if weight is None:
            continue
        checked += 1
        inner = weight[1]["shape"][-1] * 32 // bits
        if (weight[1]["dtype"] != "U32" or entry["shape"][-1] * group != inner
                or weight[1]["shape"][:-1] != entry["shape"][:-1]):
            bad.append(base)
    report.add("quantized layout", FAIL if bad else PASS,
               f"{len(bad)} of {checked} quantized tensors are not {bits}-bit/group-{group}, e.g. {bad[:2]}" if bad
               else f"{checked} quantized tensors are uint32-packed {bits}-bit with group {group}")
    routers = [(n, found[n][1]) for n in found if re.search(r"(^|\.)mlp\.gate\.weight$", n)]
    quantized = [n for n, e in routers if e["dtype"] in ("U32", "I8", "U8")]
    report.add("router dtype", FAIL if quantized or not routers else PASS,
               f"quantized routers {quantized[:2]}" if quantized else
               (f"{len(routers)} routers, dtypes {sorted({e['dtype'] for _, e in routers})}" if routers else "no routers found"))
    experts = [(n, e) for n, (_, e) in found.items()
               if n.endswith("layers.0.mlp.switch_mlp.gate_proj.weight") and "mtp" not in n]
    if experts and text:
        shape = experts[0][1]["shape"]
        want = [text.get("num_experts"), text.get("moe_intermediate_size"), text.get("hidden_size", 0) * bits // 32]
        report.add("expert tensor shape", PASS if shape == want else FAIL,
                   f"layer 0 gate_proj {shape}" + ("" if shape == want else f" (expected {want})"))


def _check_identity(report: Report, root: Path) -> None:
    """Opt-in. The repo's content identity (hashes shard headers only)."""
    from .checkpoint_policy import (
        VONTRA_4BIT_MTP_IDENTITY, identity_for_checkpoint, resolve_resident_experts,
    )

    identity = identity_for_checkpoint(root)
    if identity is None:
        report.add("content identity", FAIL, "could not compute (a shard header is unreadable)")
    elif identity == VONTRA_4BIT_MTP_IDENTITY:
        report.add("content identity", PASS,
                   f"matches the validated Vontra 4-bit MTP checkpoint; the resident-experts policy "
                   f"({resolve_resident_experts(None, root)}) applies")
    else:
        report.add("content identity", WARN,
                   f"{identity[:12]} differs from the validated checkpoint; the resident-experts policy "
                   f"({RESIDENT_EXPERTS_POLICY}) will NOT apply and the generic default is used")


def check_checkpoint(path, *, profile: dict | None = PROFILE, manifest="auto",
                     headers: bool = False, identity: bool = False) -> Report:
    """Run every check. ``profile=None`` validates a checkpoint on its own terms
    (structure only). ``manifest`` is "auto", a name or path, or None."""
    root = Path(os.path.expanduser(str(path)))
    report = Report(str(root))
    if not root.is_dir():
        report.add("directory", FAIL, "does not exist")
        report.verdict = MISSING
        return report
    report.add("directory", PASS, "exists")
    loaded = _check_files(report, root)
    if loaded["config"] is None or loaded["index"] is None:
        report.verdict = INCOMPLETE if _incomplete_files(root) else (
            INCOMPATIBLE if (root / "config.json").is_file() else MISSING)
        return report
    parsed = _check_config(report, loaded["config"], profile)
    index_info = _check_index(report, root, loaded["index"], parsed.get("layer_types", []), profile)
    chosen = None
    if manifest == "auto":
        candidate = load_manifest()
        if candidate and (root.name.lower() == candidate["directory_name"].lower()
                          or _index_matches(root, candidate)):
            chosen = candidate
    elif manifest:
        chosen = load_manifest(manifest)
    if index_info:
        state = _check_download(report, root, index_info, chosen)
        if chosen:
            report.add("manifest", INFO, f"{chosen['repo']} @ {chosen['revision'][:10]}")
    else:
        state = {"complete": False}
    for name, wanted in (("headers", headers), ("identity", identity)):
        if wanted and not state.get("complete"):
            report.add(name, SKIP, "refused: the download is incomplete; shards are not opened until it finishes")
    if index_info and state.get("complete"):
        if headers:
            _check_headers(report, root, index_info, parsed.get("text", {}), parsed.get("quant", {}))
        if identity:
            _check_identity(report, root)
    report.verdict = _verdict(report, state.get("complete", False))
    return report


def _index_matches(root: Path, manifest: dict) -> bool:
    meta = manifest["files"].get("model.safetensors.index.json")
    path = root / "model.safetensors.index.json"
    return bool(meta) and path.is_file() and path.stat().st_size == meta["size"]


def _verdict(report: Report, complete: bool) -> str:
    download_names = {"shards", "download in progress", "disk space", "headers", "identity",
                      "required files", "chat template"}      # repairable by finishing the download
    if any(c.status == FAIL and c.name not in download_names for c in report.checks):
        return INCOMPATIBLE
    if any(c.status == FAIL for c in report.checks if c.name == "shards"):
        return INCOMPATIBLE
    if not complete or any(c.status == FAIL for c in report.checks):
        return INCOMPLETE
    return READY


def resolve_target(target: str | None) -> str:
    """A path, an alias such as ``vontra-mtp``, or the default install location."""
    from veystrix.checkpoints import FLASHNEXT_ALIASES, model_root

    value = str(target or os.environ.get("VEYSTRIX_FLASHNEXT_MODEL", "") or "auto").strip()
    if value != "auto":
        candidate = FLASHNEXT_ALIASES.get(value.lower(), value)
        path = Path(candidate).expanduser()
        return str(path if path.is_absolute() else model_root() / path)
    return str(model_root() / DEFAULT_DIRECTORY)


DEFAULT_DIRECTORY = "Qwen3.8-Flash-Next-MLX-4bit-MTP"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", help="directory or alias (default: $VEYSTRIX_FLASHNEXT_MODEL, "
                        f"else ~/models/{DEFAULT_DIRECTORY})")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--headers", action="store_true",
                        help="also read safetensors headers (only once the download is complete)")
    parser.add_argument("--identity", action="store_true",
                        help="also compute the content identity (reads headers; only when complete)")
    parser.add_argument("--no-profile", action="store_true",
                        help="check structure only, not the validated architecture and Q4 profile")
    parser.add_argument("--manifest", help="manifest name or path (default: auto-detect)")
    args = parser.parse_args(argv)
    report = check_checkpoint(
        resolve_target(args.checkpoint), profile=None if args.no_profile else PROFILE,
        manifest=args.manifest or "auto", headers=args.headers, identity=args.identity)
    print(json.dumps(report.as_dict(), indent=1) if args.json else report.text())
    return {READY: 0, INCOMPLETE: 1}.get(report.verdict, 2)


if __name__ == "__main__":
    sys.exit(main())
