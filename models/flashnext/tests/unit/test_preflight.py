"""Offline tests of the checkpoint preflight with synthetic checkpoint metadata.

Shard files are sparse placeholders, never real weights. A spy on ``open`` proves
the default checks never open a shard. No model, no MLX, no download."""
from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from models.flashnext import preflight as P
from models.flashnext.checkpoint_policy import VONTRA_4BIT_MTP_IDENTITY

SHA = {"model-00001-of-00003.safetensors": "a" * 64, "model-00002-of-00003.safetensors": "b" * 64,
       "model-00003-of-00003.safetensors": "c" * 64}
SHARD_SIZES = {"model-00001-of-00003.safetensors": 4096, "model-00002-of-00003.safetensors": 8192,
               "model-00003-of-00003.safetensors": 2048}


def layer_types(count=48):
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(count)]


def triplet(base):
    return [f"{base}.weight", f"{base}.scales", f"{base}.biases"]


def hyper(prefix):          # written out by hand, independent of preflight's tables
    names = []
    for kind in ("attn", "mlp"):
        names += triplet(f"{prefix}{kind}_hyper_connection.block_inject_weight")
        names += [f"{prefix}{kind}_hyper_connection.hc_norm.weight"]
        names += triplet(f"{prefix}{kind}_hyper_connection.input_mix_weight_down")
        names += triplet(f"{prefix}{kind}_hyper_connection.input_mix_weight_up")
    return names


def mlp(prefix):
    names = [f"{prefix}mlp.gate.weight"] + triplet(f"{prefix}mlp.shared_expert_gate")
    for part in ("gate_proj", "up_proj", "down_proj"):
        names += triplet(f"{prefix}mlp.shared_expert.{part}") + triplet(f"{prefix}mlp.switch_mlp.{part}")
    return names


def full_attention(prefix):
    names = []
    for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
        names += triplet(f"{prefix}self_attn.{part}")
    names += [f"{prefix}self_attn.q_norm.weight", f"{prefix}self_attn.k_norm.weight",
              f"{prefix}self_attn.indexer.k_layernorm.weight", f"{prefix}self_attn.indexer.q_layernorm.weight"]
    return names + triplet(f"{prefix}self_attn.indexer.index_qk_proj")


def linear_attention(prefix):
    names = [f"{prefix}linear_attn.A_log", f"{prefix}linear_attn.dt_bias",
             f"{prefix}linear_attn.conv1d.weight", f"{prefix}linear_attn.norm.weight"]
    for part in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"):
        names += triplet(f"{prefix}linear_attn.{part}")
    return names


def all_tensor_names(types):
    names = []
    for base in ("language_model.model.embed_tokens", "language_model.lm_head"):
        names += triplet(base)
    mixer = "language_model.model.hyper_connection_mixer"
    names += [f"{mixer}.hc_norm.weight"] + triplet(f"{mixer}.input_mix_weight_down") + triplet(f"{mixer}.input_mix_weight_up")
    for index, kind in enumerate(types):
        prefix = f"language_model.model.layers.{index}."
        names += hyper(prefix) + mlp(prefix) + (full_attention(prefix) if kind == "full_attention"
                                                 else linear_attention(prefix))
    mtp = "language_model.mtp."
    for base in ("fc_embedding", "fc_hidden"):
        names += triplet(mtp + base)
    names += [mtp + "pre_fc_norm_embedding.weight", mtp + "pre_fc_norm_hidden.weight",
              mtp + "hyper_connection_mixer.hc_norm.weight"]
    names += triplet(mtp + "hyper_connection_mixer.input_mix_weight_down") + triplet(mtp + "hyper_connection_mixer.input_mix_weight_up")
    names += hyper(mtp + "layers.0.") + mlp(mtp + "layers.0.") + full_attention(mtp + "layers.0.")
    # parts the preflight does not require: the PLE n-gram table and the vision tower
    names += [f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight" for i in range(3)]
    names += ["vision_tower.blocks.0.attn.qkv.weight", "vision_tower.patch_embed.proj.weight"]
    return names


def config(**text_changes):
    types = layer_types()
    text = {"model_type": "qwen4_exp_text", "hidden_size": 2560, "num_hidden_layers": 48,
            "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640, "vocab_size": 248320,
            "mtp_num_hidden_layers": 1, "layer_types": types}
    text.update(text_changes)
    return {"model_type": "qwen4_exp", "text_config": text,
            "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}


def build(root: Path, *, cfg=None, names=None, shard_sizes=None, tokenizer=True,
          drop_shards=(), partial=None, incomplete=(), extra_files=()):
    """A synthetic checkpoint: real tensor naming, sparse placeholder shards, a manifest."""
    cfg = cfg or config()
    names = list(names if names is not None else all_tensor_names(cfg["text_config"]["layer_types"]))
    shards = sorted(SHARD_SIZES)
    weight_map = {name: shards[i % 3] for i, name in enumerate(names)}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(SHARD_SIZES.values())}, "weight_map": weight_map}))
    if tokenizer:
        (root / "tokenizer.json").write_text("{}")
        (root / "tokenizer_config.json").write_text("{}")
        (root / "chat_template.jinja").write_text("x")
    (root / "generation_config.json").write_text("{}")
    for extra in extra_files:
        (root / extra).write_text("x")
    for name, size in (shard_sizes or SHARD_SIZES).items():
        if name in drop_shards:
            continue
        with (root / name).open("wb") as handle:
            handle.truncate((partial or {}).get(name, size))
    if incomplete:
        folder = root / ".cache" / "huggingface" / "download"
        folder.mkdir(parents=True)
        for name in incomplete:
            (folder / f"etag.{SHA[name]}.1234.incomplete").write_bytes(b"x" * 100)
            (folder / f"{name}.lock").write_bytes(b"")
    manifest = {"name": "synthetic", "repo": "test/synthetic", "revision": "f" * 40,
                "directory_name": root.name,
                "files": {n: {"size": s, "sha256": SHA[n]} for n, s in SHARD_SIZES.items()}}
    (root.parent / "manifest.json").write_text(json.dumps(manifest))
    return root.parent / "manifest.json"


class Base(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "ckpt"
        self.root.mkdir()

    def run_check(self, manifest=None, **options):
        return P.check_checkpoint(self.root, manifest=str(manifest) if manifest else None, **options)

    def status(self, report, name):
        return next(c.status for c in report.checks if c.name == name)

    def detail(self, report, name):
        return next(c.detail for c in report.checks if c.name == name)


class CompleteCheckpointTests(Base):
    def test_a_complete_synthetic_checkpoint_is_ready(self):
        manifest = build(self.root)
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.READY, report.text())
        for name in ("model type", "quantization", "architecture profile", "layer types",
                     "required tensors", "quantized triplets", "router precision", "shards", "MTP"):
            self.assertEqual(self.status(report, name), P.PASS, name)

    def test_the_default_checks_never_open_a_shard(self):
        manifest = build(self.root)
        opened = []
        real_open = Path.open

        def spy(path, *args, **kwargs):
            if str(path).endswith(".safetensors"):
                opened.append(str(path))
            return real_open(path, *args, **kwargs)

        shards = [self.root / n for n in SHARD_SIZES]
        for shard in shards:
            shard.chmod(0)                      # stat still works; a read would fail
        self.addCleanup(lambda: [s.chmod(0o644) for s in shards])
        with mock.patch.object(Path, "open", spy), mock.patch("builtins.open", wraps=open) as plain:
            report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.READY, report.text())
        self.assertEqual(opened, [])
        self.assertFalse([c for c in plain.call_args_list if str(c.args[0]).endswith(".safetensors")])

    def test_it_writes_nothing_into_the_checkpoint(self):
        manifest = build(self.root)
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        self.run_check(manifest)
        self.assertEqual(sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")), before)

    def test_ple_and_vision_tensors_are_not_required_but_are_reported(self):
        manifest = build(self.root)
        report = self.run_check(manifest)
        self.assertEqual(self.status(report, "unrecognised tensors"), P.INFO)
        self.assertIn("PLE", self.detail(report, "unrecognised tensors"))

    def test_an_unknown_tensor_family_is_flagged(self):
        names = all_tensor_names(layer_types()) + ["language_model.model.surprise.weight"]
        manifest = build(self.root, names=names)
        report = self.run_check(manifest)
        self.assertEqual(self.status(report, "unrecognised tensors"), P.WARN)
        self.assertEqual(report.verdict, P.READY)

    def test_json_output_and_exit_codes(self):
        manifest = build(self.root)
        out = io.StringIO()
        with redirect_stdout(out):
            code = P.main([str(self.root), "--json", "--manifest", str(manifest)])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["verdict"], "READY")
        self.assertTrue(all("status" in c for c in data["checks"]))

    def test_the_text_report_lists_every_check_and_the_verdict(self):
        manifest = build(self.root)
        text = self.run_check(manifest).text()
        self.assertIn("VERDICT: READY", text)
        self.assertIn("[PASS] router precision", text)


class IncompleteDownloadTests(Base):
    def test_a_missing_shard_is_reported_by_name_without_loading(self):
        manifest = build(self.root, drop_shards={"model-00002-of-00003.safetensors"})
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.INCOMPLETE)
        self.assertIn("model-00002-of-00003.safetensors=missing", self.detail(report, "shards"))
        self.assertIn("2 of 3 complete", self.detail(report, "shards"))

    def test_a_partial_shard_is_incomplete_not_complete(self):
        manifest = build(self.root, partial={"model-00001-of-00003.safetensors": 1000})
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.INCOMPLETE)
        self.assertIn("model-00001-of-00003.safetensors=partial", self.detail(report, "shards"))

    def test_a_zero_byte_shard_is_partial(self):
        manifest = build(self.root, partial={"model-00003-of-00003.safetensors": 0})
        self.assertIn("=partial", self.detail(self.run_check(manifest), "shards"))

    def test_an_oversize_shard_is_incompatible(self):
        manifest = build(self.root, partial={"model-00001-of-00003.safetensors": 9999})
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.INCOMPATIBLE)
        self.assertIn("oversize", self.detail(report, "shards"))

    def test_a_download_in_progress_is_recognised_and_mapped_to_its_shard(self):
        name = "model-00002-of-00003.safetensors"
        manifest = build(self.root, drop_shards={name}, incomplete=[name])
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.INCOMPLETE)
        self.assertIn(f"{name}=downloading", self.detail(report, "shards"))
        self.assertEqual(self.status(report, "download in progress"), P.WARN)

    def test_lock_files_alone_do_not_mean_a_download_is_running(self):
        manifest = build(self.root)
        folder = self.root / ".cache" / "huggingface" / "download"
        folder.mkdir(parents=True)
        (folder / "config.json.lock").write_bytes(b"")
        self.assertEqual(self.run_check(manifest).verdict, P.READY)

    def test_a_leftover_incomplete_file_blocks_ready_even_if_shards_look_whole(self):
        name = "model-00001-of-00003.safetensors"
        manifest = build(self.root, incomplete=[name])
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.INCOMPLETE)

    def test_metadata_still_downloading_is_incomplete_not_incompatible(self):
        manifest = build(self.root, incomplete=["model-00001-of-00003.safetensors"])
        (self.root / "model.safetensors.index.json").unlink()
        (self.root / "tokenizer.json").unlink()
        report = self.run_check(manifest)
        self.assertEqual(report.verdict, P.INCOMPLETE)

    def test_without_a_manifest_the_index_still_finds_missing_shards(self):
        build(self.root, drop_shards={"model-00003-of-00003.safetensors"})
        report = self.run_check()
        self.assertEqual(report.verdict, P.INCOMPLETE)
        self.assertIn("model-00003-of-00003.safetensors=missing", self.detail(report, "shards"))

    def test_without_a_manifest_a_size_total_that_disagrees_with_the_index_warns(self):
        build(self.root, shard_sizes={n: s * 1000 for n, s in SHARD_SIZES.items()})
        report = self.run_check(None)
        self.assertEqual(self.status(report, "index total size"), P.WARN)

    def test_free_disk_is_compared_with_what_is_left_to_download(self):
        manifest = build(self.root, drop_shards={"model-00002-of-00003.safetensors"})
        with mock.patch.object(P.shutil, "disk_usage", return_value=mock.Mock(free=10)):
            report = self.run_check(manifest)
        self.assertEqual(self.status(report, "disk space"), P.FAIL)

    def test_a_missing_directory_is_missing(self):
        report = P.check_checkpoint(self.root / "nowhere")
        self.assertEqual(report.verdict, P.MISSING)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(P.main([str(self.root / "nowhere")]), 2)

    def test_exit_codes(self):
        manifest = build(self.root, drop_shards={"model-00002-of-00003.safetensors"})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(P.main([str(self.root), "--manifest", str(manifest)]), 1)


class IncompatibleCheckpointTests(Base):
    def verdict_for(self, **kwargs):
        manifest = build(self.root, **kwargs)
        return self.run_check(manifest)

    def test_the_wrong_model_type_is_incompatible(self):
        cfg = config()
        cfg["model_type"] = "llama"
        cfg["text_config"]["model_type"] = "llama"
        report = self.verdict_for(cfg=cfg)
        self.assertEqual(report.verdict, P.INCOMPATIBLE)
        self.assertEqual(self.status(report, "model type"), P.FAIL)

    def test_other_quantization_is_incompatible(self):
        for change in ({"bits": 8}, {"group_size": 64}, {"mode": "mxfp4"}):
            self.setUp()
            cfg = config()
            cfg["quantization"].update(change)
            report = self.verdict_for(cfg=cfg)
            self.assertEqual(report.verdict, P.INCOMPATIBLE, change)
            self.assertEqual(self.status(report, "quantization"), P.FAIL, change)

    def test_a_missing_quantization_block_is_incompatible(self):
        cfg = config()
        del cfg["quantization"]
        self.assertEqual(self.verdict_for(cfg=cfg).verdict, P.INCOMPATIBLE)

    def test_per_module_quantization_overrides_are_incompatible(self):
        cfg = config()
        cfg["quantization"]["language_model.model.layers.0.mlp.gate"] = {"bits": 8, "group_size": 64}
        report = self.verdict_for(cfg=cfg)
        self.assertEqual(self.status(report, "quantization"), P.FAIL)
        self.assertEqual(report.verdict, P.INCOMPATIBLE)

    def test_changed_routing_or_layout_fields_are_incompatible(self):
        for change in ({"num_experts": 256}, {"num_experts_per_tok": 8}, {"hidden_size": 2048},
                       {"moe_intermediate_size": 512}, {"vocab_size": 1000}):
            self.setUp()
            report = self.verdict_for(cfg=config(**change))
            self.assertEqual(report.verdict, P.INCOMPATIBLE, change)
            self.assertEqual(self.status(report, "architecture profile"), P.FAIL, change)

    def test_layer_types_that_disagree_with_the_layer_count_are_incompatible(self):
        report = self.verdict_for(cfg=config(layer_types=layer_types(47)))
        self.assertEqual(self.status(report, "layer types"), P.FAIL)
        self.setUp()
        report = self.verdict_for(cfg=config(layer_types=["full_attention"] * 48))
        self.assertEqual(report.verdict, P.INCOMPATIBLE)

    def test_a_missing_required_tensor_is_incompatible_and_named(self):
        names = [n for n in all_tensor_names(layer_types())
                 if n != "language_model.model.layers.5.mlp.switch_mlp.up_proj.biases"]
        report = self.verdict_for(names=names)
        self.assertEqual(report.verdict, P.INCOMPATIBLE)
        self.assertEqual(self.status(report, "required tensors"), P.FAIL)
        self.assertIn("1 of", self.detail(report, "required tensors"))

    def test_a_missing_layer_is_incompatible(self):
        names = [n for n in all_tensor_names(layer_types()) if ".layers.7." not in n or ".ple." in n]
        self.assertEqual(self.verdict_for(names=names).verdict, P.INCOMPATIBLE)

    def test_an_orphan_scales_tensor_is_incompatible(self):
        names = [n for n in all_tensor_names(layer_types()) if n != "language_model.lm_head.biases"]
        report = self.verdict_for(names=names)
        self.assertEqual(self.status(report, "quantized triplets"), P.FAIL)

    def test_a_quantized_router_changes_routing_and_is_incompatible(self):
        names = all_tensor_names(layer_types()) + [
            "language_model.model.layers.3.mlp.gate.scales", "language_model.model.layers.3.mlp.gate.biases"]
        report = self.verdict_for(names=names)
        self.assertEqual(self.status(report, "router precision"), P.FAIL)
        self.assertIn("not be exact", self.detail(report, "router precision"))
        self.assertEqual(report.verdict, P.INCOMPATIBLE)

    def test_incomplete_mtp_structure_is_incompatible(self):
        names = [n for n in all_tensor_names(layer_types()) if "mtp.fc_hidden" not in n]
        report = self.verdict_for(names=names)
        self.assertEqual(self.status(report, "MTP"), P.FAIL)

    def test_both_a_sidecar_and_indexed_mtp_fail_closed(self):
        manifest = build(self.root, extra_files=["model-mtp.safetensors"])
        report = self.run_check(manifest)
        self.assertEqual(self.status(report, "MTP"), P.FAIL)
        self.assertEqual(report.verdict, P.INCOMPATIBLE)

    def test_no_mtp_at_all_still_loads_but_warns(self):
        names = [n for n in all_tensor_names(layer_types()) if "language_model.mtp." not in n]
        report = self.verdict_for(names=names)
        self.assertEqual(self.status(report, "MTP"), P.WARN)
        self.assertEqual(report.verdict, P.READY)

    def test_unsafe_shard_names_are_incompatible(self):
        manifest = build(self.root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        first = next(iter(index["weight_map"]))
        index["weight_map"][first] = "../escape.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        report = self.run_check(manifest)
        self.assertEqual(self.status(report, "shard names"), P.FAIL)
        self.assertEqual(report.verdict, P.INCOMPATIBLE)

    def test_an_empty_or_broken_index_is_not_ready(self):
        manifest = build(self.root)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        self.assertEqual(self.run_check(manifest).verdict, P.INCOMPATIBLE)
        (self.root / "model.safetensors.index.json").write_text("{not json")
        self.assertNotEqual(self.run_check(manifest).verdict, P.READY)

    def test_missing_tokenizer_files_are_incompatible(self):
        manifest = build(self.root, tokenizer=False)
        report = self.run_check(manifest)
        self.assertEqual(self.status(report, "required files"), P.FAIL)
        self.assertEqual(report.verdict, P.INCOMPLETE)     # index and config exist, tokenizer absent: a repairable gap

    def test_a_missing_chat_template_fails(self):
        manifest = build(self.root)
        (self.root / "chat_template.jinja").unlink()
        self.assertEqual(self.status(self.run_check(manifest), "chat template"), P.FAIL)

    def test_a_chat_template_in_the_tokenizer_config_is_enough(self):
        manifest = build(self.root)
        (self.root / "chat_template.jinja").unlink()
        (self.root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{x}}"}))
        self.assertEqual(self.status(self.run_check(manifest), "chat template"), P.PASS)

    def test_no_profile_mode_checks_structure_only(self):
        cfg = config(num_experts=8, hidden_size=64)
        cfg["quantization"]["bits"] = 8
        manifest = build(self.root, cfg=cfg)
        report = self.run_check(manifest, profile=None)
        self.assertNotIn("architecture profile", [c.name for c in report.checks])
        self.assertEqual(self.status(report, "quantization"), P.PASS)


def write_shard(path: Path, tensors: dict, pad_to: int | None = None):
    """A real safetensors container: header JSON plus zero bytes of the stated sizes."""
    header, offset = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
        if pad_to:
            handle.write(b"\0" * pad_to)
    return offset


class HeaderAndIdentityTests(Base):
    """Opt-in checks that read shard headers only. A tiny checkpoint, real containers."""

    def make(self, *, router_dtype="BF16", scales_cols=2, weight_dtype="U32", pad=0, drop_tensor=None):
        cfg = config(num_experts=4, hidden_size=64, moe_intermediate_size=32)
        tensors = {
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": (weight_dtype, [4, 32, 8], 4 * 32 * 8 * 4),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": ("BF16", [4, 32, scales_cols], 4 * 32 * scales_cols * 2),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": ("BF16", [4, 32, scales_cols], 4 * 32 * scales_cols * 2),
            "language_model.model.layers.0.mlp.gate.weight": (router_dtype, [4, 64], 4 * 64 * 2),
        }
        if drop_tensor:
            tensors.pop(drop_tensor)
        total = write_shard(self.root / "model-00001-of-00001.safetensors", tensors, pad)
        (self.root / "config.json").write_text(json.dumps(cfg))
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total},
            "weight_map": {n: "model-00001-of-00001.safetensors" for n in
                           list(tensors) + ([drop_tensor] if drop_tensor else [])}}))
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (self.root / name).write_text("{}")

    def run_headers(self, **options):
        return P.check_checkpoint(self.root, profile=None, manifest=None, headers=True, **options)

    def test_a_consistent_tiny_checkpoint_passes_the_header_checks(self):
        self.make()
        report = self.run_headers()
        for name in ("shard headers", "tensor bytes", "quantized layout", "router dtype", "expert tensor shape"):
            self.assertEqual(self.status(report, name), P.PASS, report.text())
        self.assertEqual(report.verdict, P.READY)

    def test_a_quantized_router_dtype_fails(self):
        self.make(router_dtype="U32")
        self.assertEqual(self.status(self.run_headers(), "router dtype"), P.FAIL)

    def test_a_scales_shape_that_disagrees_with_the_group_size_fails(self):
        self.make(scales_cols=4)
        self.assertEqual(self.status(self.run_headers(), "quantized layout"), P.FAIL)

    def test_a_non_uint32_weight_fails_the_q4_layout(self):
        self.make(weight_dtype="BF16")
        self.assertEqual(self.status(self.run_headers(), "quantized layout"), P.FAIL)

    def test_a_truncated_or_padded_shard_fails(self):
        self.make(pad=7)
        self.assertEqual(self.status(self.run_headers(), "shard headers"), P.FAIL)

    def test_a_tensor_listed_in_the_index_but_absent_from_the_shard_fails(self):
        self.make(drop_tensor="language_model.model.layers.0.mlp.gate.weight")
        self.assertEqual(self.status(self.run_headers(), "shard headers"), P.FAIL)

    def test_an_unreadable_header_is_reported_not_raised(self):
        self.make()
        (self.root / "model-00001-of-00001.safetensors").write_bytes(b"abc")
        self.assertEqual(self.status(self.run_headers(), "shard headers"), P.FAIL)

    def test_headers_and_identity_are_refused_while_the_download_is_incomplete(self):
        manifest = build(self.root, drop_shards={"model-00002-of-00003.safetensors"},
                         incomplete=["model-00002-of-00003.safetensors"])
        opened = []
        real_open = Path.open
        with mock.patch.object(Path, "open", lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1]):
            report = P.check_checkpoint(self.root, manifest=str(manifest), headers=True, identity=True)
        self.assertEqual(report.verdict, P.INCOMPLETE)
        self.assertEqual(self.status(report, "headers"), P.SKIP)
        self.assertEqual(self.status(report, "identity"), P.SKIP)
        self.assertFalse([p for p in opened if p.endswith(".safetensors")])

    def test_identity_matching_the_validated_checkpoint_passes(self):
        self.make()
        with mock.patch("models.flashnext.checkpoint_policy.identity_for_checkpoint",
                        return_value=VONTRA_4BIT_MTP_IDENTITY):
            report = P.check_checkpoint(self.root, profile=None, manifest=None, identity=True)
        self.assertEqual(self.status(report, "content identity"), P.PASS)
        self.assertIn("policy (8) applies", self.detail(report, "content identity"))

    def test_a_different_identity_warns_that_the_resident_experts_policy_will_not_apply(self):
        self.make()
        with mock.patch("models.flashnext.checkpoint_policy.identity_for_checkpoint", return_value="0" * 64):
            report = P.check_checkpoint(self.root, profile=None, manifest=None, identity=True)
        self.assertEqual(self.status(report, "content identity"), P.WARN)
        self.assertIn("will NOT apply", self.detail(report, "content identity"))

    def test_the_real_identity_function_reads_only_headers_of_a_tiny_checkpoint(self):
        self.make()
        report = P.check_checkpoint(self.root, profile=None, manifest=None, identity=True)
        self.assertEqual(self.status(report, "content identity"), P.WARN)   # not the Vontra hash


class KnowledgeTests(unittest.TestCase):
    """Facts about the validated checkpoint the checker relies on."""

    def test_expert_record_size_matches_the_slab_layout_and_the_documented_value(self):
        self.assertEqual(P.expert_record_bytes(2560, 640, 4, 32), 3_072_000)
        from models.flashnext.slab_pack import get_slab_layout

        self.assertEqual(get_slab_layout(32).record_stride, 3_072_000)

    def test_the_family_tables_name_the_number_of_tensors_validated_against_the_real_index(self):
        # 3,017 backbone, expert and MTP tensors, checked against the real 3,747-tensor index
        # (the rest is the PLE n-gram table and the vision tower); 76 of them are MTP.
        names = P.tensor_names(P.expected_families(layer_types()))
        self.assertEqual(len(names), 3017)
        self.assertEqual(sum(n.startswith("language_model.mtp.") for n in names), 76)

    def test_family_tables_match_the_independent_builder_in_this_file(self):
        built = {n for n in all_tensor_names(layer_types()) if ".ple." not in n and not n.startswith("vision_tower.")}
        self.assertEqual(P.tensor_names(P.expected_families(layer_types())), built)

    def test_linear_and_full_attention_layers_have_their_own_tensors(self):
        names = P.tensor_names(P.expected_families(layer_types()))
        self.assertIn("language_model.model.layers.0.linear_attn.A_log", names)
        self.assertIn("language_model.model.layers.3.self_attn.q_proj.weight", names)
        self.assertNotIn("language_model.model.layers.0.self_attn.q_proj.weight", names)

    def test_the_router_is_a_plain_weight_in_every_layer_and_the_mtp_layer(self):
        families = P.expected_families(layer_types())
        routers = {b: s for b, s in families.items() if b.endswith(".mlp.gate")}
        self.assertEqual(len(routers), 49)
        self.assertEqual(set(routers.values()), {"W"})

    def test_the_shipped_manifest_describes_the_validated_revision(self):
        manifest = P.load_manifest()
        files = manifest["files"]
        shards = {n: m for n, m in files.items() if n.endswith(".safetensors")}
        self.assertEqual(len(shards), 22)
        self.assertEqual(sum(m["size"] for m in files.values()), 113_233_048_033)
        self.assertEqual(manifest["revision"], "2b170fa6309d5d1ee380b35636075fac7945f286")
        self.assertTrue(all(len(m["sha256"]) == 64 for m in shards.values()))
        self.assertEqual(manifest["directory_name"], "Qwen3.8-Flash-Next-MLX-4bit-MTP")
        self.assertIn("model.safetensors.index.json", files)

    def test_resolve_target_accepts_paths_aliases_and_the_default(self):
        with mock.patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": "/models", "VEYSTRIX_FLASHNEXT_MODEL": ""}):
            self.assertEqual(P.resolve_target("/x/y"), "/x/y")
            self.assertEqual(P.resolve_target("vontra-mtp"), "/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
            self.assertEqual(P.resolve_target(None), "/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
        with mock.patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": "/models", "VEYSTRIX_FLASHNEXT_MODEL": "/z"}):
            self.assertEqual(P.resolve_target(None), "/z")

    def test_the_alias_target_equals_the_download_directory(self):
        from veystrix.checkpoints import FLASHNEXT_ALIASES

        self.assertEqual(FLASHNEXT_ALIASES["vontra-mtp"], P.load_manifest()["directory_name"])

    def test_the_preflight_leaves_the_runtime_untouched(self):
        root = Path(P.__file__).parent
        for name in ("loader.py", "checkpoint_compat.py", "checkpoint_policy.py", "expert_cache.py",
                     "prefill.py", "prefill_ring.py", "expert_pool.py"):
            self.assertNotIn("preflight", (root / name).read_text(), name)

    def test_the_module_imports_no_mlx(self):
        text = Path(P.__file__).read_text()
        self.assertEqual([l for l in text.splitlines() if l.startswith(("import mlx", "from mlx"))], [])


if __name__ == "__main__":
    unittest.main()
