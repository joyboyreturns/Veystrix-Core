#!/usr/bin/env python3
"""Generate Veystrix's dependency-license inventory, third-party notices and SPDX SBOM.

Run it with the interpreter whose site-packages will ship (the engine runtime):

    .venv/bin/python packaging/legal/generate_legal.py --out legal
    .venv/bin/python packaging/legal/generate_legal.py --out legal --emit-texts build/Licenses

It reads installed distribution metadata only. It never imports the packages it
lists, so it cannot load a model or initialise Metal. Output is deterministic for
a given environment, so a diff shows a dependency or license change.

``licenseConcluded`` is always NOASSERTION: this tool reports what package
metadata declares, it does not make a legal conclusion.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Components that ship outside the Python environment.
EXTRA_COMPONENTS = [
    {
        "name": "Sparkle",
        "version": "2.10.1",
        "license": "MIT-style (see LICENSES/Sparkle-2.10.1.txt, includes bsdiff/sais/Orson Peters components)",
        "spdx": "NOASSERTION",
        "purl": "pkg:github/sparkle-project/Sparkle@2.10.1",
        "homepage": "https://sparkle-project.org",
        "role": "macOS app: update framework (bundled in Veystrix.app/Contents/Frameworks)",
    },
]

# License families that need a reviewer's attention if the package is bundled.
ATTENTION = {
    "MPL": "file-level copyleft: keep the notice; do not modify the files",
    "LGPL": "dynamic-link/relink obligations",
    "GPL": "strong copyleft",
    "AGPL": "network copyleft",
    "CC-BY-NC": "non-commercial restriction",
    "UNKNOWN": "license not declared in metadata: review manually",
}

# Wheels that bundle native libraries with their own licenses, beyond the declared one.
BUNDLED_NATIVE_NOTES = {
    "opencv-python": "wheel bundles FFmpeg (LGPL-2.1+) and other native libraries; review before bundling, or exclude it",
    "numpy": "wheel bundles OpenBLAS and other libraries (see numpy's own license file)",
    "scipy": "wheel bundles OpenBLAS/gfortran runtime libraries (see scipy's own license file)",
    "pillow": "wheel bundles libjpeg-turbo, freetype, libtiff, etc. (see Pillow's own license file)",
    "sentencepiece": "wheel is Apache-2.0; check bundled third-party notices",
    "mlx-metal": "contains the compiled Metal shader library (MIT, Apple Inc.)",
}

SPDX_SIMPLE = re.compile(r"^[A-Za-z0-9.\-+]+( (AND|OR|WITH) [A-Za-z0-9.\-+]+)*$")


def declared_license(dist: metadata.Distribution) -> tuple[str, str]:
    """Return (display text, spdx-or-NOASSERTION)."""
    meta = dist.metadata
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        return expr, expr if SPDX_SIMPLE.match(expr) else "NOASSERTION"
    raw = (meta.get("License") or "").strip()
    classifiers = [c.split("::")[-1].strip() for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    first = raw.splitlines()[0][:80] if raw else ""
    if len(raw) > 80 or "\n" in raw:
        first = (classifiers[0] if classifiers else first) or "see license file"
    text = first or "; ".join(classifiers) or "UNKNOWN"
    spdx = text if SPDX_SIMPLE.match(text) and text != "UNKNOWN" and " " not in text.replace(" AND ", "").replace(" OR ", "") else "NOASSERTION"
    return text, spdx


def license_files(dist: metadata.Distribution) -> list[tuple[str, str]]:
    out = []
    for f in dist.files or []:
        low = f.name.lower()
        if any(k in low for k in ("license", "licence", "copying", "notice", "authors")) and "/licenses/" in str(f).lower().replace("\\", "/") or (
            f.parent.name.endswith(".dist-info") and any(k in low for k in ("license", "licence", "copying", "notice"))
        ):
            try:
                out.append((str(f), f.read_text(errors="replace")))
            except Exception:
                pass
    return out


def attention_for(text: str, name: str) -> str:
    notes = []
    up = text.upper()
    for key, why in ATTENTION.items():
        if key in up and not (key == "GPL" and ("LGPL" in up or "AGPL" in up)):
            notes.append(f"{key}: {why}")
    if name.lower() in BUNDLED_NATIVE_NOTES:
        notes.append(BUNDLED_NATIVE_NOTES[name.lower()])
    return "; ".join(notes)


def collect(exclude: set[str]) -> list[dict]:
    rows = []
    for dist in metadata.distributions():
        meta = dist.metadata
        name = meta["Name"]
        if not name or name.lower() in exclude:
            continue
        text, spdx = declared_license(dist)
        rows.append({
            "name": name,
            "version": dist.version,
            "license": text,
            "spdx": spdx,
            "homepage": (meta.get("Home-page") or (meta.get("Project-URL") or "").split(",")[-1].strip() or ""),
            "attention": attention_for(text, name),
            "files": license_files(dist),
        })
    return sorted(rows, key=lambda r: r["name"].lower())


def write_inventory(rows: list[dict], out: Path, stamp: str) -> None:
    lines = [
        "# Dependency license inventory",
        "",
        f"Generated {stamp} by `packaging/legal/generate_legal.py` from installed package metadata.",
        "The license column is what each package *declares*. It is not a legal conclusion.",
        "Components flagged in the last column need a human check before they ship inside the app.",
        "",
        "| Package | Version | Declared license | Needs attention |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['version']} | {r['license'].replace('|', '/')} | {r['attention'] or ''} |")
    for extra in EXTRA_COMPONENTS:
        lines.append(f"| {extra['name']} (Swift/binary) | {extra['version']} | {extra['license']} | |")
    lines += ["", f"{len(rows)} Python distributions + {len(EXTRA_COMPONENTS)} non-Python component(s).", ""]
    (out / "dependency-licenses.md").write_text("\n".join(lines))


def write_sbom(rows: list[dict], out: Path, stamp: str, root_version: str) -> None:
    packages = [{
        "SPDXID": "SPDXRef-Veystrix",
        "name": "Veystrix",
        "versionInfo": root_version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "LicenseRef-Veystrix-Freeware-Draft",
        "copyrightText": "NOASSERTION",
        "comment": "Proprietary freeware (draft). Contains MIT-licensed code published earlier as MACQWEN/FlashNext; see LICENSES/MIT-1architect.txt.",
    }]
    rels = []
    for i, r in enumerate(rows):
        spdxid = f"SPDXRef-py-{i}-{re.sub(r'[^A-Za-z0-9.-]', '-', r['name'])}"
        packages.append({
            "SPDXID": spdxid, "name": r["name"], "versionInfo": r["version"],
            "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION", "licenseDeclared": r["spdx"],
            "copyrightText": "NOASSERTION",
            "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                              "referenceLocator": f"pkg:pypi/{r['name'].lower().replace('_', '-')}@{r['version']}"}],
        })
        rels.append({"spdxElementId": "SPDXRef-Veystrix", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": spdxid})
    for j, extra in enumerate(EXTRA_COMPONENTS):
        spdxid = f"SPDXRef-ext-{j}-{extra['name']}"
        packages.append({
            "SPDXID": spdxid, "name": extra["name"], "versionInfo": extra["version"],
            "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION", "licenseDeclared": extra["spdx"],
            "copyrightText": "NOASSERTION",
            "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl", "referenceLocator": extra["purl"]}],
        })
        rels.append({"spdxElementId": "SPDXRef-Veystrix", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": spdxid})
    doc = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "Veystrix runtime SBOM",
        "documentNamespace": f"https://github.com/joyboyreturns/Veystrix/sbom/{root_version}/{stamp}",
        "creationInfo": {"created": stamp + "T00:00:00Z", "creators": ["Tool: veystrix-generate-legal"]},
        "hasExtractedLicensingInfos": [{
            "licenseId": "LicenseRef-Veystrix-Freeware-Draft",
            "name": "Veystrix Freeware License (draft v0.1)",
            "extractedText": (ROOT / "legal" / "FREEWARE-LICENSE.txt").read_text(),
        }],
        "packages": packages,
        "relationships": [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-Veystrix"}] + rels,
    }
    (out / "sbom.spdx.json").write_text(json.dumps(doc, indent=2) + "\n")


def write_notices(rows: list[dict], out: Path, stamp: str, public_out: Path | None) -> None:
    mit = (ROOT / "LICENSES" / "MIT-1architect.txt").read_text().rstrip()
    sparkle = (ROOT / "LICENSES" / "Sparkle-2.10.1.txt").read_text().rstrip()
    body = [
        "# Third-party notices",
        "",
        "Veystrix is proprietary freeware (see LICENSE). It contains, links to or was derived from the",
        "third-party and earlier open-source works below. Each keeps its own license. Nothing in the",
        "Veystrix license reduces your rights under them.",
        "",
        f"_Generated {stamp}. Full license texts of the Python packages ship in the app bundle under_",
        "_`Contents/Resources/Licenses/`._",
        "",
        "## 1. Earlier MIT-licensed work (MACQWEN / FlashNext)",
        "",
        "Large parts of the Veystrix engine descend from code published under the MIT license in the",
        "repository then named MACQWEN (also FlashNext). That code was published and remains available",
        "under the MIT license; the license of copies already distributed is not withdrawn. The copyright",
        "holder named in that license is `1architect`. The license text, which must accompany the code, is:",
        "",
        "```",
        mit,
        "```",
        "",
        "Some modules also reproduce or adapt MIT-licensed code from MLX and MLX-LM (Apple Inc. and",
        "contributors), for example the compiled gate chain in `models/flashnext/compiled.py`, which",
        "follows the structure of the upstream `Qwen4ExpPLELayer`.",
        "",
        "## 2. Sparkle 2 (macOS app update framework)",
        "",
        "```",
        sparkle,
        "```",
        "",
        "## 3. Python runtime and libraries bundled with the engine helper",
        "",
        "| Package | Version | Declared license |",
        "|---|---|---|",
    ]
    for r in rows:
        body.append(f"| {r['name']} | {r['version']} | {r['license'].replace('|', '/')} |")
    body += [
        "",
        "The CPython interpreter is licensed under the PSF License Agreement.",
        "",
        "## 4. Model checkpoints are not included",
        "",
        "Veystrix does not include or redistribute model weights. Each checkpoint (for example Qwen3.8",
        "Flash-Next variants, K2-Horizon, Ternary-Bonsai-2, Qwen3.8-27B) is distributed by its publisher",
        "under that publisher's license. Read it before you download, use, modify or redistribute the checkpoint.",
        "",
    ]
    text = "\n".join(body)
    (out / "THIRD_PARTY_NOTICES.md").write_text(text)
    if public_out:
        (public_out / "THIRD_PARTY_NOTICES.md").write_text(text)


def emit_texts(rows: list[dict], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for r in rows:
        d = target / f"{r['name']}-{r['version']}"
        wrote = False
        for relname, text in r["files"]:
            d.mkdir(exist_ok=True)
            (d / Path(relname).name).write_text(text)
            wrote = True
        if not wrote:
            d.mkdir(exist_ok=True)
            (d / "NO-LICENSE-FILE-IN-DISTRIBUTION.txt").write_text(
                f"{r['name']} {r['version']} declares: {r['license']}\nNo license file was found in its metadata; review manually.\n")
    for name in ("MIT-1architect.txt", "Sparkle-2.10.1.txt"):
        (target / name).write_text((ROOT / "LICENSES" / name).read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "legal")
    ap.add_argument("--public-out", type=Path, help="also write THIRD_PARTY_NOTICES.md into this directory")
    ap.add_argument("--emit-texts", type=Path, help="copy every package's license files into this directory (for the app bundle)")
    ap.add_argument("--exclude", action="append", default=["pip", "setuptools", "wheel", "macqwen", "veystrix"],
                    help="distribution names to leave out (build tools and the project itself)")
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args(argv)
    rows = collect({e.lower() for e in args.exclude})
    args.out.mkdir(parents=True, exist_ok=True)
    write_inventory(rows, args.out, args.date)
    write_sbom(rows, args.out, args.date, args.version)
    write_notices(rows, args.out, args.date, args.public_out)
    if args.emit_texts:
        emit_texts(rows, args.emit_texts)
    flagged = [r for r in rows if r["attention"]]
    print(f"{len(rows)} distributions; {len(flagged)} need attention:")
    for r in flagged:
        print(f"  - {r['name']} {r['version']}: {r['attention']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
