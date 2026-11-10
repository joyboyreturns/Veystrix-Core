#!/usr/bin/env python3
"""Validate a Sparkle appcast before it is published.

Structure checks (always): well-formed XML, every item has an https enclosure URL on an allowed host, a positive
length, a base64 sparkle:edSignature that decodes to 64 bytes, a sparkle:version and shortVersionString, and build
numbers strictly decrease from the first (newest) item to the last.
Cryptographic check (optional): --public-key (base64, 32 bytes) with --archive-dir verifies each enclosure's
Ed25519 signature over the archive bytes, using a self-contained RFC 8032 verifier, so a release pipeline can
prove the feed is consistent with the *public* key that ships in Info.plist (SUPublicEDKey).
Usage: validate_appcast.py appcast.xml [--allowed-host github.com ...] [--public-key B64 --archive-dir DIR]
"""
from __future__ import annotations
import argparse, base64, hashlib, sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

SPARKLE = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"

# ---- RFC 8032 Ed25519 verification (pure Python; verification only, no secret material involved)
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _inv(x): return pow(x, _P - 2, _P)


def _xrecover(y):
    xx = (y * y - 1) * _inv(_D * y * y + 1)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P:
        x = (x * _I) % _P
    return _P - x if x & 1 else x


_BY = 4 * _inv(5) % _P
_B = (_xrecover(_BY), _BY)


def _add(a, b):
    (x1, y1), (x2, y2) = a, b
    t = _D * x1 * x2 * y1 * y2
    return ((x1 * y2 + x2 * y1) * _inv(1 + t) % _P, (y1 * y2 + x1 * x2) * _inv(1 - t) % _P)


def _mul(p, e):
    q = (0, 1)
    while e:
        if e & 1:
            q = _add(q, p)
        p = _add(p, p)
        e >>= 1
    return q


def _decode_point(s: bytes):
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    x = _xrecover(y)
    if (x & 1) != (s[31] >> 7):
        x = _P - x
    pt = (x, y)
    if (-x * x + y * y - 1 - _D * x * x * y * y) % _P:
        raise ValueError("point not on curve")
    return pt


def _encode_point(p) -> bytes:
    x, y = p
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    try:
        a = _decode_point(public_key)
        r = _decode_point(signature[:32])
    except ValueError:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public_key + message).digest(), "little") % _L
    return _encode_point(_mul(_B, s)) == _encode_point(_add(r, _mul(a, h)))


def validate(path: Path, allowed_hosts: set[str], public_key: bytes | None = None, archive_dir: Path | None = None) -> list[str]:
    problems: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return [f"not well-formed XML: {e}"]
    items = root.findall("./channel/item")
    if not items:
        return ["no <item> entries"]
    builds: list[int] = []
    for n, item in enumerate(items, 1):
        tag = f"item {n}"
        enc = item.find("enclosure")
        if enc is None:
            problems.append(f"{tag}: no <enclosure>")
            continue
        url = urlparse(enc.get("url", ""))
        if url.scheme != "https":
            problems.append(f"{tag}: enclosure URL is not https ({enc.get('url')})")
        if allowed_hosts and url.hostname not in allowed_hosts:
            problems.append(f"{tag}: host {url.hostname} is not in the allowed set {sorted(allowed_hosts)}")
        try:
            if int(enc.get("length", "0")) <= 0:
                problems.append(f"{tag}: enclosure length must be positive")
        except ValueError:
            problems.append(f"{tag}: enclosure length is not an integer")
        sig = enc.get(SPARKLE + "edSignature")
        raw = b""
        if not sig:
            problems.append(f"{tag}: missing sparkle:edSignature (updates must be EdDSA-signed)")
        else:
            try:
                raw = base64.b64decode(sig, validate=True)
                if len(raw) != 64:
                    problems.append(f"{tag}: edSignature decodes to {len(raw)} bytes, expected 64")
            except ValueError:
                problems.append(f"{tag}: edSignature is not valid base64")
        version = enc.get(SPARKLE + "version") or (item.findtext(SPARKLE + "version") or "")
        short = enc.get(SPARKLE + "shortVersionString") or (item.findtext(SPARKLE + "shortVersionString") or "")
        if not version.isdigit():
            problems.append(f"{tag}: sparkle:version must be an integer build number (got {version!r})")
        else:
            builds.append(int(version))
        if not short:
            problems.append(f"{tag}: missing sparkle:shortVersionString")
        if public_key and archive_dir and raw and url.path:
            archive = archive_dir / Path(url.path).name
            if not archive.is_file():
                problems.append(f"{tag}: archive {archive.name} not found in {archive_dir}")
            elif not ed25519_verify(public_key, archive.read_bytes(), raw):
                problems.append(f"{tag}: EdDSA signature does NOT verify against the supplied public key")
    if builds != sorted(builds, reverse=True) or len(set(builds)) != len(builds):
        problems.append(f"build numbers must strictly decrease from newest to oldest: {builds}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("appcast")
    ap.add_argument("--allowed-host", action="append", default=["github.com"])
    ap.add_argument("--public-key")
    ap.add_argument("--archive-dir")
    a = ap.parse_args(argv)
    key = base64.b64decode(a.public_key, validate=True) if a.public_key else None
    problems = validate(Path(a.appcast), set(a.allowed_host), key, Path(a.archive_dir) if a.archive_dir else None)
    for p in problems:
        print("FAIL:", p)
    print("appcast OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
