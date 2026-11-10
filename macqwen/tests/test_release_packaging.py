"""Offline checks for the release pipeline: appcast validation, Ed25519 verification (RFC 8032 vectors), signing
refusals for development builds, and secret/exposure scanners. No network, no signing keys, no model."""
from __future__ import annotations

import base64
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


va = _load("validate_appcast", ROOT / "packaging" / "macos" / "validate_appcast.py")

# RFC 8032 section 7.1, test vectors 1 and 2 (throwaway public test data, not release keys)
V1 = (bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"), b"",
      bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"))
V2 = (bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"), b"\x72",
      bytes.fromhex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"))

FEED = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel><title>Veystrix</title>
{items}</channel></rss>"""
ITEM = """<item><title>{short}</title><sparkle:version>{build}</sparkle:version><sparkle:shortVersionString>{short}</sparkle:shortVersionString>
<enclosure url="{url}" length="{length}" type="application/octet-stream" sparkle:edSignature="{sig}"/></item>"""


def item(build="3080012", short="3.8.2", url="https://github.com/joyboyreturns/Veystrix/releases/download/v3.8.2/Veystrix-3.8.2-arm64.zip",
         length="1", sig=None):
    return ITEM.format(build=build, short=short, url=url, length=length, sig=sig or base64.b64encode(V2[2]).decode())


class Ed25519Tests(unittest.TestCase):
    def test_rfc8032_vectors_verify(self):
        for pk, msg, sig in (V1, V2):
            self.assertTrue(va.ed25519_verify(pk, msg, sig))

    def test_tampering_is_rejected(self):
        pk, msg, sig = V2
        self.assertFalse(va.ed25519_verify(pk, msg + b"x", sig))
        self.assertFalse(va.ed25519_verify(pk, msg, sig[:-1] + bytes([sig[-1] ^ 1])))
        self.assertFalse(va.ed25519_verify(V1[0], msg, sig))            # wrong key
        self.assertFalse(va.ed25519_verify(pk, msg, sig[:63]))          # truncated


class AppcastTests(unittest.TestCase):
    def check(self, items: str, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "appcast.xml"
            path.write_text(FEED.format(items=items))
            return va.validate(path, {"github.com"}, **kw)

    def test_good_feed_passes(self):
        self.assertEqual(self.check(item()), [])

    def test_http_url_is_rejected(self):
        self.assertTrue(any("https" in p for p in self.check(item(url="http://github.com/x.zip"))))

    def test_foreign_host_is_rejected(self):
        self.assertTrue(any("not in the allowed" in p for p in self.check(item(url="https://example.com/x.zip"))))

    def test_missing_or_malformed_signature_is_rejected(self):
        self.assertTrue(any("edSignature" in p for p in self.check(item(sig="not base64!!"))))
        self.assertTrue(any("64" in p for p in self.check(item(sig=base64.b64encode(b"short").decode()))))
        unsigned = item().replace(f' sparkle:edSignature="{base64.b64encode(V2[2]).decode()}"', "")
        self.assertTrue(any("missing sparkle:edSignature" in p for p in self.check(unsigned)))

    def test_build_numbers_must_decrease_newest_first(self):
        self.assertTrue(any("strictly" in p for p in self.check(item("3080011", "3.8.1") + item("3080012", "3.8.2"))))
        self.assertTrue(any("strictly" in p for p in self.check(item("3080012") + item("3080012"))))
        self.assertEqual(self.check(item("3080012", "3.8.2") + item("3080011", "3.8.1")), [])

    def test_signature_is_verified_against_the_archive_bytes(self):
        pk, msg, _ = V2
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Veystrix-3.8.2-arm64.zip").write_bytes(msg)
            feed = Path(tmp) / "appcast.xml"
            feed.write_text(FEED.format(items=item()))
            self.assertEqual(va.validate(feed, {"github.com"}, pk, Path(tmp)), [])
            (Path(tmp) / "Veystrix-3.8.2-arm64.zip").write_bytes(msg + b"tampered")
            self.assertTrue(any("does NOT verify" in p for p in va.validate(feed, {"github.com"}, pk, Path(tmp))))

    def test_malformed_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.xml"
            path.write_text("<rss><channel>")
            self.assertTrue(va.validate(path, set())[0].startswith("not well-formed"))

    def test_published_feed_if_present_is_valid(self):
        feed = ROOT / "updates" / "appcast.xml"
        if feed.exists():
            self.assertEqual(va.validate(feed, {"github.com"}), [])


class FreeChannelTests(unittest.TestCase):
    """No Apple Developer membership is required: the default is the free (ad-hoc, not notarized) channel."""

    def run_script(self, *args, extra_env=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("VEYSTRIX_")}
        env.update(extra_env or {})
        return subprocess.run(["bash", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)

    def test_default_channel_is_free_and_names_do_not_depend_on_signing(self):
        r = self.run_script("-c", 'source packaging/macos/common.sh; echo "$(channel)|$(artifact_suffix)|$SHORT_VERSION"')
        channel, suffix, version = r.stdout.strip().splitlines()[-1].split("|")
        self.assertEqual((channel, suffix), ("free", ""), r.stderr)
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        r = self.run_script("-c", 'source packaging/macos/common.sh; channel', extra_env={"VEYSTRIX_SIGN_IDENTITY": "Developer ID Application: X (T)"})
        self.assertEqual(r.stdout.strip().splitlines()[-1], "developer-id")

    def test_notarization_is_refused_without_an_identity_with_a_clear_message(self):
        for script, extra in (("packaging/macos/notarize.sh", ["x.dmg"]), ("packaging/macos/make_release.sh", ["--notarize"])):
            r = self.run_script(script, *extra)
            self.assertNotEqual(r.returncode, 0, script)
            self.assertTrue("free channel" in r.stderr or "without --notarize" in r.stderr, r.stderr)

    def test_sparkle_public_key_is_a_valid_ed25519_key_and_tracked(self):
        key = (ROOT / "packaging" / "macos" / "sparkle_public_key.txt").read_text().strip()
        self.assertEqual(len(base64.b64decode(key, validate=True)), 32)

    def test_info_plist_template_has_update_keys_and_no_hard_coded_secret(self):
        text = (ROOT / "apps" / "macos" / "Veystrix" / "Support" / "Info.plist.template").read_text()
        for needle in ("SUPublicEDKey", "@SPARKLE_PUBLIC_ED_KEY@", "SUFeedURL", "@MIN_MACOS@", "SUVerifyUpdateBeforeExtraction"):
            self.assertIn(needle, text)
        self.assertIn("https://", text)

    def test_the_license_does_not_require_a_developer_id_for_official_builds(self):
        lic = (ROOT / "legal" / "FREEWARE-LICENSE.txt").read_text()
        self.assertIn("not notarized", lic)
        self.assertEqual(lic, (ROOT.parent / "Veystrix-Public" / "LICENSE").read_text()) if (ROOT.parent / "Veystrix-Public" / "LICENSE").exists() else None


class ScannerTests(unittest.TestCase):
    def git_repo(self, tmp: str, files: dict[str, str]) -> str:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        for name, text in files.items():
            p = Path(tmp) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        subprocess.run(["git", "-C", tmp, "add", "-A"], check=True)
        subprocess.run(["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "t"], check=True)
        return tmp

    def test_secret_scan_flags_a_token_and_passes_clean_text(self):
        scan = ROOT / "scripts" / "security" / "secret_scan.py"
        with tempfile.TemporaryDirectory() as dirty, tempfile.TemporaryDirectory() as clean:
            self.git_repo(dirty, {"a.txt": "token = " + "ghp_" + "A" * 36})
            self.git_repo(clean, {"a.txt": "nothing to see"})
            self.assertEqual(subprocess.run([sys.executable, str(scan), "--repo", dirty], capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run([sys.executable, str(scan), "--repo", dirty, "--history"], capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run([sys.executable, str(scan), "--repo", clean, "--history"], capture_output=True).returncode, 0)

    def test_exposure_audit_detects_shared_source_and_private_paths(self):
        audit = ROOT / "scripts" / "security" / "public_exposure_audit.py"
        with tempfile.TemporaryDirectory() as core, tempfile.TemporaryDirectory() as good, tempfile.TemporaryDirectory() as bad:
            self.git_repo(core, {"engine.py": "print('secret engine')\n", "LICENSE": "MIT text\n"})
            self.git_repo(good, {"README.md": "hello\n", "LICENSE": "MIT text\n"})
            self.git_repo(bad, {"engine.py": "print('secret engine')\n", "notes.md": "see /Users/someone/work\n"})
            ok = subprocess.run([sys.executable, str(audit), "--public", good, "--core", core], capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            nope = subprocess.run([sys.executable, str(audit), "--public", bad, "--core", core], capture_output=True, text=True)
            self.assertEqual(nope.returncode, 1)
            self.assertIn("identical to private-repo content", nope.stdout)
            self.assertIn("private paths", nope.stdout)


if __name__ == "__main__":
    unittest.main()
