from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORE_REPOSITORY = "https://github.com/joyboyreturns/Veystrix-Core"      # private: source, history
PUBLIC_REPOSITORY = "https://github.com/joyboyreturns/Veystrix"          # public: README, releases, update feed


class DistributionTests(unittest.TestCase):
    def test_readme_uses_the_core_clone_url_and_links_the_public_home(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn(CORE_REPOSITORY + ".git", readme)
        self.assertIn(PUBLIC_REPOSITORY, readme)

    def test_package_metadata_points_at_the_public_repository_only(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        urls = data["project"]["urls"]
        self.assertEqual(urls["Homepage"], PUBLIC_REPOSITORY)
        self.assertEqual(urls["Issues"], PUBLIC_REPOSITORY + "/issues")
        # Package metadata ships inside wheels: it must not advertise the private repository.
        self.assertFalse([u for u in urls.values() if "Veystrix-Core" in u])
        changelog = (ROOT / "CHANGELOG.md").read_text()
        release = data["project"]["version"].split(".dev")[0]
        self.assertIn(f"## Veystrix {release} ", changelog)

    def test_shared_runtime_dependencies_are_pinned_in_project_metadata(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            set(data["project"]["dependencies"]),
            {
                "mlx==0.32.2",
                "mlx-lm==0.31.3",
                "mlx-vlm==0.6.17",
                "transformers==5.16.1",
                "numpy==2.5.2",
                "requests==2.34.2",
                "huggingface-hub==1.29.0",
            },
        )

    def test_ci_and_release_automation_are_separate(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
        self.assertIn("models/flashnext", ci)
        self.assertIn("models.nemotron35", ci)
        for removed in ("models/qwen27b", "models/k2_horizon", "models/bonsai2"):
            self.assertNotIn(removed, ci)
            self.assertNotIn(removed, release)
        self.assertNotIn("gh release create", ci)
        self.assertIn("tags: ['v*']", release)
        self.assertIn("needs: test", release)
        self.assertIn("gh release create", release)
        # Publishing is gated: an approval environment, a token scoped to the PUBLIC repo, and a signed-only condition.
        self.assertIn("environment: release", release)
        self.assertIn("-R joyboyreturns/Veystrix", release)
        self.assertIn("needs.build.outputs.publish == 'true'", release)
        self.assertIn("--draft", release)                     # the public release is never published automatically
        self.assertIn("PUBLIC_RELEASES_TOKEN", release)
        self.assertNotIn("github.token", release)            # the default token must never publish
        self.assertIn("qwen3.8-flash-next", ci)
        self.assertIn("swift test", ci)


if __name__ == "__main__":
    unittest.main()
