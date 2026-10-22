"""Regression checks for CI selection and the Cargo commands it executes."""

import json
import os
import subprocess
import unittest
from unittest.mock import patch

import ci_scope


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads(
            subprocess.check_output(
                ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
                cwd=ci_scope.ROOT,
                text=True,
            )
        )

    def select(self, *files):
        return ci_scope.select(self.meta, list(files))

    def test_isolated_module(self):
        scope = self.select("crates/sci-filter/src/lib.rs")
        self.assertEqual(scope["packages"], ["sci-filter"])
        self.assertEqual(scope["coverage"], ["sci-filter"])
        self.assertIn("filter", scope["test"])
        self.assertIn("filter_interop", scope["test"])
        self.assertNotIn("interpolation", scope["test"])

    def test_dependents(self):
        scope = self.select("crates/sci-special/src/lib.rs")
        self.assertIn("sci-special", scope["packages"])
        self.assertIn("sci-stats", scope["packages"])
        self.assertIn("statistics", scope["test"])

    def test_core_reaches_every_crate(self):
        scope = self.select("crates/sci-core/src/lib.rs")
        self.assertEqual(len(scope["packages"]), len(self.meta["packages"]) - 1)
        self.assertIn("core", scope["test"])

    def test_test_and_fixture_only_changes(self):
        for path in [
            "tests/interpolation.rs",
            "tests/fixtures/interpolation_expected.csv",
        ]:
            with self.subTest(path=path):
                scope = self.select(path)
                self.assertEqual(scope["test"], ["interpolation"])
                self.assertEqual(scope["features"], ["interpolate"])
                self.assertEqual(scope["packages"], [])
                self.assertEqual(scope["coverage"], ["sci-interpolate"])

    def test_shared_and_unknown_fixture_fallback(self):
        for path in [
            "Cargo.lock",
            "Cargo.toml",
            "codecov.yml",
            "build.rs",
            "src/lib.rs",
            ".github/workflows/test.yml",
            "tests/common/mod.rs",
            "benches/common/mod.rs",
            "tests/fixtures/unknown.csv",
            "crates/removed/src/lib.rs",
        ]:
            with self.subTest(path=path):
                self.assertTrue(self.select(path)["full"])
        self.assertTrue(ci_scope.select(self.meta, None)["full"])

    def test_docs_and_empty_diff_skip(self):
        self.assertFalse(self.select("README.md")["active"])
        self.assertFalse(self.select()["active"])

    def test_reference_generator_does_not_force_workspace_run(self):
        scope = self.select(
            "crates/sci-filter/src/lib.rs", "tests/scipy/generate_filter.py"
        )
        self.assertFalse(scope["full"])
        self.assertEqual(scope["packages"], ["sci-filter"])

    def test_unknown_fixture_with_other_changes(self):
        self.assertTrue(
            self.select("tests/interpolation.rs", "tests/fixtures/unknown.csv")["full"]
        )

    def test_inventory_consumer(self):
        scope = self.select("function_coverages/scipy_function_list/scipy.special.md")
        self.assertIn("special_properties", scope["test"])

    def test_multiple_modules_and_adapters(self):
        scope = self.select(
            "crates/sci-filter/src/lib.rs", "crates/sci-linalg/src/lib.rs"
        )
        self.assertIn("sci-filter", scope["packages"])
        self.assertIn("sci-linalg", scope["packages"])
        self.assertIn("linalg-ndarray", scope["features"])

    @patch("ci_scope.subprocess.run")
    def test_scoped_commands(self, run):
        scope = self.select("tests/interpolation.rs")
        ci_scope.run(scope, "test")
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "cargo",
                    "test",
                    "-p",
                    "sci",
                    "--no-default-features",
                    "--features",
                    "interpolate",
                    "--lib",
                    "--test",
                    "interpolation",
                    "--locked",
                ],
                [
                    "cargo",
                    "test",
                    "-p",
                    "sci",
                    "--no-default-features",
                    "--features",
                    "interpolate",
                    "--doc",
                    "--locked",
                ],
            ],
        )

    @patch("ci_scope.subprocess.run")
    def test_full_and_inactive_commands(self, run):
        ci_scope.run({"full": True, "active": True}, "clippy")
        self.assertEqual(
            run.call_args.args[0],
            [
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
        )
        run.reset_mock()
        ci_scope.run(self.select("README.md"), "test")
        run.assert_not_called()

    def test_coverage_accumulates_before_reporting(self):
        with (
            patch("ci_scope.subprocess.run") as run,
            patch("ci_scope.metadata", return_value=self.meta),
            patch("ci_scope.write_reports", return_value=["sci-filter"]) as reports,
            patch("ci_scope.output_flags") as output,
        ):
            ci_scope.run(self.select("crates/sci-filter/src/lib.rs"), "coverage")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(
                commands[0], ["cargo", "+stable", "llvm-cov", "clean", "--workspace"]
            )
            self.assertTrue(all("--no-report" in c for c in commands[1:-1]))
            self.assertEqual(commands[-1][3], "report")
            self.assertEqual(reports.call_args.args[2], {"sci-filter"})
            output.assert_called_once_with(["sci-filter"])

    def test_fixture_change_refreshes_complete_module_suite(self):
        with (
            patch("ci_scope.subprocess.run") as run,
            patch("ci_scope.metadata", return_value=self.meta),
            patch(
                "ci_scope.write_reports", return_value=["sci-interpolate"]
            ) as reports,
            patch("ci_scope.output_flags"),
        ):
            # The interpolation crate also has coverage in numerics and coverage.
            scope = self.select("tests/fixtures/interpolation_expected.csv")
            self.assertEqual(scope["test"], ["interpolation"])
            ci_scope.run(scope, "coverage")
            commands = [call.args[0] for call in run.call_args_list]
            facade = next(c for c in commands if "sci" in c)
            for target in ["interpolation", "numerics", "coverage"]:
                self.assertIn(target, facade)
            self.assertEqual(reports.call_args.args[2], {"sci-interpolate"})

    def test_full_coverage_reports_every_package(self):
        with (
            patch("ci_scope.subprocess.run") as run,
            patch("ci_scope.metadata", return_value=self.meta),
            patch("ci_scope.write_reports", return_value=["sci-core"]) as reports,
            patch("ci_scope.output_flags"),
        ):
            ci_scope.run({"full": True, "active": True}, "coverage")
            self.assertIn("--workspace", run.call_args_list[1].args[0])
            self.assertIn("--all-features", run.call_args_list[1].args[0])
            self.assertEqual(
                reports.call_args.args[2], {p["name"] for p in self.meta["packages"]}
            )

    def test_benchmark_only_changes_do_not_refresh_coverage(self):
        with (
            patch("ci_scope.subprocess.run") as run,
            patch("ci_scope.metadata", return_value=self.meta),
            patch("ci_scope.output_flags") as output,
        ):
            ci_scope.run(self.select("benches/filter.rs"), "coverage")
            run.assert_not_called()
            output.assert_called_once_with([])


class EventTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {"GITHUB_EVENT_PATH": "event.json", "GITHUB_EVENT_NAME": "pull_request"},
    )
    @patch(
        "ci_scope.Path.read_text",
        return_value='{"pull_request":{"base":{"sha":"base"}}}',
    )
    @patch(
        "ci_scope.git",
        side_effect=["ancestor\n", "crates/sci-filter/src/a b.rs\0tests/filter.rs\0"],
    )
    def test_pull_request_uses_merge_base_and_preserves_spaces(self, git, read):
        self.assertEqual(
            ci_scope.changed_files(),
            ["crates/sci-filter/src/a b.rs", "tests/filter.rs"],
        )
        self.assertEqual(git.call_args_list[0].args, ("merge-base", "base", "HEAD"))
        self.assertEqual(git.call_args_list[1].args[-2:], ("ancestor", "HEAD"))
        self.assertIn("--no-renames", git.call_args_list[1].args)

    @patch.dict(
        os.environ, {"GITHUB_EVENT_PATH": "event.json", "GITHUB_EVENT_NAME": "push"}
    )
    @patch("ci_scope.Path.read_text", return_value='{"before":"previous"}')
    @patch("ci_scope.git", return_value="tests/filter.rs\0")
    def test_push_uses_previous_head(self, git, read):
        self.assertEqual(ci_scope.changed_files(), ["tests/filter.rs"])
        self.assertEqual(git.call_args.args[-2:], ("previous", "HEAD"))

    @patch.dict(
        os.environ, {"GITHUB_EVENT_PATH": "event.json", "GITHUB_EVENT_NAME": "push"}
    )
    @patch("ci_scope.Path.read_text", return_value='{"before":"missing"}')
    @patch("ci_scope.git", side_effect=subprocess.CalledProcessError(128, "git"))
    def test_missing_history_selects_everything(self, git, read):
        self.assertIsNone(ci_scope.changed_files())

    @patch.dict(
        os.environ,
        {"GITHUB_EVENT_PATH": "event.json", "GITHUB_EVENT_NAME": "workflow_dispatch"},
    )
    @patch("ci_scope.Path.read_text", return_value="{}")
    def test_manual_run_selects_everything(self, read):
        self.assertIsNone(ci_scope.changed_files())


if __name__ == "__main__":
    unittest.main()
