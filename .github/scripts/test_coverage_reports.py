"""Verify that carried coverage reports contain only the requested crate."""

import tempfile
import unittest
from pathlib import Path

from coverage_reports import write_reports


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.meta = {
            "packages": [
                {
                    "name": name,
                    "manifest_path": str(self.root / directory / "Cargo.toml"),
                }
                for name, directory in [
                    ("sci", "."),
                    ("sci-core", "crates/sci-core"),
                    ("sci-filter", "crates/sci-filter"),
                ]
            ]
        }
        self.source = self.root / "lcov.info"
        self.output = self.root / "flags"

    @staticmethod
    def record(path, count=1):
        return f"SF:{path}\nFN:1,example\nFNDA:{count},example\nDA:1,{count}\nLF:1\nLH:{int(count > 0)}\nend_of_record\n"

    def write(self, selected):
        return write_reports(self.root, self.meta, selected, self.source, self.output)

    def test_disjoint_reports_and_relative_paths(self):
        core = "crates/sci-core/src/lib.rs"
        filter_ = "crates/sci-filter/src/lib.rs"
        self.source.write_text(
            self.record(self.root / core)
            + self.record(filter_, 0)
            + self.record("tests/filter.rs")
        )
        self.assertEqual(
            self.write({"sci-core", "sci-filter", "sci"}), ["sci-core", "sci-filter"]
        )
        self.assertEqual((self.output / "sci-core.info").read_text(), self.record(core))
        self.assertEqual(
            (self.output / "sci-filter.info").read_text(), self.record(filter_, 0)
        )
        self.assertFalse((self.output / "sci.info").exists())

    def test_incidental_dependencies_are_not_uploaded(self):
        self.source.write_text(
            self.record("crates/sci-core/src/lib.rs")
            + self.record("crates/sci-filter/src/lib.rs")
        )
        self.assertEqual(self.write({"sci-filter"}), ["sci-filter"])
        self.assertEqual(
            list(self.output.glob("*.info")), [self.output / "sci-filter.info"]
        )

    def test_multiple_source_records_are_preserved(self):
        content = self.record("crates/sci-filter/src/lib.rs") + self.record(
            "crates/sci-filter/src/update.rs", 0
        )
        self.source.write_text(content)
        self.write({"sci-filter"})
        self.assertEqual((self.output / "sci-filter.info").read_text(), content)

    def test_final_record_has_a_newline(self):
        content = self.record("crates/sci-filter/src/lib.rs")
        self.source.write_text(content.rstrip("\n"))
        self.write({"sci-filter"})
        self.assertEqual((self.output / "sci-filter.info").read_text(), content)

    def test_old_reports_are_removed(self):
        self.output.mkdir()
        (self.output / "old.info").write_text("stale")
        self.source.write_text(self.record("crates/sci-filter/src/lib.rs"))
        self.write({"sci-filter"})
        self.assertFalse((self.output / "old.info").exists())

    def test_unknown_package_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown workspace"):
            self.write({"unknown"})

    def test_empty_or_unrelated_reports_are_rejected(self):
        for content in [
            "",
            self.record("tests/filter.rs"),
            self.record("/external/lib.rs"),
        ]:
            self.source.write_text(content)
            with (
                self.subTest(content=content),
                self.assertRaisesRegex(ValueError, "No library coverage"),
            ):
                self.write({"sci-filter"})

    def test_malformed_reports_are_rejected(self):
        for content in [
            "SF:unfinished\n",
            "DA:1,1\nend_of_record\n",
            "SF:a\nSF:b\nend_of_record\n",
        ]:
            self.source.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.write({"sci-filter"})


if __name__ == "__main__":
    unittest.main()
