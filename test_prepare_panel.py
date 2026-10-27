import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
import prepare_panel as module


def compatible_manifest():
    jobs = copy.deepcopy(module.build_jobs())
    for job in jobs:
        job["status"] = "complete"
    return {
        "status": "complete",
        "collection_id": "test_collection",
        "start_period": config.START_PERIOD,
        "end_period": config.END_PERIOD,
        "classification": config.CLASSIFICATION,
        "index_unit": config.INDEX_UNIT,
        "euro_area_entry": copy.deepcopy(config.EURO_AREA_ENTRY),
        "jobs": jobs,
    }


class PreparationCheck(unittest.TestCase):
    def test_missing_inside_outside_and_never_observed_are_distinct(self):
        dates = pd.date_range("2026-01", "2026-05", freq="MS")
        frame = pd.DataFrame(
            {
                "geo": ["FR"] * 10,
                "coicop18": ["A"] * 5 + ["B"] * 5,
                "date": list(dates) * 2,
                "value": pd.array(
                    [None, 100, None, 101, None] + [None] * 5, dtype="Float64"
                ),
            }
        )
        table = module.coverage(frame, dates).set_index("coicop18")
        self.assertEqual(table.loc["A", "observed_months"], 2)
        self.assertEqual(table.loc["A", "missing_inside"], 1)
        self.assertEqual(table.loc["A", "missing_before_first"], 1)
        self.assertEqual(table.loc["A", "missing_after_last"], 1)
        self.assertTrue(table.loc["B", "never_observed"])
        self.assertEqual(table.loc["B", "missing_inside"], 0)
        self.assertTrue(pd.isna(table.loc["B", "missing_before_first"]))

    def test_weight_sums_use_disjoint_divisions_not_children_or_specials(self):
        codes = module.DIVISIONS + ["TOTAL", "CP011", "NRG"]
        frame = pd.DataFrame(
            {
                "geo": ["FR"] * len(codes) * 2,
                "year": [2025] * len(codes) + [2026] * len(codes),
                "coicop18": codes * 2,
                "value": pd.array(
                    [None] * len(codes) + [1000] + [0] * 12 + [1000, 500, 100],
                    dtype="Float64",
                ),
            }
        )
        sums = module.item_weight_sums(frame).set_index("year")
        self.assertFalse(sums.loc[2025, "available"])
        self.assertFalse(sums.loc[2025, "bad_partition"])
        self.assertEqual(sums.loc[2026, "division_sum"], 1000)
        self.assertFalse(sums.loc[2026, "bad_partition"])
        broken = frame.copy()
        broken.loc[
            (broken["year"] == 2026) & (broken["coicop18"] == "CP02"), "value"
        ] = None
        self.assertTrue(
            module.item_weight_sums(broken).set_index("year").loc[2026, "bad_partition"]
        )

    def test_country_weights_respect_backcast_and_adoption(self):
        frame = pd.DataFrame(
            {
                "geo": ["FR", "HR"] * 3,
                "year": [1996, 1996, 2022, 2022, 2023, 2023],
                "value": pd.array([1000, None, 1000, None, 900, 100], dtype="Float64"),
            }
        )
        table, sums, errors = module.audit_country_weights(
            frame, ["FR", "HR"], [1996, 2022, 2023]
        )
        self.assertFalse(any(errors.values()))
        self.assertEqual(sums["expected_countries"].tolist(), [1, 1, 2])
        self.assertTrue(
            table.loc[
                (table["geo"] == "FR") & (table["year"] == 1996),
                "included_in_ea_reference",
            ].iloc[0]
        )
        self.assertTrue(
            table.loc[(table["geo"] == "HR") & (table["year"] == 2022), "weight_share"]
            .isna()
            .all()
        )
        broken = frame.copy()
        broken.loc[(broken["geo"] == "HR") & (broken["year"] == 2022), "value"] = 100
        broken.loc[(broken["geo"] == "FR") & (broken["year"] == 2023), "value"] = None
        _, _, bad = module.audit_country_weights(
            broken, ["FR", "HR"], [1996, 2022, 2023]
        )
        self.assertEqual(bad["positive_weights_outside_ea_reference"], 1)
        self.assertEqual(bad["missing_or_zero_active_country_weights"], 1)

    def test_weight_index_alignment_preserves_missing_weights(self):
        weights = pd.DataFrame(
            {
                "geo": ["FR"] * 3,
                "coicop18": ["A", "B", "C"],
                "year": [2026] * 3,
                "value": pd.array([10, 20, 0], dtype="Float64"),
            }
        )
        presence = pd.DataFrame(
            {
                "geo": ["FR"] * 3,
                "coicop18": ["A", "C", "D"],
                "year": [2026] * 3,
                "index_months": [2, 0, 3],
            }
        )
        result = module.align_weights(
            weights, presence, pd.date_range("2026-01", "2026-03", freq="MS")
        ).set_index("coicop18")
        self.assertTrue(result.loc["A", "positive_weight_partial_index"])
        self.assertTrue(result.loc["B", "positive_weight_without_index"])
        self.assertFalse(result.loc["C", "positive_weight_without_index"])
        self.assertTrue(result.loc["D", "index_without_item_weight"])
        self.assertTrue(pd.isna(result.loc["D", "item_weight"]))

    def test_manifest_rejects_incomplete_or_incompatible_jobs(self):
        real = compatible_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            for issue in ("job_status", "missing_job", "wrong_filter"):
                manifest = copy.deepcopy(real)
                if issue == "job_status":
                    manifest["jobs"][0]["status"] = "pending"
                elif issue == "missing_job":
                    manifest["jobs"].pop()
                else:
                    manifest["jobs"][0]["filters"]["unit"] = "I15"
                path.write_text(json.dumps(manifest))
                with self.subTest(issue=issue), self.assertRaises(ValueError):
                    module.select_manifest(path)

    def test_corrupted_raw_file_is_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text("{}")
            (root / "metadata.json").write_text(json.dumps({"sha256": "invalid"}))
            job = {
                "raw_file": "data.json",
                "metadata_file": "metadata.json",
                "sha256": "invalid",
            }
            with (
                patch.object(config, "PROJECT_ROOT", root),
                self.assertRaises(ValueError),
            ):
                module.read_job(job)

    def test_failed_preparation_removes_old_success_marker_and_records_failure(self):
        real_path = Path("fixture_manifest.json")
        real_manifest = compatible_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "panel" / real_manifest["collection_id"]
            output.mkdir(parents=True)
            (output / "_SUCCESS").write_text("stale")
            with (
                patch.object(config, "PROCESSED_DATA_DIR", Path(tmp)),
                patch.object(
                    module, "select_manifest", return_value=(real_path, real_manifest)
                ),
                patch.object(
                    module, "read_job", side_effect=ValueError("bad checksum")
                ),
                redirect_stdout(io.StringIO()),
                self.assertRaises(ValueError),
            ):
                module.main()
            self.assertFalse((output / "_SUCCESS").exists())
            report = json.loads((output / "audit.json").read_text())
            self.assertEqual(report["integrity_status"], "failed")
            self.assertIn("bad checksum", report["exception"])


if __name__ == "__main__":
    unittest.main()
