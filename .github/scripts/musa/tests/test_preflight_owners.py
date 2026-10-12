# Copyright (c) 2026 BAAI. All rights reserved.
"""Pure CPU checks of imported-path ownership using real RECORD fixtures."""

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "musa_preflight_owner_tests", HERE / "preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


class RecordOwnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.site = Path(self.temporary.name)
        self.source = self.site / "openai/__init__.py"
        self.source.parent.mkdir()
        self.source.write_bytes(b"VERSION = 'fixture'\n")

    def distribution(
        self,
        name="openai",
        *,
        claim=True,
        bad_hash=False,
        bad_size=False,
        record=True,
        metadata_member=True,
        extra_record=False,
    ):
        metadata = self.site / (name + "-1.0.dist-info")
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: " + name + "\nVersion: 1.0\n",
            encoding="utf-8",
        )
        if record:
            body = self.source.read_bytes()
            digest = hashlib.sha256(b"different" if bad_hash else body).digest()
            encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            member = "openai/__init__.py" if claim else "other/__init__.py"
            rows = [(member, "sha256=" + encoded, str(len(body) + int(bad_size)))]
            if metadata_member:
                rows.append((metadata.name + "/METADATA", "", ""))
            rows.append((metadata.name + "/RECORD", "", ""))
            if extra_record:
                rows.append(("other-1.0.dist-info/RECORD", "", ""))
                extra = self.site / "other-1.0.dist-info"
                extra.mkdir()
                (extra / "RECORD").write_bytes(b"")
            with (metadata / "RECORD").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                csv.writer(stream).writerows(rows)
        return importlib.metadata.PathDistribution(metadata)

    def owner(self, distributions, advertised=None):
        owners = {canonical(dist.metadata["Name"]): dist for dist in distributions}

        def lookup(name):
            if canonical(name) not in owners:
                raise importlib.metadata.PackageNotFoundError(name)
            return owners[canonical(name)]

        with (
            patch.object(
                preflight.importlib.metadata,
                "packages_distributions",
                return_value={} if advertised is None else {"openai": advertised},
            ),
            patch.object(
                preflight.importlib.metadata,
                "distributions",
                return_value=distributions,
            ) as scan,
            patch.object(
                preflight.importlib.metadata, "distribution", side_effect=lookup
            ) as selected,
        ):
            result = preflight.selected_owner("openai", self.source.resolve())
        return result, scan.call_count, selected.call_count

    def test_advertised_owner_retains_record_verification(self):
        dist = self.distribution()
        owner, scans, selected = self.owner([dist], ["openai"])
        self.assertEqual((owner, scans, selected), ("openai", 0, 1))

    def test_advertised_same_project_aliases_are_deduplicated(self):
        dist = self.distribution("open-ai")
        self.assertEqual(self.owner([dist], ["Open_AI", "open-ai"]), ("open-ai", 0, 1))

    def test_empty_advertisement_uses_actual_record_path(self):
        dist = self.distribution()
        self.assertEqual(self.owner([dist]), ("openai", 1, 1))

    def test_repeated_same_project_metadata_remains_one_candidate(self):
        dist = self.distribution()
        self.assertEqual(self.owner([dist, dist]), ("openai", 1, 1))

    def test_unadvertised_wrong_hash_fails(self):
        with self.assertRaisesRegex(ValueError, "differs from installed RECORD"):
            self.owner([self.distribution(bad_hash=True)])

    def test_unadvertised_wrong_size_fails(self):
        with self.assertRaisesRegex(ValueError, "differs from installed RECORD"):
            self.owner([self.distribution(bad_size=True)])

    def test_unadvertised_zero_owner_fails(self):
        with self.assertRaisesRegex(ValueError, "No unique actual RECORD owner"):
            self.owner([])

    def test_member_from_another_path_is_not_an_owner(self):
        with self.assertRaisesRegex(ValueError, "No unique actual RECORD owner"):
            self.owner([self.distribution(claim=False)])

    def test_unadvertised_multiple_project_owners_fail(self):
        with self.assertRaisesRegex(ValueError, "No unique actual RECORD owner"):
            self.owner([self.distribution("first"), self.distribution("second")])

    def test_missing_record_fails(self):
        with self.assertRaisesRegex(ValueError, "No unique actual RECORD owner"):
            self.owner([self.distribution(record=False)])

    def test_multiple_record_members_fail(self):
        with self.assertRaisesRegex(ValueError, "No unique ordinary RECORD"):
            self.owner([self.distribution(extra_record=True)])

    def test_metadata_must_be_recorded(self):
        with self.assertRaisesRegex(ValueError, "No unique distribution METADATA"):
            self.owner([self.distribution(metadata_member=False)])

    def test_nonempty_wrong_advertisement_does_not_enable_fallback(self):
        dist = self.distribution()
        with self.assertRaisesRegex(ValueError, "PackageNotFoundError"):
            self.owner([dist], ["missing-project"])

    def test_distribution_scan_cap_fails(self):
        first = self.distribution("first", claim=False)
        second = self.distribution("second")
        with (
            patch.object(preflight, "OWNER_DISTRIBUTION_CAP", 1),
            self.assertRaisesRegex(ValueError, "distribution scan cap exceeded"),
        ):
            self.owner([first, second])

    def test_member_cap_accumulates_across_distributions(self):
        first = self.distribution("first")
        second = self.distribution("second", claim=False)
        with (
            patch.object(preflight, "OWNER_MEMBER_CAP", 3),
            self.assertRaisesRegex(ValueError, "member scan cap exceeded"),
        ):
            self.owner([first, second])

    def test_member_scan_cap_fails(self):
        with (
            patch.object(preflight, "OWNER_MEMBER_CAP", 1),
            self.assertRaisesRegex(ValueError, "member scan cap exceeded"),
        ):
            self.owner([self.distribution()])


if __name__ == "__main__":
    unittest.main(verbosity=2)
