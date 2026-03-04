"""Unit tests for src/data/loader.py"""

import pandas as pd
import pytest


class TestValidateDataConsistency:
    def test_consistent_datasets(self, synthetic_features, synthetic_classes):
        from src.data.loader import validate_data_consistency

        result = validate_data_consistency(synthetic_classes, synthetic_features)
        assert result["is_consistent"] is True
        assert result["classes_txids"] == result["features_txids"] == result["common_txids"]

    def test_inconsistent_extra_class_detected(self, synthetic_features, synthetic_classes):
        from src.data.loader import validate_data_consistency

        classes_extra = pd.concat(
            [synthetic_classes, pd.DataFrame({"txId": [99999], "class": [1]})]
        ).reset_index(drop=True)
        result = validate_data_consistency(classes_extra, synthetic_features)
        assert result["is_consistent"] is False

    def test_returns_expected_keys(self, synthetic_features, synthetic_classes):
        from src.data.loader import validate_data_consistency

        result = validate_data_consistency(synthetic_classes, synthetic_features)
        assert set(result.keys()) == {
            "classes_txids",
            "features_txids",
            "common_txids",
            "is_consistent",
        }


class TestMergeAndFilterLabeled:
    def test_labeled_subset_of_all_data(self, synthetic_features, synthetic_classes):
        from src.data.loader import merge_and_filter_labeled

        data, labeled = merge_and_filter_labeled(synthetic_features, synthetic_classes)
        assert len(labeled) <= len(data)

    def test_all_labeled_have_valid_class(self, synthetic_features, synthetic_classes):
        from src.data.loader import merge_and_filter_labeled

        _, labeled = merge_and_filter_labeled(synthetic_features, synthetic_classes)
        assert labeled["class"].isin([1, 2]).all()

    def test_class_stored_as_integer(self, synthetic_features, synthetic_classes):
        from src.data.loader import merge_and_filter_labeled

        _, labeled = merge_and_filter_labeled(synthetic_features, synthetic_classes)
        assert labeled["class"].dtype in [int, "int64", "int32"]

    def test_txid_preserved_in_output(self, synthetic_features, synthetic_classes):
        from src.data.loader import merge_and_filter_labeled

        _, labeled = merge_and_filter_labeled(synthetic_features, synthetic_classes)
        assert "txId" in labeled.columns

    def test_string_class_labels_handled(self, synthetic_features, synthetic_classes):
        """Regression test: classes stored as strings should still pass filtering."""
        from src.data.loader import merge_and_filter_labeled

        string_classes = synthetic_classes.copy()
        string_classes["class"] = string_classes["class"].astype(str)
        _, labeled = merge_and_filter_labeled(synthetic_features, string_classes)
        assert len(labeled) > 0


class TestLoadEllipticDataValidation:
    def test_missing_data_dir_raises_file_not_found(self, tmp_path):
        from src.data.loader import load_elliptic_data

        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            load_elliptic_data(tmp_path / "nonexistent_dir")
