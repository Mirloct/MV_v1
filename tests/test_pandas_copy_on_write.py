# Created: 2026-11-25; last modified: 2026-11-25.
import numpy as np
import pandas as pd
import pytest

from src.data.loader import PanelSchema
from src.evaluation.event_labels import LabelSpec, load_event_labels
from src.preprocessing.mixed_view import MixedViewBuilder, MixedViewConfig


@pytest.fixture
def copy_on_write():
    with pd.option_context("mode.copy_on_write", True):
        yield


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["beta", "alpha", "beta"], [3, 2, 3]),
        ([None, np.nan, pd.NA, "new", "alpha", "beta"], [0, 0, 0, 1, 2, 3]),
        ([], []),
    ],
)
def test_encode_categorical_copy_on_write(copy_on_write, dtype, raw, expected):
    fit = pd.DataFrame({"c": ["beta", "alpha"]})
    builder = MixedViewBuilder(MixedViewConfig(min_frequency=0.0)).fit(
        fit, np.empty((2, 0)), [], np.array([True, True]), ["c"],
    )
    values = pd.Series(raw, dtype=dtype, index=np.arange(len(raw)) * 3 + 7)
    original = values.copy(deep=True)

    codes = builder.encode_categorical(values, "cat__c")

    np.testing.assert_array_equal(codes, np.array(expected, dtype=np.int64))
    assert codes.dtype == np.dtype("int64")
    assert codes.flags.writeable
    pd.testing.assert_series_equal(values, original)
    np.testing.assert_array_equal(
        builder.transform(pd.DataFrame({"c": values}), np.empty((len(values), 0))),
        np.array(expected, dtype=np.float32).reshape(-1, 1),
    )


@pytest.mark.parametrize("unlisted_as_negative", [False, True])
def test_event_labels_copy_on_write(copy_on_write, tmp_path, unlisted_as_negative):
    path = tmp_path / "reviewed_labels.csv"
    path.write_text(
        "entity_id,codmes,target,label_status\n"
        "00001,202601,1,confirmed\n"
        "00001,202602,1,pending\n"
        "00001,202603,1,confirmed\n"
        "00001,202603,0,confirmed\n"
        "00001,202604,7,confirmed\n"
        "00001,202605,,confirmed\n",
        encoding="utf-8",
    )
    keys = pd.DataFrame({
        "entity_id": ["00002"] + ["00001"] * 7,
        "period": pd.to_datetime([
            "2026-03-01", "2026-01-01", "2026-02-01", "2026-03-01",
            "2026-04-01", "2026-05-01", "2026-06-01", None,
        ]),
    }, index=[8, 3, 7, 1, 6, 2, 5, 4])
    original = keys.copy(deep=True)
    source = path.read_bytes()
    schema = PanelSchema(time_col="period", entity_col="entity_id", target_col=None)
    spec = LabelSpec(unlisted_as_negative=unlisted_as_negative, as_of="202612")
    negative = 0.0 if unlisted_as_negative else np.nan
    expected = [negative, 1.0, np.nan, np.nan, np.nan, np.nan, negative, np.nan]
    known = [unlisted_as_negative, True, False, False, False, False,
             unlisted_as_negative, False]

    for _ in range(2):
        labels = load_event_labels(str(path), schema, keys, spec)

        np.testing.assert_allclose(labels.target, expected, equal_nan=True)
        np.testing.assert_array_equal(labels.known, known)
        np.testing.assert_array_equal(labels.mature_known, known)
        assert labels.audit["conflicting_duplicate_rows"] == 2
        assert labels.audit["invalid_targets"] == 1
        assert labels.audit["null_targets"] == 1
        assert labels.audit["rows_negative"] == (2 if unlisted_as_negative else 0)
        assert labels.episodes["entity"].tolist() == ["00001"]
    pd.testing.assert_frame_equal(keys, original)
    assert path.read_bytes() == source


def test_event_labels_copy_on_write_with_no_unlisted_rows(copy_on_write, tmp_path):
    path = tmp_path / "reviewed_labels.csv"
    path.write_text("entity_id,codmes,target\n00001,202601,0\n", encoding="utf-8")
    keys = pd.DataFrame({"entity_id": ["00001"], "period": pd.to_datetime(["2026-01-01"])})
    labels = load_event_labels(
        str(path), PanelSchema(time_col="period", entity_col="entity_id", target_col=None),
        keys, LabelSpec(unlisted_as_negative=True, as_of="202612"),
    )

    np.testing.assert_array_equal(labels.target, [0.0])
    np.testing.assert_array_equal(labels.known, [True])
    assert labels.episodes.empty
