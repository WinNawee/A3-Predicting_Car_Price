"""
Unit tests for the A3 Car Price classifier.

Per the assignment (Task 3, Objective 3), two unit tests are required:
  1. the model takes the expected input
  2. the output of the model has the expected shape

These exercise the real deployment bundle (app/code/model/car_price_classifier.pkl)
through `model_inference.predict`, the exact code path the Dash app and
the MLflow pyfunc wrapper both use -- no network access, fast and
deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import model_inference as mi


@pytest.fixture(scope="module")
def bundle():
    return mi.load_bundle()


@pytest.fixture
def sample_rows():
    return pd.DataFrame([
        {
            "year": 2015, "km_driven": 50000, "mileage": 18.5, "engine": 1197.0,
            "max_power": 82.0, "seats": 5.0, "owner": 0,
            "fuel": "Petrol", "transmission": "Manual",
            "seller_type": "Individual", "brand": "Maruti",
        },
        {
            "year": 2010, "km_driven": 120000, "mileage": 21.0, "engine": 1498.0,
            "max_power": 100.0, "seats": 5.0, "owner": 2,
            "fuel": "Diesel", "transmission": "Manual",
            "seller_type": "Dealer", "brand": "Skoda",
        },
    ])


def test_model_accepts_expected_input(bundle, sample_rows):
    """(1) The model takes the expected input.

    `model_inference.predict` should accept a DataFrame with the
    feature columns the bundle expects (`bundle["feature_cols"]`),
    including rows with missing values (the fitted imputers in `prep`
    handle those), and an unseen/unknown brand (OneHotEncoder was fit
    with handle_unknown="ignore").
    """
    assert list(sample_rows.columns).sort() == list(bundle["feature_cols"]).sort()

    # should not raise on well-formed input
    preds, probs = mi.predict(bundle, sample_rows)
    assert preds is not None and probs is not None

    # should not raise on missing values (median/most-frequent imputers
    # are fit into `prep`, so NaNs are expected, valid input)
    rows_with_na = sample_rows.copy()
    rows_with_na.loc[0, "mileage"] = np.nan
    rows_with_na.loc[0, "seats"] = np.nan
    preds_na, probs_na = mi.predict(bundle, rows_with_na)
    assert preds_na is not None and probs_na is not None

    # should not raise on a brand never seen during training
    rows_unknown_brand = sample_rows.copy()
    rows_unknown_brand.loc[0, "brand"] = "TotallyMadeUpBrandXYZ"
    preds_unk, probs_unk = mi.predict(bundle, rows_unknown_brand)
    assert preds_unk is not None and probs_unk is not None

    # a required column missing entirely should fail loudly, not
    # silently produce a wrong answer
    bad_rows = sample_rows.drop(columns=["engine"])
    with pytest.raises(Exception):
        mi.predict(bundle, bad_rows)


def test_model_output_shape(bundle, sample_rows):
    """(2) The output of the model has the expected shape.

    `predict` -> (m,) class labels in {0, ..., k-1}.
    probabilities -> (m, k), each row summing to 1.
    """
    m = len(sample_rows)
    k = bundle["k"]

    preds, probs = mi.predict(bundle, sample_rows)

    assert preds.shape == (m,)
    assert set(np.unique(preds)).issubset(set(range(k)))

    assert probs.shape == (m, k)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(m), rtol=1e-6)

    # single-row input still produces well-formed (1,) / (1, k) output
    single_preds, single_probs = mi.predict(bundle, sample_rows.iloc[[0]])
    assert single_preds.shape == (1,)
    assert single_probs.shape == (1, k)
