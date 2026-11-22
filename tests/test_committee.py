import io

import numpy as np
import pandas as pd

from annz_rebuilt.architecture import Architecture
from annz_rebuilt.committee import (
    predict_committee,
    train_committee,
    unpack_committee,
    write_committee_predictions_csv,
)
from annz_rebuilt.data_preparation import fit_data_preparation


def _fixture(rows=120):
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=rows)
    x2 = rng.uniform(-1, 1, size=rows)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x1_err": np.full(rows, 0.03),
            "x2_err": np.full(rows, 0.05),
            "z": 0.2 + 0.1 * x1 - 0.04 * x2,
        }
    )
    source = frame.to_csv(index=False).encode()
    preparation = fit_data_preparation(
        io.BytesIO(source),
        ["x1", "x2"],
        ["z"],
        rows_per_batch=50,
        batch_size=16,
        train_fraction=0.7,
        validation_fraction=0.15,
        random_seed=12,
    )
    architecture = Architecture(("x1", "x2"), ("z",), (6,), "tanh")
    return source, preparation, architecture


def test_committee_members_have_independent_seeds_and_checkpoint_roundtrips():
    source, preparation, architecture = _fixture()
    trained = train_committee(
        io.BytesIO(source),
        architecture,
        preparation,
        committee_size=3,
        epochs=2,
        optimizer_name="adam",
        device="cpu",
        base_seed=90,
    )
    members, metadata = unpack_committee(trained["checkpoint_bytes"])
    assert len(members) == 3
    assert metadata["committee_size"] == 3
    assert len(set(row["seed"] for row in metadata["members"])) == 3
    assert trained["history"]["member"].nunique() == 3


def test_vmmin_committee_supports_non_tanh_activation():
    source, preparation, _ = _fixture()
    architecture = Architecture(("x1", "x2"), ("z",), (2,), "relu")
    trained = train_committee(
        io.BytesIO(source),
        architecture,
        preparation,
        committee_size=2,
        epochs=2,
        optimizer_name="vmmin",
        device="cpu",
        base_seed=17,
    )
    members, metadata = unpack_committee(trained["checkpoint_bytes"])
    assert len(members) == 2
    assert metadata["committee_size"] == 2


def test_committee_uncertainties_match_cxx_variance_and_quadrature():
    source, preparation, architecture = _fixture()
    trained = train_committee(
        io.BytesIO(source),
        architecture,
        preparation,
        committee_size=3,
        epochs=2,
        device="cpu",
        base_seed=42,
    )
    frame = pd.read_csv(io.BytesIO(source)).iloc[:7]
    x = frame[["x1", "x2"]].to_numpy()
    errors = frame[["x1_err", "x2_err"]].to_numpy()
    result = predict_committee(
        x,
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        input_errors=errors,
        device="cpu",
    )
    assert result["prediction"].shape == (7, 1)
    assert np.all(result["committee_std"] >= 0)
    assert np.all(result["input_error_std"] >= 0)
    # Regression assertion for legacy test.cpp: sum((mean_J * error)^2) per input.
    expected_input_std = np.sqrt(
        np.sum(np.square(result["mean_jacobian"] * errors[:, None, :]), axis=2)
    )
    np.testing.assert_allclose(
        result["input_error_std"], expected_input_std, rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        result["total_std"] ** 2,
        result["committee_std"] ** 2 + result["input_error_std"] ** 2,
        rtol=1e-6,
        atol=1e-10,
    )
    members, _ = unpack_committee(trained["checkpoint_bytes"])
    assert len(members) == 3
    no_errors = predict_committee(
        x, architecture, preparation, trained["checkpoint_bytes"], device="cpu"
    )
    np.testing.assert_allclose(no_errors["input_error_std"], 0.0)


def test_committee_prediction_export_includes_error_components(tmp_path):
    source, preparation, architecture = _fixture()
    trained = train_committee(
        io.BytesIO(source),
        architecture,
        preparation,
        committee_size=2,
        epochs=1,
        device="cpu",
    )
    output = tmp_path / "committee.csv"
    result = write_committee_predictions_csv(
        io.BytesIO(source),
        output,
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        error_columns={"x1": "x1_err", "x2": "x2_err"},
        device="cpu",
    )
    saved = pd.read_csv(output)
    assert result["predicted_rows"] == 120
    assert {"z", "z_committee_std", "z_input_error_std", "z_total_std"}.issubset(
        saved.columns
    )
    assert np.all(saved["z_input_error_std"] > 0)


def test_committee_of_two_matches_original_cxx_no_scatter_rule():
    source, preparation, architecture = _fixture()
    trained = train_committee(
        io.BytesIO(source),
        architecture,
        preparation,
        committee_size=2,
        epochs=1,
        device="cpu",
    )
    frame = pd.read_csv(io.BytesIO(source)).iloc[:3]
    result = predict_committee(
        frame[["x1", "x2"]].to_numpy(),
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        device="cpu",
    )
    np.testing.assert_array_equal(result["committee_std"], np.zeros((3, 1)))


def test_committee_bundle_can_be_reloaded_without_retraining():
    from annz_rebuilt.data_preparation import DataPreparation
    from annz_rebuilt.model_bundle import create_model_bundle, open_model_bundle

    source, preparation, architecture = _fixture()
    trained = train_committee(
        io.BytesIO(source),
        architecture,
        preparation,
        committee_size=3,
        epochs=1,
        device="cpu",
    )
    artifact = create_model_bundle(
        architecture.to_config_dict(),
        preparation.to_config_dict(),
        trained["checkpoint_bytes"],
        training_config={
            "committee_enabled": True,
            "committee_size": 3,
            "optimizer": "adam",
            "prediction_error_columns": {"x1": "x1_err", "x2": "x2_err"},
        },
        training_history=trained["history"],
    )
    restored = open_model_bundle(artifact)
    restored_preparation = DataPreparation.from_config_dict(restored.data_preparation)
    assert restored.training_config["committee_enabled"] is True
    assert restored.training_config["prediction_error_columns"] == {
        "x1": "x1_err",
        "x2": "x2_err",
    }
    assert restored.committee_metadata["committee_size"] == 3
    assert len(restored.committee_metadata["members"]) == 3
    frame = pd.read_csv(io.BytesIO(source)).iloc[:4]
    prediction = predict_committee(
        frame[["x1", "x2"]].to_numpy(),
        architecture,
        restored_preparation,
        restored.checkpoint_bytes,
        device="cpu",
    )
    assert prediction["prediction"].shape == (4, 1)
