from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from annz_rebuilt.architecture import Architecture
from annz_rebuilt.data_preparation import (
    DataPreparation,
    fit_data_preparation,
    iter_prepared_batches,
)
from annz_rebuilt.neural_network import (
    build_network,
    evaluate_model,
    predict_single,
    train_model,
    write_predictions_csv,
)
from annz_rebuilt.project import create_project_zip, open_project_zip


@pytest.fixture
def prepared():
    rows = 80
    x1 = np.linspace(-2.0, 2.0, rows)
    x2 = np.sin(np.linspace(0.0, 5.0, rows))
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "z": 0.25 + 0.08 * x1 + 0.02 * x2,
            "mass": 10.0 + 0.5 * x1 - 0.3 * x2,
        }
    )
    source = frame.to_csv(index=False).encode("utf-8")
    preparation = fit_data_preparation(
        io.BytesIO(source),
        ["x1", "x2"],
        ["z", "mass"],
        rows_per_batch=17,
        batch_size=8,
        train_fraction=0.7,
        validation_fraction=0.15,
        random_seed=19,
    )
    architecture = Architecture(("x1", "x2"), ("z", "mass"), (8,), "tanh")
    return source, preparation, architecture


def test_architecture_rejects_input_target_overlap():
    with pytest.raises(ValueError, match="distinct"):
        Architecture(("x",), ("x",), (8,))


def test_split_and_epoch_shuffle_are_reproducible(prepared):
    source, preparation, _ = prepared
    assert sum(preparation.rows_by_split.values()) == preparation.usable_rows

    def batches(seed):
        return list(
            iter_prepared_batches(
                io.BytesIO(source), preparation, split="train", shuffle_seed=seed
            )
        )

    first, repeated, changed = batches(123), batches(123), batches(124)
    assert len(first) == len(repeated) == len(changed)
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    assert any(not np.array_equal(a[0], b[0]) for a, b in zip(first, changed))


def test_adam_loss_is_mean_over_all_targets_and_metrics_work(prepared):
    torch = pytest.importorskip("torch")
    source, preparation, architecture = prepared
    torch.manual_seed(preparation.random_seed)
    model = build_network(architecture).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-12)
    weighted_loss = 0.0
    row_count = 0
    for x_np, y_np in iter_prepared_batches(
        io.BytesIO(source),
        preparation,
        split="train",
        shuffle_seed=preparation.random_seed + 1,
    ):
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        weighted_loss += loss.item() * len(x_np)
        row_count += len(x_np)
        loss.backward()
        optimizer.step()
    expected = weighted_loss / row_count
    result = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=1,
        learning_rate=1e-12,
        optimizer_name="adam",
        device="cpu",
    )
    assert result["history"].iloc[0]["training_loss"] == pytest.approx(
        expected, rel=1e-5, abs=1e-7
    )
    report = evaluate_model(
        io.BytesIO(source),
        architecture,
        preparation,
        result["checkpoint_bytes"],
        split="test",
        device="cpu",
    )
    assert set(report["target"]) == {"z", "mass"}
    assert {"MAE", "RMSE", "R2", "Q2", "sigma_RMS"}.issubset(report.columns)


def test_vmmin_training_and_checkpoint_resume(prepared):
    torch = pytest.importorskip("torch")
    source, preparation, architecture = prepared
    first = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=2,
        optimizer_name="vmmin",
        device="cpu",
    )
    assert first["completed_epoch"] >= 1
    state = torch.load(
        io.BytesIO(first["checkpoint_bytes"]), map_location="cpu", weights_only=True
    )
    assert state["optimizer_name"] == "vmmin"
    assert state["optimizer_state"] is None
    assert state["vmmin_alpha"] > 0
    assert state["vmmin_beta"] > 0
    resumed = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=1,
        optimizer_name="vmmin",
        checkpoint_bytes=first["checkpoint_bytes"],
        device="cpu",
    )
    assert resumed["completed_epoch"] >= first["completed_epoch"]


def test_prediction_export_skips_invalid_rows(prepared, tmp_path):
    source, preparation, architecture = prepared
    trained = train_model(
        io.BytesIO(source), architecture, preparation, epochs=1, device="cpu"
    )
    predictions = pd.DataFrame({"x1": [0.1, np.nan, 0.3], "x2": [0.2, 0.4, 0.5]})
    output = tmp_path / "predictions.csv"
    result = write_predictions_csv(
        io.BytesIO(predictions.to_csv(index=False).encode("utf-8")),
        output,
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        device="cpu",
    )
    assert (result["predicted_rows"], result["skipped_rows"]) == (2, 1)
    saved = pd.read_csv(output)
    assert list(saved.columns) == ["source_row", "z", "mass"]
    assert saved["source_row"].tolist() == [1, 3]


def test_network_supports_a_different_activation_in_each_hidden_layer(prepared):
    torch = pytest.importorskip("torch")
    from annz_rebuilt.architecture import architecture_from_selections

    _, _, _ = prepared
    architecture = architecture_from_selections(
        ["x1", "x2"],
        ["z", "mass"],
        [8, 4, 3],
        "relu",
        hidden_activations=["relu", "tanh", "softplus"],
    )
    model = build_network(architecture)
    activation_types = [type(model[index]).__name__ for index in (1, 3, 5)]
    assert activation_types == ["ReLU", "Tanh", "Softplus"]
    assert architecture.to_config_dict()["hidden_activations"] == [
        "relu",
        "tanh",
        "softplus",
    ]
    x = torch.zeros((2, 2), dtype=torch.float32)
    assert model(x).shape == (2, 2)


def test_architecture_requires_one_activation_per_hidden_layer():
    with pytest.raises(ValueError, match="exactly one activation"):
        Architecture(
            ("x",),
            ("z",),
            (8, 4),
            "relu",
            hidden_activations=("relu",),
        )


def test_uniform_activation_config_remains_backward_compatible():
    architecture = Architecture(("x",), ("z",), (4, 3), "tanh")
    assert architecture.effective_activations == ("tanh", "tanh")
    assert "hidden_activations" not in architecture.to_config_dict()


def test_mixed_activation_checkpoint_can_be_reloaded(prepared):
    source, preparation, _ = prepared
    from annz_rebuilt.architecture import architecture_from_selections

    architecture = architecture_from_selections(
        ["x1", "x2"],
        ["z", "mass"],
        [8, 4],
        "relu",
        hidden_activations=["relu", "tanh"],
    )
    trained = train_model(
        io.BytesIO(source), architecture, preparation, epochs=2, device="cpu"
    )
    report = evaluate_model(
        io.BytesIO(source),
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        split="test",
        device="cpu",
    )
    assert len(report) == 2
    assert architecture.to_config_dict()["hidden_activations"] == ["relu", "tanh"]


def test_project_zip_round_trip(prepared):
    source, preparation, architecture = prepared
    blob = create_project_zip(
        architecture.to_config_dict(),
        data_preparation_config=preparation.to_config_dict(),
        data_source_name="sample.csv",
        data_source_bytes=source,
        training_config={
            "optimizer": "adam",
            "prediction_error_columns": {"x1": "x1_err"},
        },
    )
    restored = open_project_zip(blob)
    assert restored.architecture == architecture.to_config_dict()
    assert restored.data_source_bytes == source
    assert restored.training_config["optimizer"] == "adam"
    assert restored.training_config["prediction_error_columns"] == {"x1": "x1_err"}
    assert (
        DataPreparation.from_config_dict(restored.data_preparation).input_columns
        == preparation.input_columns
    )


def test_project_zip_rejects_unsafe_member_path():
    manifest = (
        '{"format_version":5,"architecture":{"input_columns":["x"],'
        '"target_columns":["z"],"hidden_layers":[4],"activation":"tanh"},'
        '"files":{"data_source":"../outside.csv"}}'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("project.json", manifest)
        archive.writestr("../outside.csv", "unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        open_project_zip(buffer.getvalue())


def test_switching_between_adam_and_vmmin_keeps_model_weights(prepared):
    source, preparation, architecture = prepared
    adam = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=1,
        optimizer_name="adam",
        device="cpu",
    )
    quasi_newton = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=1,
        optimizer_name="vmmin",
        checkpoint_bytes=adam["checkpoint_bytes"],
        device="cpu",
    )
    switched_back = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=1,
        optimizer_name="adam",
        checkpoint_bytes=quasi_newton["checkpoint_bytes"],
        device="cpu",
    )
    assert quasi_newton["checkpoint_bytes"]
    assert switched_back["checkpoint_bytes"]
    assert switched_back["completed_epoch"] >= 1


def test_legacy_vmmin_minimises_quadratic_objective():
    from annz_rebuilt.legacy_vmmin import vmmin

    target = np.array([1.5, -2.0, 0.25], dtype=np.float64)

    def objective(weights):
        difference = weights - target
        return 0.5 * float(np.dot(difference, difference))

    def gradient(weights):
        return weights - target

    result = vmmin(
        np.array([8.0, 4.0, -5.0]),
        objective,
        gradient,
        max_iterations=30,
    )
    assert result.objective < 1e-10
    np.testing.assert_allclose(result.weights, target, atol=1e-5)


def test_vmmin_training_on_cuda_when_available(prepared):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    source, preparation, architecture = prepared
    result = train_model(
        io.BytesIO(source),
        architecture,
        preparation,
        epochs=1,
        optimizer_name="vmmin",
        device="cuda",
    )
    assert result["checkpoint_bytes"]
    assert result["device"].startswith("cuda")


def test_portable_model_bundle_round_trip(prepared):
    from annz_rebuilt.model_bundle import create_model_bundle, open_model_bundle

    _, preparation, architecture = prepared
    history = pd.DataFrame(
        [
            {"epoch": 1, "training_loss": 0.12, "validation_loss": 0.15},
            {"epoch": 2, "training_loss": 0.08, "validation_loss": 0.11},
        ]
    )
    results = pd.DataFrame([{"architecture": 8, "validation_loss": 0.11}])
    metrics = pd.DataFrame(
        [{"target": "z", "MAE": 0.1, "RMSE": 0.2, "R2": 0.9, "Q2": 0.88}]
    )
    artifact = create_model_bundle(
        architecture.to_config_dict(),
        preparation.to_config_dict(),
        b"synthetic-checkpoint-bytes",
        training_config={
            "optimizer": "adam",
            "learning_rate": 0.001,
            "prediction_error_columns": {"x1": "x1_err"},
            "chunk_rows": preparation.rows_per_batch,
            "training_batch_size": preparation.batch_size,
        },
        training_history=history,
        optimisation_results=results,
        evaluation_report=metrics,
        source_data_sha256="abc123",
        bundle_metadata={"uncertainty_method": "legacy Jacobian propagation"},
    )
    restored = open_model_bundle(artifact)
    assert restored.architecture == architecture.to_config_dict()
    from annz_rebuilt.data_preparation import DataPreparation

    assert DataPreparation.from_config_dict(restored.data_preparation) == preparation
    assert restored.checkpoint_bytes == b"synthetic-checkpoint-bytes"
    assert restored.training_config["optimizer"] == "adam"
    assert restored.training_config["prediction_error_columns"] == {"x1": "x1_err"}
    assert restored.training_config["chunk_rows"] == preparation.rows_per_batch
    assert (
        restored.bundle_metadata["uncertainty_method"] == "legacy Jacobian propagation"
    )
    pd.testing.assert_frame_equal(restored.training_history, history)
    pd.testing.assert_frame_equal(
        restored.optimisation_results, results, check_dtype=False
    )
    pd.testing.assert_frame_equal(restored.evaluation_report, metrics)
    assert restored.source_data_sha256 == "abc123"


def test_single_network_jacobian_propagates_input_measurement_errors(prepared):
    source, preparation, architecture = prepared
    trained = train_model(
        io.BytesIO(source), architecture, preparation, epochs=2, device="cpu"
    )
    frame = pd.read_csv(io.BytesIO(source)).iloc[:6]
    inputs = frame[list(architecture.input_columns)].to_numpy(dtype=float)
    errors = np.full_like(inputs, 0.02)
    result = predict_single(
        inputs,
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        input_errors=errors,
        device="cpu",
    )
    assert result["prediction"].shape == (6, architecture.n_outputs)
    assert result["input_error_std"].shape == result["prediction"].shape
    assert np.all(np.isfinite(result["input_error_std"]))
    assert np.all(result["input_error_std"] > 0)
    no_errors = predict_single(
        inputs,
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        device="cpu",
    )
    np.testing.assert_array_equal(no_errors["input_error_std"], 0.0)


def test_single_network_prediction_export_maps_error_columns(prepared, tmp_path):
    source, preparation, architecture = prepared
    trained = train_model(
        io.BytesIO(source), architecture, preparation, epochs=1, device="cpu"
    )
    frame = pd.read_csv(io.BytesIO(source)).iloc[:12].copy()
    frame["x1_err"] = 0.03
    frame["x2_err"] = 0.04
    output = tmp_path / "single_with_errors.csv"
    result = write_predictions_csv(
        io.BytesIO(frame.to_csv(index=False).encode()),
        output,
        architecture,
        preparation,
        trained["checkpoint_bytes"],
        error_columns={"x1": "x1_err", "x2": "x2_err"},
        device="cpu",
    )
    saved = pd.read_csv(output)
    assert result["predicted_rows"] == 12
    assert {"z_input_error_std", "mass_input_error_std"}.issubset(saved.columns)
    assert np.all(saved["z_input_error_std"] > 0)
    assert np.all(saved["mass_input_error_std"] > 0)
    assert "x1_err" not in saved.columns
    assert "x2_err" not in saved.columns


def test_portable_model_bundle_rejects_invalid_archive():
    from annz_rebuilt.model_bundle import open_model_bundle

    with pytest.raises(ValueError, match="valid model bundle"):
        open_model_bundle(b"not a zip archive")


def test_saved_model_bundle_predicts_without_retraining(prepared, tmp_path):
    from annz_rebuilt.data_preparation import DataPreparation
    from annz_rebuilt.model_bundle import create_model_bundle, open_model_bundle

    source, preparation, architecture = prepared
    trained = train_model(
        io.BytesIO(source), architecture, preparation, epochs=2, device="cpu"
    )
    artifact = create_model_bundle(
        architecture.to_config_dict(),
        preparation.to_config_dict(),
        trained["checkpoint_bytes"],
        training_config={"optimizer": "adam", "learning_rate": 0.001},
        training_history=trained["history"],
    )
    restored = open_model_bundle(artifact)
    restored_preparation = DataPreparation.from_config_dict(restored.data_preparation)
    prediction_input = pd.DataFrame({"x1": [0.1, 0.2], "x2": [0.3, 0.4]})
    output = tmp_path / "from_saved_model.csv"
    result = write_predictions_csv(
        io.BytesIO(prediction_input.to_csv(index=False).encode("utf-8")),
        output,
        architecture,
        restored_preparation,
        restored.checkpoint_bytes,
        device="cpu",
    )
    assert result["predicted_rows"] == 2
    assert len(pd.read_csv(output)) == 2
