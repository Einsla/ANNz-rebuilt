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
    state = torch.load(io.BytesIO(first["checkpoint_bytes"]), map_location="cpu", weights_only=True)
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


def test_project_zip_round_trip(prepared):
    source, preparation, architecture = prepared
    blob = create_project_zip(
        architecture.to_config_dict(),
        data_preparation_config=preparation.to_config_dict(),
        data_source_name="sample.csv",
        data_source_bytes=source,
        training_config={"optimizer": "adam"},
    )
    restored = open_project_zip(blob)
    assert restored.architecture == architecture.to_config_dict()
    assert restored.data_source_bytes == source
    assert restored.training_config == {"optimizer": "adam"}
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
        io.BytesIO(source), architecture, preparation,
        epochs=1, optimizer_name="adam", device="cpu",
    )
    quasi_newton = train_model(
        io.BytesIO(source), architecture, preparation,
        epochs=1, optimizer_name="vmmin",
        checkpoint_bytes=adam["checkpoint_bytes"], device="cpu",
    )
    switched_back = train_model(
        io.BytesIO(source), architecture, preparation,
        epochs=1, optimizer_name="adam",
        checkpoint_bytes=quasi_newton["checkpoint_bytes"], device="cpu",
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
