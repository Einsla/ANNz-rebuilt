# Python code reading guide ? ANNzPlus rebuild

This guide complements explanations placed beside important library calls in the source. It assumes familiarity with basic Python, classes, functions and pandas DataFrames.

## How to read the code

- **Parameter:** a value supplied to a function. A type hint such as `source: Path` describes the expected kind of value; it does not automatically convert it.
- **Local variable:** a name holding an intermediate result while a function runs.
- **Return value:** the result sent back with `return`. A `DataFrame` is a pandas table; a `dict` is a collection of named values.
- **`self`:** the current object instance inside a class method.
- **Leading underscore:** names such as `_restore_model` are conventionally internal helper functions.

## Main libraries

| Library | Role in this app | Common calls |
|---|---|---|
| pandas (`pd`) | Tables, CSV files, column selection and summaries | `read_csv`, `DataFrame`, `to_numeric`, `concat`, `to_csv`, `loc` |
| NumPy (`np`) | Numerical arrays and mathematical operations | `asarray`, `mean`, `sum`, `square`, `sqrt`, `stack`, `concatenate`, `isfinite` |
| PyTorch (`torch`, `nn`) | Neural-network layers, tensors, gradients and optimisation | `as_tensor`, `nn.Linear`, `nn.Sequential`, `no_grad`, `backward`, `autograd.grad`, `optim.Adam` |
| Streamlit (`st`) | Interactive browser interface | `selectbox`, `multiselect`, `button`, `dataframe`, `session_state`, `spinner` |
| Python standard library | Files, paths, ZIP archives, JSON and in-memory streams | `Path`, `json.loads`, `json.dumps`, `zipfile.ZipFile`, `io.BytesIO` |

## pandas concepts

- A **DataFrame** is a two-dimensional table with named columns and row labels.
- `pd.read_csv(path)` normally reads a CSV into a DataFrame. With `chunksize=...`, it yields smaller DataFrames one at a time. This is essential for large galaxy catalogues.
- `df.loc[:, columns]` selects columns by their labels.
- `pd.to_numeric(series, errors="coerce")` converts values to numbers; unconvertible values become `NaN` (missing values).
- `pd.concat([...])` joins multiple DataFrames. `df.to_csv(...)` writes a DataFrame as CSV text.

## NumPy concepts

- A **NumPy array** stores numerical values and supports fast vectorised calculations.
- `np.asarray(value)` converts compatible input into an array.
- `np.mean` and `np.sum` calculate averages and totals. `axis=0` works down rows for each column; `axis=1` works across columns for each row.
- `np.square(x)` squares each element; `np.sqrt(x)` takes the square root of each element.
- `np.stack` adds a dimension to combine arrays; `np.concatenate` joins arrays along an existing dimension.
- `np.isfinite` identifies values that are neither infinite nor NaN.

## PyTorch concepts

- A **tensor** is PyTorch numerical data. It can be stored on a CPU or supported GPU.
- `torch.as_tensor(array, device=...)` makes a tensor from NumPy data and places it on the selected device.
- `nn.Linear(in_features, out_features)` creates a fully connected layer.
- `nn.Sequential(layer1, layer2, ...)` runs layers in the supplied order.
- `model(x)` performs a **forward pass** and calculates predictions for input tensor `x`.
- `loss.backward()` calculates gradients of the loss with respect to trainable parameters.
- `optimizer.step()` updates parameters using those gradients. `optimizer.zero_grad()` clears gradients before the next mini-batch.
- `torch.no_grad()` disables gradient tracking when only predictions/evaluation values are needed, saving memory.
- `torch.autograd.grad(output, inputs)` calculates derivatives. In this app these derivatives form the prediction Jacobian, used to estimate how input measurement errors affect predicted targets.

## Streamlit concepts

- Calls such as `st.selectbox(...)` create a widget and return the current selection.
- `st.button(...)` is true on the rerun triggered by a click. Code inside its `if` block handles that click.
- Streamlit reruns the script after interactions. `st.session_state` preserves selections and results across those reruns.
- `st.dataframe(df)` displays a pandas table in the browser; it does not perform the analysis itself.

## Main ANNz+ pipeline

1. `app.py` collects files and settings through Streamlit.
2. `data_analysis.py` scans the catalogue in chunks and creates data-quality summaries.
3. `architecture.py` separates `NetworkStructure` (input/target columns and hidden-layer sizes) from `Architecture` (the trainable model with one activation assigned to each hidden layer).
4. `app.py` lets the user select activation candidates independently for each hidden layer; optimisation evaluates their combinations while keeping the selected network structure fixed.
5. `data_preparation.py` creates train/validation/test splits, calculates scaling values and yields batches.
6. `neural_network.py` builds, trains, evaluates and predicts with one network.
7. `committee.py` trains independent networks, averages predictions and estimates committee uncertainty.
8. `legacy_vmmin.py` and `vmmin_training.py` implement the legacy variable-metric optimiser.
9. `model_bundle.py` saves a trained model for later prediction; `project.py` saves the wider project and optional source catalogue/reports.

The exported `architecture_config.json`, trained-model bundle and project ZIP include `hidden_activations` as an explicit ordered list. Its first item belongs to hidden layer 1, its second item to hidden layer 2, and so on. This is written even when all layers use the same activation, so the saved architecture always shows the activation assigned to every layer.

## Understanding the uncertainty calculation

`J[j, i]` is the derivative of predicted target `j` with respect to input feature `i`. For independent input measurement errors:

`variance(target j) = sum_i((J[j, i] * input_error[i]) ** 2)`

The standard deviation is the square root of that variance. A committee additionally estimates network disagreement from the spread of member predictions. See the comments beside `torch.autograd.grad`, `np.square`, `np.sum` and `np.sqrt` in the source.

## Suggested learning order

Start with `architecture.py`, then `data_analysis.py`, `data_preparation.py`, `neural_network.py`, `committee.py`, and finally `app.py`. The app is longest because it connects the user interface to smaller reusable functions in `src/annz_rebuilt`.
