# Notes

The project is organised so the NumPy implementation and the PyTorch implementation can be compared directly.

- `numpy_models.py` contains the perceptron, logistic regression and MLP written from scratch.
- `torch_models.py` reimplements the same two models with `torch.nn` so hyperparameter sweeps are cheap.
- `utils.py` handles dataset loading and batching.

The dataset (`intel_landscapes.v2.npz`) is not included. Place it anywhere and point `-data_path` at it.

Runs are recorded under the root folder: `*-results.txt` holds the final accuracy and training time, and the PDFs are the corresponding curves.
