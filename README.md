# Neural Networks from Scratch

The same classifiers written twice: **once from scratch in NumPy**, then again in **PyTorch** — to see exactly what the frameworks do for you and where the numbers come from.

## What's implemented

### From scratch (NumPy)

| Model | File |
|---|---|
| Perceptron | `src/numpy_models.py` |
| Logistic regression (+ optional L2) | `src/numpy_models.py` |
| Multi-layer perceptron | `src/numpy_models.py` |

No autograd, no `nn.Module`: forward passes, cross-entropy, backpropagation and weight updates are written by hand.

### The same models in PyTorch

`src/torch_models.py` reimplements logistic regression and an MLP with `torch.nn`, then uses the framework to sweep hyperparameters that would be painful to hand-roll.

## Dataset

Intel Image Classification (`intel_landscapes.v2.npz`), a six-class landscape dataset. The archive is not committed; download it separately and pass the path with `-data_path`.

## Results

| Model | Test accuracy |
|---|---|
| Perceptron (NumPy) | 37.4% |
| Logistic regression (NumPy) | 46.0% |
| Logistic regression + L2 (NumPy) | 50.5% |
| MLP (NumPy) | 52.3% |
| Logistic regression (PyTorch, lr 0.1) | 49.2% |
| Logistic regression (PyTorch, lr 0.001) | 51.4% |
| MLP (PyTorch, default hyperparameters) | 59.6% |

The gap between the linear models and the MLP is the point: the from-scratch implementation reproduces the framework's numbers, so the improvement comes from the model, not the library.

## Run

```bash
cd src
python numpy_models.py -model mlp -epochs 20 -hidden_size 100
python torch_models.py -model mlp -epochs 20
```

Useful flags: `-epochs`, `-learning_rate`, `-hidden_size`, `-l2_penalty`, `-batch_size`, `-data_path`.

## Structure

```
src/numpy_models.py     NumPy models
src/torch_models.py     PyTorch models
src/utils.py            Dataset loading and batching
*.pdf                   Training curves and weight-norm plots
*-results*.txt          Recorded runs
problem-statement.pdf   Problem statement
notes.md                Data and setup notes
```

## Notes

- Requires `numpy`, `matplotlib` and (for the PyTorch half) `torch`.
- The `.npz` dataset is intentionally excluded from the repo.
- The companion repo [deep-learning-experiments](https://github.com/jaimevidex/deep-learning-experiments) covers CNNs, sequence models and attention.
