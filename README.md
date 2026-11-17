# Deep Learning — Homework 1

Implementing the same classifiers twice: **once from scratch in NumPy**, then again in **PyTorch**, to see what the frameworks actually do for you.

Built for the *Deep Learning* course at Instituto Superior Técnico.

## What's implemented

### Part 1 — from scratch (NumPy)

| Model | File |
|---|---|
| Perceptron | `skeleton_code/hw1-q1.py` |
| Logistic regression (+ optional L2) | `skeleton_code/hw1-q1.py` |
| Multi-layer perceptron | `skeleton_code/hw1-q1.py` |

No autograd, no `nn.Module` — forward passes, cross-entropy, backpropagation and the weight updates are written by hand.

### Part 2 — the same models in PyTorch

`skeleton_code/hw1-q2.py` reimplements logistic regression and an MLP with `torch.nn`, then uses the framework to sweep hyperparameters that would be painful to hand-roll.

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

The gap between the linear models and the MLP is the whole point of the exercise: the from-scratch implementation reproduces the framework's numbers, so the improvement comes from the model, not the library.

## Run

```bash
cd skeleton_code
python hw1-q1.py -model mlp -epochs 20 -hidden_size 100
python hw1-q2.py -model mlp -epochs 20
```

Useful flags: `-epochs`, `-learning_rate`, `-hidden_size`, `-l2_penalty`, `-batch_size`, `-data_path`.

## Structure

```
skeleton_code/hw1-q1.py    NumPy models
skeleton_code/hw1-q2.py    PyTorch models
skeleton_code/utils.py     Dataset loading and batching
*.pdf                      Training curves and weight-norm plots
*-results*.txt             Recorded runs
homework1.pdf              Assignment statement
```

## Notes

- Requires `numpy`, `matplotlib` and (for part 2) `torch`.
- The `.npz` dataset is intentionally excluded from the repo.
- The accompanying [Deep Learning coursework repo](https://github.com/jaimevidex/DL) contains Homework 2.
