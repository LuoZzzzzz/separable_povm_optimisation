# Optimisation of Separable POVMs

Short run-through of the POVM optimisation code. 

---

## Cholesky Parameterisation

We would like to optimise a set of measurement operators `{E_i}` satisfying:

- `E_i ≥ 0`  
- `∑_i E_i = I`

Instead of explicitly enforcing these constraints during optimisation, we enforce them implicitly through a suitable parameterisation.

First, create a set of `d × r` real matrices `L_i`, where:

- `d` is the dimension of the Hilbert space  
- `r ≤ d` is the desired operator rank  

These are the free parameters of the model, stored as learnable PyTorch tensors.

Second, compute:

```
A_i = L_i L_i^T
```

Each `A_i` is positive semidefinite by construction.

Third, form:

```
S = ∑_i A_i
```

and compute `S^{-1/2}` via eigendecomposition:

```
S = V diag(λ) V^T
S^{-1/2} = V diag(λ^{-1/2}) V^T
```

Eigenvalues are clamped to a floor of `1e-12` to avoid division by zero.

Finally, construct:

```
E_i = S^{-1/2} A_i S^{-1/2}
```

These operators are positive semidefinite and sum to the identity by construction, forming a valid POVM.

A final symmetrisation step:

```
E_i ← ½ (E_i + E_i^T)
```

corrects any floating-point asymmetry introduced during computation.

---

## Success Calculation

The quantity being maximised is the **k-copy MAP success rate**: the probability that a MAP decision rule correctly identifies the class of an unknown state from `k` independent measurement outcomes.

Its computation proceeds in three stages.

---

### Single-Copy Probabilities

For each density matrix `ρ` in the dataset and each POVM element `E_m`, the Born-rule probability is:

```
p(m | ρ) = Tr[E_m ρ]
```

This is evaluated for an entire batch simultaneously using:

```python
torch.einsum('ij,nji->n', E_m, rhos)
```

giving a matrix of probabilities of shape `(batch, M)`.

The results are clamped to be non-negative and row-normalised to absorb any rounding error from the POVM construction.

---

### k-Copy Likelihoods

With `k` independent measurements, each aggregate outcome is a count vector

```
s = (s_1, …, s_M)
```

where `s_m` is the number of times outcome `m` was observed, with

```
∑_m s_m = k
```

The method `_generate_count_vectors(k, M)` recursively enumerates all such vectors, producing a count grid of shape

```
binom(k + M - 1, M - 1) × M
```

For each class `c`, the likelihood of observing `s` is averaged over all states in the class:

```
P(s | c) = (1 / N_c) ∑_{i=1}^{N_c} Multinomial(s; k, p_i)
```

where

```
p_i = (p(1 | ρ_i), …, p(M | ρ_i))
```

are the single-copy Born probabilities for state `ρ_i` in class `c`.

This correctly accounts for intra-class variability. It is **not** the same as computing the multinomial for the class-average state, which would conflate a mixture of states with a single average state.

For numerical stability the multinomial is computed in log-space:

```
log P(s | ρ_i)
  = log(k! / ∏_m s_m!) + ∑_m s_m log p(m | ρ_i)
```

The log-factorials are evaluated with `torch.lgamma`.

The key operation is the matrix product:

```python
log_p @ count_grid.T
```

which simultaneously evaluates `∑_m s_m log p(m | ρ_i)` for every count vector in a single batched multiply, yielding a tensor of shape `(N_c, num_outcomes)`.

After exponentiation this is averaged over `i` to give the final likelihoods of shape `(M, num_outcomes)`.

---

### MAP Decision and Success Rate

Given class priors `π_c` (uniform by default), the joint probability that class `c` was the true class and count vector `s` was observed is:

```
J(c, s) = π_c · P(s | c)
```

The MAP decision for each outcome `s` is to predict whichever class maximises `J(c, s)`.

The overall success rate is:

```
P(success) = ∑_s max_c J(c, s)
```

---

## k = 1 Baseline

For `k = 1` the MAP success rate reduces to:

```
P(success, k = 1)
  = ∑_m max_c π_c Tr[E_m σ^c]
```

where

```
σ^c = (1 / N_c) ∑_i ρ_i^c
```

is the class-average state.

This is a semidefinite programme (SDP) and is solved globally via `cvxpy`.

The resulting POVM elements are post-processed (symmetrised, eigendecomposed, eigenvalues clamped) and used to warm-start the Cholesky parameterisation for the gradient-based optimisation at higher `k`.

---

## Optimisation Tricks

If we try to optimise MNIST directly, the eigendecomposition in the POVM construction can fail at step 3. The reason appears to be the `2 × 2` zero padding on the sides of MNIST images.

This padding introduces a subspace common to all data classes that is entirely zero, causing the eigenvalues of `S` to cluster near zero and making `S^{-1/2}` numerically unstable. The instability is related to terms of the form `1 / (λ_i - λ_j)` that appear in the derivative of the eigendecomposition. When two eigenvalues are nearly equal and close to machine precision these become arbitrarily large.

The solution is to strip the padding before feeding images into the optimiser.

A second issue concerns interpretability. Due to the inherent jaggedness of real MNIST images, the later eigenvectors of an optimised measurement appear grainy and noise-like. Applying a small Gaussian blur to the images before optimisation smooths the eigenvectors and makes them substantially more interpretable.

---

## Iterative Optimisation

The optimisation for general `k` is non-convex.

The `optimize()` method addresses this through a multi-round strategy.

- In round 1, `k = 1` is handled by the SDP (globally optimal).
- For `k > 1`, optimisation uses L-BFGS with a strong Wolfe line search, a second-order quasi-Newton method well-suited to smooth non-convex objectives.

To mitigate local minima, **multi-start initialisation** is used: when optimising for `k`, the model is restarted from the best parameter states found for all previously optimised values of `k`, and the overall best result across all starts is retained (although there is usually no improvement). This can be toggled using the `cross_k_restarts` flag.

Each run terminates early when the gradient norm falls below `grad_tol` or when the success rate fails to improve by more than `improvement_tol` for `patience` consecutive epochs.

A common pathology is that the POVM optimised for `l` copies may in fact score higher at some target `k ≠ l` than the POVM that was actually trained for `k`.

To correct this, each subsequent round:

1. Evaluates every state in the pool on every `k` to build a cross-`k` score table.
2. Identifies which target values would benefit from re-optimisation (i.e. where some other state scores strictly higher by more than `reassignment_tol`).
3. Re-runs L-BFGS for each such `k` using the winning foreign state as the warm-start.

This iterates until no `k` wants to switch (a fixed point) or `max_rounds` is reached. Candidates are prioritised by the size of the available improvement.

The `k = 1` entry is never re-optimised, since its SDP solution is already globally optimal.

Example of an optimisation history:
```
round 1/5 k=3 start=k1 [1/1]:   8%|▊         | 376/5000 [01:14<15:13,  5.06it/s, best=0.626284, epoch=377, lr=1.00e-05, success=0.626284]
round 1/5 k=5 start=k3 [1/1]:   7%|▋         | 366/5000 [01:12<15:18,  5.05it/s, best=0.701666, epoch=367, lr=1.00e-05, success=0.701666]
round 1/5 k=7 start=k5 [1/1]:   7%|▋         | 344/5000 [01:08<15:29,  5.01it/s, best=0.752497, epoch=345, lr=1.00e-05, success=0.752497]
round 1/5 k=9 start=k7 [1/1]:   7%|▋         | 365/5000 [01:14<15:44,  4.91it/s, best=0.788563, epoch=366, lr=1.00e-05, success=0.788563]
round 1/5 k=11 start=k9 [1/1]:   7%|▋         | 368/5000 [01:17<16:15,  4.75it/s, best=0.815474, epoch=369, lr=1.00e-05, success=0.815474]
round 1/5 k=13 start=k11 [1/1]:   7%|▋         | 364/5000 [01:21<17:11,  4.49it/s, best=0.836505, epoch=365, lr=1.00e-05, success=0.836505]
round 1/5 k=15 start=k13 [1/1]:   7%|▋         | 368/5000 [01:28<18:32,  4.16it/s, best=0.853264, epoch=369, lr=1.00e-05, success=0.853264]
round 1/5 k=17 start=k15 [1/1]:   7%|▋         | 350/5000 [01:34<20:50,  3.72it/s, best=0.866891, epoch=351, lr=1.00e-05, success=0.866891]
round 1/5 k=19 start=k17 [1/1]:   7%|▋         | 342/5000 [01:47<24:28,  3.17it/s, best=0.878318, epoch=343, lr=1.00e-05, success=0.878318]
round 1/5 k=21 start=k19 [1/1]:   7%|▋         | 362/5000 [02:12<28:20,  2.73it/s, best=0.888004, epoch=363, lr=1.00e-05, success=0.888004]
round 1/5 k=23 start=k21 [1/1]:   7%|▋         | 370/5000 [02:40<33:34,  2.30it/s, best=0.896472, epoch=371, lr=1.00e-05, success=0.896472]
round 1/5 k=25 start=k23 [1/1]:   6%|▋         | 320/5000 [02:44<40:07,  1.94it/s, best=0.903561, epoch=321, lr=1.00e-05, success=0.903561]
round 1/5 k=27 start=k25 [1/1]:   7%|▋         | 346/5000 [03:33<47:54,  1.62it/s, best=0.909872, epoch=347, lr=1.00e-05, success=0.909872]
round 1/5 k=29 start=k27 [1/1]:   6%|▋         | 321/5000 [03:57<57:42,  1.35it/s, best=0.915359, epoch=322, lr=1.00e-05, success=0.915359]
round 2/5 k=21 start=adopt_k25 [1/1]:   7%|▋         | 340/5000 [02:04<28:22,  2.74it/s, best=0.888710, epoch=341, lr=1.00e-05, success=0.888710]
round 2/5 k=19 start=adopt_k23 [1/1]:   6%|▋         | 323/5000 [01:42<24:39,  3.16it/s, best=0.878989, epoch=324, lr=1.00e-05, success=0.878989]
round 2/5 k=25 start=adopt_k29 [1/1]:   7%|▋         | 335/5000 [02:54<40:36,  1.91it/s, best=0.904145, epoch=336, lr=1.00e-05, success=0.904145]
round 2/5 k=23 start=adopt_k27 [1/1]:   7%|▋         | 329/5000 [02:22<33:41,  2.31it/s, best=0.896992, epoch=330, lr=1.00e-05, success=0.896992]
round 2/5 k=27 start=adopt_k29 [1/1]:   6%|▌         | 309/5000 [03:09<47:53,  1.63it/s, best=0.910157, epoch=310, lr=1.00e-05, success=0.910157]
round 2/5 k=17 start=adopt_k19 [1/1]:   6%|▋         | 313/5000 [01:25<21:18,  3.67it/s, best=0.867822, epoch=314, lr=1.00e-05, success=0.867822]
round 2/5 k=15 start=adopt_k17 [1/1]:   7%|▋         | 330/5000 [01:19<18:50,  4.13it/s, best=0.854246, epoch=331, lr=1.00e-05, success=0.854246]
round 2/5 k=11 start=adopt_k13 [1/1]:   7%|▋         | 330/5000 [01:10<16:33,  4.70it/s, best=0.815895, epoch=331, lr=1.00e-05, success=0.815895]
round 3/5 k=13 start=adopt_k15 [1/1]:   6%|▋         | 314/5000 [01:10<17:32,  4.45it/s, best=0.837636, epoch=315, lr=1.00e-05, success=0.837636]
round 3/5 k=29 start=adopt_k27 [1/1]:   6%|▌         | 306/5000 [03:45<57:43,  1.36it/s, best=0.915589, epoch=307, lr=1.00e-05, success=0.915589]
round 3/5 k=23 start=adopt_k25 [1/1]:   6%|▌         | 312/5000 [02:14<33:38,  2.32it/s, best=0.897227, epoch=313, lr=1.00e-05, success=0.897227]
round 3/5 k=21 start=adopt_k23 [1/1]:   6%|▋         | 320/5000 [01:57<28:32,  2.73it/s, best=0.889183, epoch=321, lr=1.00e-05, success=0.889183]
round 3/5 k=19 start=adopt_k21 [1/1]:   6%|▋         | 324/5000 [01:42<24:41,  3.16it/s, best=0.879598, epoch=325, lr=1.00e-05, success=0.879598]
round 3/5 k=27 start=adopt_k25 [1/1]:   6%|▌         | 309/5000 [03:08<47:47,  1.64it/s, best=0.910309, epoch=310, lr=1.00e-05, success=0.910309]
round 4/5 k=17 start=adopt_k19 [1/1]:   6%|▋         | 323/5000 [01:27<21:13,  3.67it/s, best=0.868223, epoch=324, lr=1.00e-05, success=0.868223]
round 4/5 k=25 start=adopt_k27 [1/1]:   6%|▌         | 309/5000 [02:37<39:56,  1.96it/s, best=0.904335, epoch=310, lr=1.00e-05, success=0.904335]
round 4/5 k=23 start=adopt_k21 [1/1]:   6%|▌         | 312/5000 [02:14<33:44,  2.32it/s, best=0.897388, epoch=313, lr=1.00e-05, success=0.897388]
round 5/5 k=27 start=adopt_k25 [1/1]:   6%|▌         | 305/5000 [03:09<48:30,  1.61it/s, best=0.910526, epoch=306, lr=1.00e-05, success=0.910526]
round 5/5 k=21 start=adopt_k23 [1/1]:   6%|▌         | 305/5000 [01:52<28:54,  2.71it/s, best=0.889322, epoch=306, lr=1.00e-05, success=0.889322]
round 5/5 k=29 start=adopt_k25 [1/1]:   6%|▌         | 312/5000 [03:52<58:07,  1.34it/s, best=0.915862, epoch=313, lr=1.00e-05, success=0.915862]  
```
