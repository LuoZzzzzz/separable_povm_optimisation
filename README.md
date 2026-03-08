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

In code:

```python
joint = likelihoods * priors.view(-1, 1)
best_per_s, _ = torch.max(joint, dim=0)
success = best_per_s.sum()
loss = -success
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
