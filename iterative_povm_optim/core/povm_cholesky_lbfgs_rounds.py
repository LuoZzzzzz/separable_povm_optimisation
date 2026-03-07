from typing import List, Optional, Tuple
import torch
from torch import nn, optim
from torch.distributions import Multinomial
import math
import matplotlib.pyplot as plt

import numpy as np
import cvxpy as cp
from tqdm import tqdm

# set device and define dtypes (real-only)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
real_dtype = torch.float64
cplx_dtype = torch.float64

class CholeskyPOVM(nn.Module):
    """
    Real symmetric POVM using Cholesky-style factors:
        A_m = L_m L_m^T
        E_m = S^{-1/2} A_m S^{-1/2}
    Supports rank control via `rank` (<= d).
    """

    def __init__(self, d: int, num_classes: int, rank: Optional[int] = None, init_scale: float = 0.05):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.rank = rank if rank is not None else d

        # Each L is (d, rank)
        self.L_factors = nn.ParameterList([
            nn.Parameter(torch.randn(d, self.rank, dtype=real_dtype, device=device) * init_scale)
            for _ in range(num_classes)
        ])

    def forward(self) -> List[torch.Tensor]:
        """
        Return POVM elements E_m (real symmetric, PSD, sum to identity).
        Each A_m = L_m @ L_m.T
        E_m = S^{-1/2} A_m S^{-1/2} where S = sum_m A_m
        """
        # build A_m = L L^T (PSD)
        A_list = [L @ L.T for L in self.L_factors]  # each (d,d), real

        # form S = sum A_m
        S = sum(A_list)

        # compute S^{-1/2} via eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(S)
        eigvals = torch.clamp(eigvals, min=1e-12)
        S_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

        # create POVM elements
        E_list = [S_inv_sqrt @ A @ S_inv_sqrt for A in A_list]

        # symmetrize numerically
        E_list = [0.5 * (E + E.T) for E in E_list]
        return E_list


class QuantumMulticlassClassifier:
    """
    Multiclass quantum classifier that evaluates k-copy MAP success rates
    and optimizes the POVM parameters (now using CholeskyPOVM).
    """
    def __init__(self, d: int, num_classes: int, warm_start: bool = True, povm_rank: Optional[int] = None):
        self.d = d
        self.num_classes = num_classes
        self.povm = CholeskyPOVM(d, num_classes, rank=povm_rank).to(device)
        self.warm_start = warm_start

    @staticmethod
    def _generate_count_vectors(k: int, num_classes: int) -> torch.Tensor:
        """
        Generate all non-negative integer vectors of length `num_classes` that sum to k.
        Returns tensor shape (num_outcomes, num_classes), dtype=torch.long, device=device.
        """
        counts = []

        def helper(remaining: int, classes_left: int, prefix: List[int]):
            if classes_left == 1:
                counts.append(prefix + [remaining])
                return
            for i in range(remaining + 1):
                helper(remaining - i, classes_left - 1, prefix + [i])

        helper(k, num_classes, [])
        counts_t = torch.tensor(counts, dtype=torch.long, device=device)
        return counts_t

    def measurement_probabilities(self, rhos: torch.Tensor) -> torch.Tensor:
        """
        Compute Tr[E_m rho] for all m and all rho.
        rhos: (batch, d, d) real tensor (or complex but we expect real)
        returns: probs (batch, num_classes) real tensor
        """
        E_list = self.povm()
        batch = rhos.shape[0]
        probs = torch.zeros(batch, self.num_classes, dtype=real_dtype, device=device)

        # vectorize across batch by einsum
        for m, E_m in enumerate(E_list):
            # Tr[E_m rho] = trace(E_m @ rho) = sum_{i,j} E[i,j] * rho[j,i]
            val = torch.einsum('ij,nji->n', E_m, rhos)  # shape (batch,)
            probs[:, m] = val

        # clamp small negative numeric noise and floor tiny values
        probs = torch.clamp(probs, min=0.0)
        # renormalize each prob vector to sum to 1 (in case of rounding errors)
        row_sums = probs.sum(dim=1, keepdim=True)
        # if any row sums are zero (numerical), replace with uniform
        zero_mask = (row_sums.squeeze() == 0.0)
        if zero_mask.any():
            probs[zero_mask] = 1.0 / float(self.num_classes)
            row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        return probs
    
    def k_copy_likelihoods(self, rhos_by_class: List[torch.Tensor], k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute P(s | class) for all classes and all count outcomes s.

        For each class, draws a random state from the class and measures it k times.
        Correctly computes: P(s|c) = mean_i Multinomial(s ; k, p_i)
        instead of the incorrect: Multinomial(s ; k, mean_i p_i)

        Returns:
            count_grid:  (num_outcomes, num_classes) long tensor
            likelihoods: (num_classes, num_outcomes) float tensor
        """
        count_grid = self._generate_count_vectors(k, self.num_classes)
        num_outcomes = count_grid.shape[0]
        likelihoods = torch.zeros(self.num_classes, num_outcomes, dtype=real_dtype, device=device)

        # precompute multinomial log-coefficients: log(k!) - sum(log(s_m!))
        # shape: (num_outcomes,)
        log_k_fact = torch.lgamma(torch.tensor(k + 1, dtype=real_dtype, device=device))
        log_count_facts = torch.sum(
            torch.lgamma(count_grid.to(real_dtype) + 1), dim=1
        )
        log_coeff = log_k_fact - log_count_facts  # (num_outcomes,)

        for c, class_rhos in enumerate(rhos_by_class):
            # class_rhos: (N_c, d, d)
            probs = self.measurement_probabilities(class_rhos)  # (N_c, num_classes)
            log_p = torch.log(torch.clamp(probs, min=1e-12))    # (N_c, num_classes)

            # log P(s | state_i) = log_coeff[s] + sum_m count[s,m] * log p_i[m]
            # log_p @ count_grid.T: (N_c, num_classes) @ (num_classes, num_outcomes) -> (N_c, num_outcomes)
            log_terms = log_coeff.unsqueeze(0) + log_p @ count_grid.to(log_p.dtype).T  # (N_c, num_outcomes)

            # average likelihoods over states in the class
            likelihoods[c] = torch.exp(log_terms).mean(dim=0)  # (num_outcomes,)

        return count_grid, likelihoods

    def map_success_rate(self, rhos_by_class: List[torch.Tensor], k: int,
                         priors: Optional[List[float]] = None) -> torch.Tensor:
        """
        Compute MAP success: sum_s max_c (prior_c * P(s | class=c))
        Returns scalar tensor (on device).
        """
        if priors is None:
            priors = [1.0 / self.num_classes] * self.num_classes
        priors = torch.tensor(priors, dtype=real_dtype, device=device)
        count_grid, likelihoods = self.k_copy_likelihoods(rhos_by_class, k)

        joint = likelihoods * priors.view(-1, 1)  # (num_classes, num_outcomes)

        best_per_outcome, _ = torch.max(joint, dim=0)
        success = torch.sum(best_per_outcome)
        success = torch.clamp(success, min=0.0, max=1.0)
        return success

    def _solve_sdp_k1(self, rhos_by_class: List[torch.Tensor], priors: Optional[List[float]] = None,
                      solver: str = "SCS", verbose: bool = False, eps_floor: float = 1e-9):
        """
        Solve the convex SDP for k=1:
            max_{E_m >= 0, sum E_m = I} sum_m prior_m * Tr[sigma_m E_m]
        Returns a list of numpy arrays E_list (d x d, real symmetric).
        """
        d = self.d
        M = self.num_classes

        if priors is None:
            priors = np.ones(M) / float(M)
        priors = np.asarray(priors, dtype=float)

        # compute class-average sigma_c (use real part)
        sigma_list = []
        for class_rhos in rhos_by_class:
            mean_rho = torch.mean(class_rhos, dim=0)
            mean_rho = mean_rho.cpu().detach().numpy()
            mean_rho = np.real(mean_rho)
            mean_rho = 0.5 * (mean_rho + mean_rho.T)
            sigma_list.append(mean_rho)

        # cvxpy variables: one symmetric PSD variable per class
        E_vars = [cp.Variable((d, d), symmetric=True) for _ in range(M)]

        # constraints: PSD and sum = I
        constraints = []
        for Ev in E_vars:
            constraints.append(Ev >> 0)
        constraints.append(sum(E_vars) == np.eye(d))

        # objective
        objective = 0
        for c in range(M):
            objective += priors[c] * cp.trace(sigma_list[c] @ E_vars[c])
        prob = cp.Problem(cp.Maximize(objective), constraints)

        # solve
        try:
            prob.solve(solver=solver, verbose=verbose)
        except Exception:
            prob.solve(solver="SCS", verbose=verbose)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"SDP solver failed with status {prob.status}")

        E_list = []
        for Ev in E_vars:
            Evv = Ev.value
            Evv = 0.5 * (Evv + Evv.T)
            w, v = np.linalg.eigh(Evv)
            w_clamped = np.clip(w, a_min=eps_floor, a_max=None)
            Evv = (v * w_clamped) @ v.T
            E_list.append(Evv.astype(np.float64))

        return E_list

    def _warm_start_from_povms(
        self,
        target_E_list: List[np.ndarray],
        warm_start_iters: int = 10000,
        fit_lr: float = 1e-1,
        lr_factor: float = 0.5,
        lr_patience: int = 50,
        min_lr: float = 1e-6,
        eps_floor: float = 1e-12,
    ):
        """
        Warm-start the CholeskyPOVM's L_factors from the target_E_list (numpy arrays).
        This attempts an exact factorization:
            E = V diag(w) V^T  ->  L = V[:, :rank] * sqrt(w[:rank])
        If the exact factorization fails for numerical reasons, falls back to a short
        Adam-based fit to match the target Es.
        Returns loss history if fallback fitting used (empty list if direct init succeeded).
        """
        # convert targets to symmetric numpy arrays
        targets = []
        for E in target_E_list:
            Ev = np.array(E, dtype=float)
            Ev = 0.5 * (Ev + Ev.T)
            targets.append(Ev)

        # attempt direct factorization and assignment
        direct_ok = True
        for m, Ev in enumerate(targets):
            try:
                # eigendecompose
                w, v = np.linalg.eigh(Ev)
                # clamp small negatives/numerical
                w_clamped = np.clip(w, a_min=eps_floor, a_max=None)

                # choose rank columns (top eigenvalues) if rank < d
                if self.povm.rank >= self.d:
                    # full-rank factor
                    L_np = (v * np.sqrt(w_clamped))  # shape (d,d)
                else:
                    # pick top-rank eigenpairs
                    idx = np.argsort(w_clamped)[::-1][:self.povm.rank]
                    top_v = v[:, idx]  # (d, rank)
                    top_w = w_clamped[idx]  # (rank,)
                    L_np = top_v * np.sqrt(top_w)[None, :]  # (d, rank)

                # assign to parameter (as Torch tensor)
                L_t = torch.tensor(L_np, dtype=real_dtype, device=device)
                # ensure shape matches (d, rank)
                if L_t.shape != self.povm.L_factors[m].data.shape:
                    # if we produced (d,d) but rank<d, slice; if rank==d and shapes match, ok
                    L_t = L_t[:, : self.povm.rank]
                with torch.no_grad():
                    self.povm.L_factors[m].data.copy_(L_t)
            except Exception as e:
                # numerical failure: fallback to small fit
                direct_ok = False
                break

        # if direct factorisation is successful
        if direct_ok:
            return []

        # fallback to Adam if analytic factorisation fails
        target_tensors = [torch.tensor(E, dtype=real_dtype, device=device) for E in targets]

        opt = optim.Adam(self.povm.parameters(), lr=fit_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )

        loss_hist: List[float] = []

        pbar = tqdm(range(warm_start_iters), desc="Warm-start POVM fit", leave=False)
        for it in pbar:
            opt.zero_grad()
            current_Es = [L @ L.T for L in self.povm.L_factors]

            loss = torch.tensor(0.0, dtype=real_dtype, device=device)
            for cur, tgt in zip(current_Es, target_tensors):
                diff = cur - tgt
                loss = loss + torch.sum(diff * diff)

            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu().numpy())
            loss_hist.append(loss_val)

            scheduler.step(loss_val)

            current_lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.3e}", lr=f"{current_lr:.2e}")

            if loss_val < 1e-12:
                pbar.set_postfix(loss=f"{loss_val:.3e}", lr=f"{current_lr:.2e}", status="converged")
                break
            if current_lr <= min_lr:
                pbar.set_postfix(loss=f"{loss_val:.3e}", lr=f"{current_lr:.2e}", status="min_lr")
                break

        return loss_hist

        
    def optimize(
            self,
            rhos_by_class: List[torch.Tensor],
            k_values: List[int],
            num_epochs: int = 500,
            lr: float = 0.1,
            priors: Optional[List[float]] = None,
            sdp_solver: str = "SCS",
            sdp_verbose: bool = False,
            warm_start_fit_iters: int = 10000,
            warm_start_fit_lr: float = 1e-1,
            improvement_tol: float = 1e-8,
            patience: int = 10,
            grad_tol: float = 1e-6,
            # multi-start initialisation from previous ks
            cross_k_restarts: bool = True,
            init_scale_random: float = 0.05,
            # multi-round reassignement
            max_rounds: int = 5,
            reassignment_tol: float = 1e-10,
        ) -> dict:
            """
            Optimize a POVM for each k in k_values.

            Round 1: optimize each k once (k=1 via SDP, k>1 via multi-start LBFGS).
            Then iterate additional rounds:
              - Evaluate every learned POVM (from each k) on every other k.
              - If some k is beaten by another k's POVM (by > reassignment_tol),
                re-optimize that specific k using the winning POVM as an extra warm start.
              - Repeat until no k wants to switch, or max_rounds reached.

            After convergence, each k's stored POVM is
            (locally) best among the pool when evaluated at that k.
            """

            # ensure deterministic iteration order but preserve user's order in outputs
            k_values = list(k_values)

            # results are kept in dict form during rounds and then packed into lists at the end
            best_success_by_k = {k: -1.0 for k in k_values}
            best_povms_by_k = {}

            # pool of good parameter initializations to use as restarts.
            all_best_states: dict = {}   # k -> state_dict (tensors)
            k1_state = None              # parametric warm-start state derived from SDP for k=1

            def _clone_state() -> dict:
                return {name: param.clone().detach() for name, param in self.povm.state_dict().items()}

            def _load_init(state: Optional[dict]):
                if state is None:
                    # random init
                    with torch.no_grad():
                        for L in self.povm.L_factors:
                            L.copy_(torch.randn_like(L) * float(init_scale_random))
                else:
                    self.povm.load_state_dict(state)

            def _optimize_k_lbfgs(
                k: int,
                round_idx: int,
                extra_init: Optional[Tuple[str, dict]] = None
            ) -> Tuple[float, Optional[dict], Optional[List[np.ndarray]]]:
                """Optimize for a fixed k>1 via multi-start LBFGS.

                Returns (best_success, best_state_dict, best_povm_numpy_list).
                """
                assert k > 1

                # build candidate initializations
                candidates: List[Tuple[str, Optional[dict]]] = []

                # cross-k basin hopping if round_idx == 1
                if round_idx == 1:
                    if cross_k_restarts and len(all_best_states) > 0:
                        for kk, st in all_best_states.items():
                            if kk == k:
                                continue
                            candidates.append((f"k{kk}", st))
                    
                    else:
                        kk, st = next(reversed(all_best_states.items()))
                        candidates.append((f"k{kk}", st))

                if round_idx > 1:
                    if extra_init is not None:
                        candidates.append((extra_init[0], extra_init[1]))
                
                overall_best_success = -1.0
                overall_best_state = None

                # optimize from each candidate
                for cand_i, (label, init_state) in enumerate(candidates, start=1):
                    _load_init(init_state)

                    optimizer = optim.LBFGS(
                        self.povm.parameters(),
                        lr=lr,
                        max_iter=20,
                        line_search_fn="strong_wolfe"
                    )

                    best_success = -1.0
                    best_state = None
                    no_improve_counter = 0
                    prev_success_local = None

                    desc = f"round {round_idx}/{max_rounds} k={k} start={label} [{cand_i}/{len(candidates)}]"
                    pbar = tqdm(range(1, num_epochs + 1), desc=desc, leave=True)

                    for epoch in pbar:

                        def closure():
                            optimizer.zero_grad()
                            success_tensor = self.map_success_rate(rhos_by_class, k, priors=priors)
                            loss = -success_tensor
                            loss.backward()
                            return loss

                        _ = optimizer.step(closure)

                        with torch.no_grad():
                            success_tensor = self.map_success_rate(rhos_by_class, k, priors=priors)
                            success = float(success_tensor.item())

                        if success > best_success:
                            best_success = success
                            best_state = _clone_state()

                        # adaptive stopping (plateau): use signed improvement (only count if not improving)
                        if prev_success_local is not None:
                            improvement = success - prev_success_local
                            if improvement < improvement_tol:
                                no_improve_counter += 1
                            else:
                                no_improve_counter = 0
                            if no_improve_counter >= patience:
                                break
                        prev_success_local = success

                        # optional: gradient norm stop
                        total_grad_norm = 0.0
                        for p in self.povm.parameters():
                            if p.grad is not None:
                                total_grad_norm += (p.grad.norm().item() ** 2)
                        total_grad_norm = total_grad_norm ** 0.5
                        if total_grad_norm < grad_tol:
                            break

                        pbar.set_postfix(
                            epoch=epoch,
                            success=f"{success:.8f}",
                            best_overall=f"{overall_best_success:.8f}",
                            grad=f"{total_grad_norm:.2e}",
                        )

                    # update overall best across candidates
                    if best_success > overall_best_success and best_state is not None:
                        overall_best_success = best_success
                        overall_best_state = best_state

                if overall_best_state is None:
                    return -1.0, None, None

                # restore overall best and export POVM
                self.povm.load_state_dict(overall_best_state)
                povm_np = [E.detach().clone().cpu().numpy() for E in self.povm()]

                return overall_best_success, overall_best_state, povm_np

            # round 1 (initial optimisation)
            round_idx = 1
            for k in k_values:

                if k == 1:
                    # solve SDP for k=1 (global optimum)
                    E_list = self._solve_sdp_k1(
                        rhos_by_class,
                        priors=priors,
                        solver=sdp_solver,
                        verbose=sdp_verbose
                    )

                    # compute success for the SDP POVM (k=1)
                    E_torch = [torch.tensor(E, dtype=real_dtype, device=device) for E in E_list]

                    def compute_success_fixed(E_fixed_list):
                        if priors is None:
                            pvec = torch.tensor(
                                [1.0 / self.num_classes] * self.num_classes,
                                dtype=real_dtype,
                                device=device
                            )
                        else:
                            pvec = torch.tensor(priors, dtype=real_dtype, device=device)

                        sigmas = [torch.mean(class_rhos, dim=0) for class_rhos in rhos_by_class]

                        M = self.num_classes
                        P = torch.zeros(M, M, dtype=real_dtype, device=device)

                        for m, E_m in enumerate(E_fixed_list):
                            for c, sigma_c in enumerate(sigmas):
                                val = torch.einsum('ij,ji->', E_m, sigma_c)
                                P[c, m] = torch.real(val)

                        P = torch.clamp(P, min=0.0)
                        joint = P * pvec.view(-1, 1)
                        best_per_m, _ = torch.max(joint, dim=0)
                        success = torch.sum(best_per_m)
                        success = torch.clamp(success, min=0.0, max=1.0)
                        return float(success.item())

                    best_success = compute_success_fixed(E_torch)
                    best_success_by_k[k] = best_success
                    best_povms_by_k[k] = [E.astype(np.float64) for E in E_list]

                    # warm start parametric model from SDP solution
                    if self.warm_start:
                        self._warm_start_from_povms(
                            E_list,
                            warm_start_iters=warm_start_fit_iters,
                            fit_lr=warm_start_fit_lr,
                        )
                        k1_state = _clone_state()
                        all_best_states[k] = k1_state

                else:
                    # optimize k>1 using multi-start LBFGS
                    best_success, best_state, povm_np = _optimize_k_lbfgs(k, round_idx=round_idx, extra_init=None)

                    if best_state is not None and povm_np is not None:
                        best_success_by_k[k] = best_success
                        best_povms_by_k[k] = povm_np
                        all_best_states[k] = best_state

            # round 2 (iterative improvements)
            for round_idx in range(2, int(max_rounds) + 1):

                # Build a cross-k score table: for each target k, find best source state among all ks
                reopt_plan: List[Tuple[int, int, float, float]] = []  # (k_target, k_source, best_val, current_val)

                # Precompute evaluations (could be expensive but k pool is typically small)
                for k_tgt in k_values:
                    if k_tgt == 1:
                        continue  # keep convex optimum for k=1

                    current_val = float(best_success_by_k.get(k_tgt, -1.0))
                    best_src = None
                    best_val = -1.0

                    for k_src, state in all_best_states.items():
                        if state is None:
                            continue
                        self.povm.load_state_dict(state)
                        with torch.no_grad():
                            val = float(self.map_success_rate(rhos_by_class, k_tgt, priors=priors).item())
                        if val > best_val:
                            best_val = val
                            best_src = k_src

                    if best_src is None:
                        continue

                    if best_src != k_tgt and (best_val - current_val) > reassignment_tol:
                        reopt_plan.append((k_tgt, best_src, best_val, current_val))

                if len(reopt_plan) == 0:
                    break  # fixed point reached

                # re-optimize only the ks that want to "adopt" another k's measurement as warm start.
                # sort by how much improvement is available (largest first), then by k.
                reopt_plan.sort(key=lambda t: (-(t[2] - t[3]), t[0]))

                for k_tgt, k_src, _, _ in reopt_plan:
                    extra = (f"adopt_k{k_src}", all_best_states[k_src])
                    best_success, best_state, povm_np = _optimize_k_lbfgs(k_tgt, round_idx=round_idx, extra_init=extra)

                    if best_state is not None and povm_np is not None:
                        best_success_by_k[k_tgt] = best_success
                        best_povms_by_k[k_tgt] = povm_np
                        all_best_states[k_tgt] = best_state

                # Note: we intentionally do not update prev_best_state across this reopt batch,
                # because each k is being re-optimized independently in this phase.

            # pack results
            results = {
                "k_values": k_values,
                "success_rates": [best_success_by_k[k] for k in k_values],
                "best_povms": {k: best_povms_by_k[k] for k in k_values if k in best_povms_by_k},
            }
            return results