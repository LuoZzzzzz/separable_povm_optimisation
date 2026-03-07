import numpy as np
from scipy.special import gammaln


def compute_success_vs_k(
    povm_list,
    states_by_class,
    k_list,
    priors=None,
):
    """
    Parameters
    ----------
    povm_list : list of (d,d) np.ndarray
    states_by_class : list of (N_c, d) np.ndarray
    k_max : int
    priors : optional list of floats

    Returns
    -------
    dict with:
        "k_values"
        "overall_success"
        "per_class_success"
    """

    num_classes = len(povm_list)

    if priors is None:
        priors = np.ones(num_classes) / num_classes
    priors = np.asarray(priors)

    # precompute probabilities for each state and class
    probs_by_class = []  # list of arrays (N_c, C)

    for c in range(num_classes):
        psi = states_by_class[c]  # (N_c, d)

        N_c = psi.shape[0]
        probs = np.zeros((N_c, num_classes), dtype=np.float64)

        for m, E in enumerate(povm_list):
            # Works for real or complex
            probs[:, m] = np.einsum(
                "ni,ij,nj->n",
                psi.conj(),
                E,
                psi
            ).real

        probs = np.clip(probs, 1e-15, None)
        probs /= probs.sum(axis=1, keepdims=True)

        probs_by_class.append(probs)

    # generate all possible count vectors for k measurements and M classes
    def generate_counts(k, M):
        counts = []

        def helper(rem, m_left, prefix):
            if m_left == 1:
                counts.append(prefix + [rem])
                return
            for i in range(rem + 1):
                helper(rem - i, m_left - 1, prefix + [i])

        helper(k, M, [])
        return np.array(counts, dtype=int)

    results = {
        "k_values": k_list,
        "overall_success": [],
        "per_class_success": {c: [] for c in range(num_classes)},
    }

    for k in k_list:

        counts = generate_counts(k, num_classes)  # (S, C)
        S = counts.shape[0]

        # multinomial coefficient term
        log_coeff = (
            gammaln(k + 1)
            - np.sum(gammaln(counts + 1), axis=1)
        )  # (S,)

        likelihoods = np.zeros((num_classes, S))

        # for each class: average of multinomials
        for c in range(num_classes):

            probs = probs_by_class[c]  # (N_c, C)

            # log multinomial for each state:
            # log P(s | state_i) = log_coeff + counts @ log(p_i)
            log_p = np.log(probs)  # (N_c, C)

            # shape: (N_c, S)
            log_terms = log_coeff[None, :] + log_p @ counts.T

            # exponentiate and average over states
            likelihoods[c] = np.exp(log_terms).mean(axis=0)

        joint = likelihoods * priors[:, None]

        winner = np.argmax(joint, axis=0)
        best = joint[winner, np.arange(joint.shape[1])]
        overall_success = best.sum()

        results["overall_success"].append(overall_success)

        # per-class success
        for c in range(num_classes):
            class_success = joint[c, winner == c].sum() / priors[c]
            results["per_class_success"][c].append(class_success)

    return results