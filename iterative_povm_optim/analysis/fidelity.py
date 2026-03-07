import numpy as np
import matplotlib.pyplot as plt

def sqrt_psd(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 0, None)  # remove tiny negatives
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T

# Uhlmann fidelity
def povm_element_fidelity(E1, E2):
    E1n = E1 / np.trace(E1)
    E2n = E2 / np.trace(E2)

    sqrtE1 = sqrt_psd(E1n)
    middle = sqrtE1 @ E2n @ sqrtE1
    sqrt_middle = sqrt_psd(middle)

    return np.real(np.trace(sqrt_middle)**2)

def plot_povm_fidelity(povm1, povm2):
    """
    Plots real parts of two POVM elements and their difference, along with the fidelity.
    """

    # compute difference and fidelity
    diff = povm2 - povm1
    fidelity = povm_element_fidelity(povm1, povm2)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # povm 1
    im0 = axes[0].imshow(np.real(povm1))
    axes[0].set_title(f"POVM 1")
    plt.colorbar(im0, ax=axes[0])

    # povm 2
    im1 = axes[1].imshow(np.real(povm2))
    axes[1].set_title(f"POVM 2")
    plt.colorbar(im1, ax=axes[1])

    # difference
    im2 = axes[2].imshow(np.real(diff))
    axes[2].set_title(f"Fidelity = {fidelity:.6f}")
    plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()