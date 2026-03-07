import torch
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from .povm_cholesky_lbfgs_rounds import real_dtype

# device and complex dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_random_pure_states(d: int, n_states: int, n_classes: int,
                               separation: float = 0.3, seed: Optional[int] = None) -> List[torch.Tensor]:
    """
    Generate list of length n_classes containing tensors rhos of shape (n_states, d, d) complex.
    Adds a class-dependent bias to introduce separability.
    """
    if seed is not None:
        torch.manual_seed(seed)

    rhos_by_class = []
    for c in range(n_classes):
        # complex normal entries
        real = torch.randn(n_states, d, dtype=real_dtype, device=device)
        imag = torch.randn(n_states, d, dtype=real_dtype, device=device)
        psi = real + 1j * imag

        # Add class-specific bias in some component to separate classes
        if c < d:
            bias = (separation * (1.0 + 1.0j))
            psi[:, c] = psi[:, c] + bias

        # Normalize states
        norms = torch.linalg.norm(psi, dim=1, keepdim=True)
        psi = psi / norms

        # Convert to density matrices
        rhos = torch.einsum('ni,nj->nij', psi, psi.conj())  # (n_states, d, d) complex
        rhos_by_class.append(rhos)

    return rhos_by_class

  
def plot_top_eigenvectors(povm_matrix, num_vectors=10):
    # eigen-decomposition (use eigh since POVM should be Hermitian)
    eigenvalues, eigenvectors = np.linalg.eigh(povm_matrix)

    # sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # plot top eigenvectors
    fig, axes = plt.subplots(1, num_vectors, figsize=(3 * num_vectors, 3))

    # get dimensions
    img_dim = int(np.sqrt(len(eigenvectors[:, 0])))

    for i in range(num_vectors):
        img = eigenvectors[:, i].reshape(img_dim, img_dim)

        ax = axes[i]
        im = ax.imshow(img, cmap="RdBu")
        ax.set_title(f"Eigenvalue {i+1}\n{eigenvalues[i]:.3e}")
        ax.axis("off")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()