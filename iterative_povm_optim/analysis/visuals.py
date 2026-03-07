"""
Collection of functions for analysing optimised POVMs.
"""

from ast import Tuple

import numpy as np
import matplotlib.pyplot as plt


def get_eigenvectors(povm_matrix, tol=0.1):
    """
    Returns eigenvalues and eigenvectors of a POVM matrix above given tolerance.
    Sorted in order of descending eigenvalue.
    """

    # get values and vectors
    eigenvalues, eigenvectors = np.linalg.eigh(povm_matrix)

    # sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # keep only eigenvalues greater than tolerance
    mask = eigenvalues > tol
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]

    return eigenvalues, eigenvectors


def track_eigenvector_evolution(
    ref_vec,
    povms_for_class,
    tol=0.1,
    image_shape=(20, 20)
):
    """
    Plots the evolution of the eigenvector most similar to a reference vector across a sequence of POVM matrices.
    """

    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    tracked_vecs = []
    similarities = []

    for k in range(len(povms_for_class)):

        povm_k = povms_for_class[k]

        eigvals, eigvecs = np.linalg.eigh(povm_k)

        # sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # keep significant eigenvectors
        mask = eigvals > tol
        eigvecs = eigvecs[:, mask]

        # normalize eigenvectors
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)

        # compute sign-invariant similarity
        sims = np.abs(eigvecs.T @ ref_vec)

        best_idx = np.argmax(sims)
        best_vec = eigvecs[:, best_idx]

        # fix sign to align with reference
        if np.dot(best_vec, ref_vec) < 0:
            best_vec = -best_vec

        tracked_vecs.append(best_vec)
        similarities.append(sims[best_idx])

    # plot similarity against k
    plt.figure(figsize=(5, 3))
    plt.plot(similarities)
    plt.xlabel("k (number of copies)")
    plt.ylabel("|<v_k , v_ref>|")
    plt.title("Eigenvector similarity across k")
    plt.show()

    # plot evolution across k
    K = len(tracked_vecs)
    fig, axes = plt.subplots(1, K, figsize=(3*K, 3))

    if K == 1:
        axes = [axes]

    for k in range(K):
        axes[k].imshow(
            tracked_vecs[k].reshape(image_shape),
            cmap="RdBu",
            vmin=-0.2,
            vmax=0.2
        )
        axes[k].set_title(f"k={k}")
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()

    return tracked_vecs, similarities


def eigenvalue_spectrum_heatmap(
    povms,
    log_scale=False,
):
    """
    Plots eigenvalue spectra of a series of POVM matrices as a heatmap.
    """

    K = len(povms)

    spectra = []

    for k in range(K):
        eigvals, _ = get_eigenvectors(povms[k], tol=-1.0)
        eigvals = np.sort(eigvals)[::-1]
        spectra.append(eigvals)

    spectra = np.asarray(spectra)

    if log_scale:
        spectra = np.log10(np.clip(spectra, 1e-15, None))

    plt.figure(figsize=(8, 6))
    plt.imshow(
        spectra,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation='none'
    )
    plt.colorbar(label="Eigenvalue" if not log_scale else "log10(Eigenvalue)")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("k")
    plt.title("Eigenvalue Spectrum Evolution")
    plt.tight_layout()
    plt.show()


def projection_heatmap(
    povms,
    correct_imgs,
    wrong_imgs,
    normalize=True,
    log_scale=False,
):  
    """
    Plots the projection of the correct and a single wrong class of images onto the eigenvectors of a series of POVM matrices as heatmaps.
    """

    K = len(povms)

    weights_correct_all = []
    weights_wrong_all = []

    for k in range(K):

        eigvals, eigvecs = get_eigenvectors(povms[k], tol=-1.0)

        probs_correct = (correct_imgs @ eigvecs) ** 2
        probs_wrong = (wrong_imgs @ eigvecs) ** 2

        w_correct = probs_correct.sum(axis=0)
        w_wrong = probs_wrong.sum(axis=0)

        # sort using correct class
        idx = np.argsort(w_correct)[::-1]

        w_correct = w_correct[idx]
        w_wrong = w_wrong[idx]

        if normalize:
            w_correct = w_correct / (w_correct.sum() + 1e-15)
            w_wrong = w_wrong / (w_wrong.sum() + 1e-15)

        weights_correct_all.append(w_correct)
        weights_wrong_all.append(w_wrong)

    weights_correct_all = np.array(weights_correct_all)
    weights_wrong_all = np.array(weights_wrong_all)

    if log_scale:
        weights_correct_all = np.log10(np.clip(weights_correct_all, 1e-15, None))
        weights_wrong_all = np.log10(np.clip(weights_wrong_all, 1e-15, None))

    # independent color ranges
    vmin_correct, vmax_correct = weights_correct_all.min(), weights_correct_all.max()
    vmin_wrong, vmax_wrong = weights_wrong_all.min(), weights_wrong_all.max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    im1 = axes[0].imshow(
        weights_correct_all,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        vmin=vmin_correct,
        vmax=vmax_correct,
        interpolation='none'
    )
    axes[0].set_title("Correct Class")
    axes[0].set_xlabel("Eigenvector index (sorted by Correct Class)")
    axes[0].set_ylabel("k")

    im2 = axes[1].imshow(
        weights_wrong_all,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        vmin=vmin_wrong,
        vmax=vmax_wrong,
        interpolation='none'
    )
    axes[1].set_title("Wrong Class")
    axes[1].set_xlabel("Eigenvector index (sorted by Correct Class)")

    # separate colorbars
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label("Projection weight (Correct)")

    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label("Projection weight (Wrong)")

    plt.tight_layout()
    plt.show()


from ipywidgets import widgets
from IPython.display import display

def povm_viewer(
    povms,
    imgs,
    image_shape: Tuple,
    D=4,
    tol=0.1
):
    """
    Interactive viewer for the eigenvectors of a series of POVM matrices, ranked by their projection weight on a set of images.
    """

    K = len(povms)

    def update(k):
        plt.close("all")

        povm_k = povms[k]

        # get eigenvectors above tolerance
        eigvals, eigvecs = get_eigenvectors(povm_k, tol=tol)

        # calculate projection weights and rank eigenvectors by importance
        inner_products = imgs @ eigvecs
        probabilities = inner_products**2

        weights = probabilities.sum(axis=0)
        idx = np.argsort(weights)[::-1]

        eigvals_k = eigvals[idx]
        eigvecs_k = eigvecs[:, idx]

        # plot
        fig, axes = plt.subplots(D, D, figsize=(5, 5))

        for i in range(D**2):
            row = i // D
            col = i % D

            if i < eigvecs_k.shape[1]:
                axes[row, col].imshow(
                    eigvecs_k[:, i].reshape(image_shape),
                    cmap="RdBu",
                    vmin=-0.2,
                    vmax=0.2
                )
            else:
                axes[row, col].axis("off")

            axes[row, col].tick_params(labelsize=6)
            axes.flatten()[i].axis("off")

        plt.tight_layout()
        plt.show()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=K-1,
        step=1,
        description='k:',
        continuous_update=False
    )

    widgets.interact(update, k=slider)


def histogram_viewer(
    povms,
    correct_imgs,
    wrong_imgs,
    D=4,
    tol=0.1,
):  
    """
    Interactive viewer for the distribution of projection weights of a set of images onto the eigenvectors of a series of POVM matrices.
    """

    K = len(povms)

    def update(k):
        plt.close("all")

        povm_k = povms[k]

        # get eigenvectors above tolerance
        eigvals, eigvecs = get_eigenvectors(povm_k, tol=tol)

        # calculate projection weights and rank eigenvectors by importance
        inner_products = correct_imgs @ eigvecs
        probabilities = inner_products**2

        error_products = wrong_imgs @ eigvecs
        error_probabilities = error_products**2

        weights = probabilities.sum(axis=0)
        idx = np.argsort(weights)[::-1]
        eigvals_k = eigvals[idx]
        probabilities = probabilities[:, idx]
        error_probabilities = error_probabilities[:, idx]

        # plot
        fig, axes = plt.subplots(D, D, figsize=(10, 10))

        for i in range(D**2):
            row = i // D
            col = i % D
            ax = axes[row, col]

            if i < probabilities.shape[1]:

                ax.hist(
                    probabilities[:, i],
                    bins=100,
                    range=(0, 1),
                    alpha=0.6,
                    label="Correct class",
                    density=True,
                )

                ax.hist(
                    error_probabilities[:, i],
                    bins=100,
                    range=(0, 1),
                    alpha=0.6,
                    label="Wrong class",
                    density=True,
                )

                ax.set_xlim(0, 1)
                ax.set_yscale("log")
                ax.set_box_aspect(1)
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=K-1,
        step=1,
        description="k:",
        continuous_update=False
    )

    widgets.interact(update, k=slider)

