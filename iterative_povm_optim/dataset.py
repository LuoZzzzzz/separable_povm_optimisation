import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
from tqdm import tqdm

from .core.povm_cholesky_lbfgs_rounds import real_dtype, cplx_dtype


class RhoDataset:
    """
    Loads MNIST (raw), and lazily applies: remove white borders, gaussian blur,
    and downsampling. Only the requested subset(s) are transformed on demand.
    """

    def __init__(
        self,
        root: str = "./data",
        image_size: Tuple[int, int] = (12, 12),
        train: bool = True,
        device: Optional[torch.device] = None,
        real_dtype: torch.dtype = real_dtype,
        complex_dtype: torch.dtype = cplx_dtype,
        download: bool = True,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
        crop_threshold: float = 0.05,
    ):
        self.root = root
        self.image_size = image_size
        self.train = train
        self.device = device or torch.device("cpu")
        self.real_dtype = real_dtype
        self.complex_dtype = complex_dtype

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.crop_threshold = crop_threshold

        # d is fixed after resize -> H * W
        self.d = self.image_size[0] * self.image_size[1]

        self._load_dataset(download)

    def density_matrices_by_class(
        self,
        classes: List[int],
        n_per_class: Optional[int] = None,
    ):
        """
        For each class, process only that class's images, convert to normalized
        vectors, and compute density matrices.
        """
        rhos_by_class = []
        vectors_by_class = []

        for c in tqdm(classes,
                      desc="Creating density matrices...",
                      leave=True):

            idx = (self.labels == c).nonzero(as_tuple=True)[0]
            if n_per_class is not None:
                idx = idx[:n_per_class]

            if idx.numel() == 0:
                # empty
                rhos_by_class.append(torch.empty((0, self.d, self.d), dtype=self.complex_dtype, device=self.device))
                vectors_by_class.append(torch.empty((0, self.d), dtype=self.real_dtype, device=self.device))
                continue

            vecs = self._process_images(idx)  # (n, d) real dtype, normalized

            # create pure-state density matrices: |v><v|
            rhos = torch.einsum("ni,nj->nij", vecs, vecs.conj()).to(dtype=self.complex_dtype)

            rhos_by_class.append(rhos)
            vectors_by_class.append(vecs)

        return rhos_by_class, vectors_by_class

    def flattened_vectors(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return flattened normalized vectors for `indices`. If indices is None,
        processes all images lazily.
        """
        if indices is None:
            all_idx = torch.arange(len(self.raw_images), device=self.device)
            return self._process_images(all_idx)
        else:
            return self._process_images(indices.to(device=self.device))

    def labels_tensor(self) -> torch.Tensor:
        return self.labels

    def dimension(self) -> int:
        return self.d


    def _load_dataset(self, download: bool):
        """
        Load MNIST and store the raw tensors (N,H,W) without running
        the heavy transforms. Labels are stored on the configured device.
        """
        base_transform = transforms.ToTensor()

        dataset = datasets.MNIST(
            root=self.root,
            train=self.train,
            download=download,
            transform=base_transform,
        )

        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        images, labels = next(iter(loader))

        images = images.squeeze(1).to(dtype=self.real_dtype)  # keep device local; move to self.device later
        labels = labels.to(device=self.device)

        self.raw_images = images.to(device=self.device)
        self.labels = labels

        # self.d already set from image_size (post-resize). We don't compute vectors yet.

    def _process_images(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Process only the images at `indices`:
        - remove white border (crop)
        - resize to image_size
        - gaussian blur
        - flatten and normalize
        Returns: (len(indices), d) tensor on self.device with dtype self.real_dtype
        """
        if indices.numel() == 0:
            return torch.empty((0, self.d), dtype=self.real_dtype, device=self.device)

        processed_list = []

        # make a plain Python list of ints for robust indexing (avoids device/index issues)
        indices_list = [int(i) for i in indices.flatten()]

        for i in indices_list:
            img = self.raw_images[i]  # already on self.device and real dtype
            img = self._remove_white_border(img)
            img = self._resize(img)
            img = self._gaussian_blur(img)
            processed_list.append(img)

        images = torch.stack(processed_list)  # (n, H, W) maybe not exactly image_size in edge cases

        # ensure final spatial size exactly matches self.image_size (H, W).
        if images.ndim == 3:
            n, H, W = images.shape
            target_h, target_w = self.image_size
            if (H != target_h) or (W != target_w):
                # add channel dim for interpolate, then remove
                images = F.interpolate(
                    images.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False
                ).squeeze(1)
        else:
            raise RuntimeError("Unexpected image tensor shape after stacking processed images.")

        N, H, W = images.shape
        if H * W != self.d:
            raise AssertionError(f"Mismatch between computed dimension ({H}x{W}={H*W}) and image_size ({self.d}).")

        vectors = images.reshape(N, self.d)
        norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)

        vectors = vectors / norms  # normalized (real dtype)

        return vectors.to(dtype=self.real_dtype, device=self.device)

    def _remove_white_border(self, img: torch.Tensor) -> torch.Tensor:
        """
        Crops away near-zero (white) borders dynamically.
        """
        mask = img > self.crop_threshold

        if not mask.any():
            return F.interpolate(
                img.unsqueeze(0).unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # fallback: resize empty image to standard size

        coords = mask.nonzero(as_tuple=False)
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values

        return img[y_min:y_max + 1, x_min:x_max + 1]

    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        img = F.interpolate(
            img,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return img.squeeze(0).squeeze(0)

    def _gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian blur using depthwise convolution.
        Ensures kernel matches image dtype and device.
        """
        k = self.blur_kernel_size
        sigma = self.blur_sigma

        device = img.device
        dtype = img.dtype

        coords = torch.arange(k, device=device, dtype=dtype) - k // 2
        grid = coords[:, None] ** 2 + coords[None, :] ** 2
        kernel = torch.exp(-grid / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        kernel = kernel.view(1, 1, k, k)

        img = img.unsqueeze(0).unsqueeze(0)
        img = F.conv2d(img, kernel, padding=k // 2)

        return img.squeeze(0).squeeze(0)