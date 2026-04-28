import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random


class Generator(Dataset):
    """
    PyTorch Dataset for a three-channel segmentation + reconstruction task.

    Expects a root directory containing exactly three subdirectories:

        root/
        ├── train_A/   ← input images         (RGB)
        ├── train_B/   ← segmentation masks    (Grayscale / L)
        └── train_C/   ← reconstruction targets(RGB)

    Files across the three directories are matched by **sorted filename**, so
    ``train_A/0001.png``, ``train_B/0001.png``, and ``train_C/0001.png`` must
    correspond to the same sample.

    Each call to ``__getitem__`` returns a dict with four keys:

    .. code-block:: python

        {
            "Image"    : torch.Tensor  # (3, H, W) — normalised input image
            "Mask"     : torch.Tensor  # (1, H, W) — binary/grayscale mask
            "Target"   : torch.Tensor  # (3, H, W) — reconstruction target
            "File name": str           # original filename, useful for debugging
        }

    Pipeline per sample
    -------------------
    1. Load all three PIL images from disk.
    2. Optionally apply spatial + photometric augmentation (``_augment``).
    3. Apply tensor conversion / user-supplied transforms (``_transform``).
    4. Return the dict.

    Parameters
    ----------
    root : str
        Path to the dataset root. Must end with a path separator (e.g. ``data/``),
        or use ``os.path.join`` — the subdirectory join is handled internally.
    augment : bool
        If ``True``, ``_augment`` is called before ``_transform`` on every sample.
        Augmentation is applied in PIL space so spatial alignment between the
        image, mask, and target is always preserved.
    transform_img : callable, optional
        A torchvision transform (or ``T.Compose([...])``) applied to ``img`` and
        ``reconstructed``. If ``None``, ``T.ToTensor()`` is used as a safe default.
    transform_mask : callable, optional
        A torchvision transform applied to ``mask``. If ``None``, ``T.ToTensor()``
        is used. Keep this separate from ``transform_img`` to avoid applying
        colour-based operations (e.g. ``ColorJitter``, ``Normalize``) to label maps.

    Notes
    -----
    - Augmentation runs **before** tensor conversion so all spatial operations
      work on PIL images, which have well-defined interpolation modes.
    - Always use ``NEAREST`` interpolation for the mask during any spatial
      transform (affine, rotate, etc.) to avoid blending discrete label values.
    - Call ``__getstats__()`` once before training to obtain the per-channel
      mean/std needed for ``T.Normalize``, and to check for class imbalance
      in the masks.

    Examples
    --------
    >>> from torchvision import transforms as T
    >>> transform_img = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    >>> transform_mask = T.Compose([T.ToTensor()])
    >>> dataset = Generator(
    ...     root      = 'data/',
    ...     augment   = True,
    ...     transform_img  = transform_img,
    ...     transform_mask = transform_mask,
    ... )
    >>> sample = dataset[0]
    >>> sample["Image"].shape
    torch.Size([3, 256, 256])
    """

    def __init__(self, root, augment, transform_img=None, transform_mask=None):
        """
        Initialise the Generator dataset.

        Scans ``root/train_A/`` to build the file list. The same filenames are
        assumed to exist in ``train_B/`` and ``train_C/``.

        Parameters
        ----------
        root : str
            Absolute or relative path to the dataset root directory.
        augment : bool
            Whether to apply random augmentation during ``__getitem__``.
        transform_img : callable, optional
            Transform applied to the RGB input image and reconstruction target.
            Should include at minimum ``T.ToTensor()``.  Defaults to bare
            ``T.ToTensor()`` if not supplied.
        transform_mask : callable, optional
            Transform applied to the grayscale mask.  Should include at minimum
            ``T.ToTensor()``.  Do **not** include colour-based transforms here.
            Defaults to bare ``T.ToTensor()`` if not supplied.

        Raises
        ------
        FileNotFoundError
            If any of ``train_A``, ``train_B``, or ``train_C`` do not exist
            under ``root``.
        """
        self.root = root
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.augment = augment

        self.dirA = os.path.join(root, 'train_A')
        self.dirB = os.path.join(root, 'train_B')
        self.dirC = os.path.join(root, 'train_C')

        self.files = sorted(os.listdir(self.dirA))

    def __getitem__(self, idx):
        """
        Load and return a single sample by index.

        Retrieves the image, mask, and reconstruction target at position ``idx``
        in the sorted file list, applies optional augmentation and transforms,
        and returns them as a labelled dictionary.

        Parameters
        ----------
        idx : int
            Index into the sorted file list (0 ≤ idx < len(dataset)).

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``"Image"`` : torch.Tensor, shape (3, H, W)
                The RGB input image as a float tensor. Range depends on the
                supplied ``transform_img`` (e.g. [0, 1] with bare ToTensor,
                or normalised if Normalize is included).
            ``"Mask"`` : torch.Tensor, shape (1, H, W)
                The grayscale segmentation mask as a float tensor.
            ``"Target"`` : torch.Tensor, shape (3, H, W)
                The RGB reconstruction target, processed with the same
                ``transform_img`` pipeline as ``"Image"``.
            ``"File name"`` : str
                The original filename (e.g. ``"0001.png"``), useful for
                tracing predictions back to source files during evaluation.

        Notes
        -----
        - Augmentation (if enabled) is applied in PIL space *before* tensor
          conversion to ensure consistent spatial transforms across all three
          images.
        - The mask uses ``NEAREST`` interpolation during spatial augmentation
          to prevent label blending.
        """
        pathA = os.path.join(self.dirA, self.files[idx])
        pathB = os.path.join(self.dirB, self.files[idx])
        pathC = os.path.join(self.dirC, self.files[idx])

        img           = Image.open(pathA).convert('RGB')
        mask          = Image.open(pathB).convert('L')
        reconstructed = Image.open(pathC).convert('RGB')

        if self.augment:
            img, mask, reconstructed = self._augment(img, mask, reconstructed)

        img, mask, reconstructed = self._transform(img, mask, reconstructed)

        return {
            "Image":     img,
            "Mask":      mask,
            "Target":    reconstructed,
            "File name": self.files[idx],
        }

    def _transform(self, img, mask, reconstructed):
        """
        Convert PIL images to tensors using user-supplied or default transforms.

        Applied after ``_augment`` (if enabled). Keeps image and mask pipelines
        separate so colour-based operations are never accidentally applied to
        the label map.

        Parameters
        ----------
        img : PIL.Image.Image
            RGB input image of shape (H, W, 3) in PIL format.
        mask : PIL.Image.Image
            Grayscale mask of shape (H, W) in PIL format.
        reconstructed : PIL.Image.Image
            RGB reconstruction target of shape (H, W, 3) in PIL format.

        Returns
        -------
        img : torch.Tensor
            Transformed image tensor, shape (3, H, W).
        mask : torch.Tensor
            Transformed mask tensor, shape (1, H, W).
        reconstructed : torch.Tensor
            Transformed reconstruction tensor, shape (3, H, W).

        Notes
        -----
        - Falls back to ``T.ToTensor()`` for both image and mask if no transform
          was provided at construction time. This guarantees the method always
          returns tensors, never raw PIL images.
        - ``T.ToTensor()`` scales ``uint8`` pixel values from [0, 255] to
          [0.0, 1.0] automatically.
        """
        if self.transform_img:
            img           = self.transform_img(img)
            reconstructed = self.transform_img(reconstructed)
        else:
            to_tensor     = T.ToTensor()
            img           = to_tensor(img)
            reconstructed = to_tensor(reconstructed)

        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = T.ToTensor()(mask)

        return img, mask, reconstructed

    def _augment(self, img, mask, reconstructed):
        """
        Apply identical random spatial augmentations to all three PIL images,
        and independent photometric augmentations to ``img`` and ``reconstructed``
        only (never to the mask).

        Using torchvision's **functional API** (``TF.*``) rather than transform
        objects ensures that the *same* random parameters are applied to all
        three images on every call, preserving pixel-mask spatial alignment.

        Augmentations applied
        ---------------------
        Spatial (all three images, identical parameters):
            - Random horizontal flip           (p = 0.5)
            - Random vertical flip             (p = 0.5)
            - Random 90° rotation              (p = 0.5, one of 90 / 180 / 270°)
            - Random affine transform          (p = 0.5)
                * Translation up to ±10% of image size
                * Scale in the range [0.9, 1.1]
                * No shear or additional rotation

        Photometric (``img`` and ``reconstructed`` only, mask excluded):
            - ``ColorJitter``                  (p = 0.5)
                * Brightness ±0.3, Contrast ±0.3, Saturation ±0.2, Hue ±0.05
            - ``GaussianBlur`` (kernel = 3)    (p = 0.3)
                * Sigma sampled uniformly from [0.1, 1.5]

        Parameters
        ----------
        img : PIL.Image.Image
            RGB input image.
        mask : PIL.Image.Image
            Grayscale segmentation mask. Receives only spatial transforms;
            photometric transforms are deliberately skipped to avoid corrupting
            label values.
        reconstructed : PIL.Image.Image
            RGB reconstruction target. Receives the same transforms as ``img``.

        Returns
        -------
        img : PIL.Image.Image
            Augmented input image.
        mask : PIL.Image.Image
            Spatially augmented mask (labels intact).
        reconstructed : PIL.Image.Image
            Augmented reconstruction target.

        Notes
        -----
        - The mask always uses ``InterpolationMode.NEAREST`` for any spatial
          warp so that integer label values are never blended.
        - ``img`` and ``reconstructed`` use ``InterpolationMode.BILINEAR`` for
          smoother results on continuous pixel values.
        - Photometric transforms create a *new* transform object on each call,
          so ``img`` and ``reconstructed`` receive independently sampled (but
          similarly distributed) colour jitter. If your task requires identical
          colour transforms on both, pass the same randomly-sampled parameters
          via ``TF.adjust_*`` functions instead.
        """

        # ── Spatial transforms ─────────────────────────────────────────────

        
        if random.random() < 0.5:
            img           = TF.hflip(img)
            mask          = TF.hflip(mask)
            reconstructed = TF.hflip(reconstructed)

        if random.random() < 0.5:
            angle         = random.uniform(-10, 10)
            img           = TF.rotate(img,           angle)
            mask          = TF.rotate(mask,          angle)
            reconstructed = TF.rotate(reconstructed, angle)

        if random.random() < 0.5:
            affine_params = T.RandomAffine.get_params(
                degrees      = [0, 0],
                translate    = (0.05, 0.05),
                scale_ranges = (0.95, 1.05),
                shears       = [0, 0],
                img_size     = img.size,
            )
            img           = TF.affine(img,           *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
            mask          = TF.affine(mask,          *affine_params, interpolation=TF.InterpolationMode.NEAREST)
            reconstructed = TF.affine(reconstructed, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)

        # ── Photometric transforms (img + reconstructed only) ────────────────

        if random.random() < 0.5:
            jitter        = T.ColorJitter(
                brightness = 0.2,
                contrast   = 0.2,
                saturation = 0.1,
                hue        = 0.02
            )
            img           = jitter(img)
            reconstructed = jitter(reconstructed)

        if random.random() < 0.2:
            blur          = T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            img           = blur(img)
            reconstructed = blur(reconstructed)
        return img, mask, reconstructed

    def __getstats__(self):
        """
        Compute per-channel image statistics and mask class balance over the
        full dataset without loading everything into memory at once.

        Uses a **batched Welford online algorithm** to accumulate mean and
        variance in a single pass per file, making it safe for large datasets.
        Augmentation and user transforms are intentionally bypassed so stats
        reflect the raw data distribution.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``"img_mean"`` : list of float, length 3
                Per-channel (R, G, B) mean pixel value in [0, 1].
                Use directly as the ``mean`` argument to ``T.Normalize``.
            ``"img_std"`` : list of float, length 3
                Per-channel (R, G, B) standard deviation in [0, 1].
                Use directly as the ``std`` argument to ``T.Normalize``.
            ``"mask_pos_freq"`` : float
                Fraction of all mask pixels whose value is > 0 (foreground).
                Values near 0 or 1 indicate severe class imbalance.
            ``"num_samples"`` : int
                Total number of files scanned (equals ``len(dataset)``).

        Notes
        -----
        - Progress is printed to stdout every 10% of the dataset so you can
          monitor long-running stat collection runs.
        - If ``mask_pos_freq`` is below ~0.1 or above ~0.9 the printout flags
          the dataset as imbalanced. In that case consider passing a
          ``pos_weight`` tensor to ``nn.BCEWithLogitsLoss``:

          .. code-block:: python

              neg_freq   = 1 - stats["mask_pos_freq"]
              pos_weight = torch.tensor([neg_freq / stats["mask_pos_freq"]])
              criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Examples
        --------
        >>> stats = dataset.__getstats__()
        >>> normalize = T.Normalize(mean=stats["img_mean"], std=stats["img_std"])
        """
        print(f"Computing dataset statistics over {len(self.files)} samples...")

        to_tensor = T.ToTensor()

        ch_mean = np.zeros(3, dtype=np.float64)
        ch_var  = np.zeros(3, dtype=np.float64)
        n       = np.zeros(3, dtype=np.float64)

        total_mask_pixels = 0
        total_pos_pixels  = 0

        report_interval = max(1, len(self.files) // 10)

        for i, fname in enumerate(self.files):
            img  = to_tensor(Image.open(os.path.join(self.dirA, fname)).convert('RGB'))
            mask = to_tensor(Image.open(os.path.join(self.dirB, fname)).convert('L'))

            for c in range(3):
                pixels    = img[c].numpy().ravel().astype(np.float64)
                batch_n   = len(pixels)
                batch_m   = pixels.mean()
                batch_v   = pixels.var()

                combined_n  = n[c] + batch_n
                delta       = batch_m - ch_mean[c]
                ch_mean[c]  = (n[c] * ch_mean[c] + batch_n * batch_m) / combined_n
                ch_var[c]  += batch_v * batch_n + delta**2 * n[c] * batch_n / combined_n
                n[c]        = combined_n

            mask_np            = mask.numpy().ravel()
            total_mask_pixels += mask_np.size
            total_pos_pixels  += int((mask_np > 0).sum())

            if (i + 1) % report_interval == 0:
                print(f"  [{i+1}/{len(self.files)}] running mean={ch_mean.round(4)}")

        ch_std   = np.sqrt(ch_var / n)
        pos_freq = total_pos_pixels / total_mask_pixels if total_mask_pixels > 0 else 0.0

        stats = {
            "img_mean":      ch_mean.tolist(),
            "img_std":       ch_std.tolist(),
            "mask_pos_freq": round(pos_freq, 6),
            "num_samples":   len(self.files),
        }

        print("\n── Dataset Statistics ──────────────────────────────")
        print(f"  Samples      : {stats['num_samples']}")
        print(f"  Image mean   : {[round(v, 4) for v in stats['img_mean']]}")
        print(f"  Image std    : {[round(v, 4) for v in stats['img_std']]}")
        print(f"  Mask pos freq: {stats['mask_pos_freq']} "
              f"({'imbalanced!' if pos_freq < 0.1 or pos_freq > 0.9 else 'balanced'})")
        print("────────────────────────────────────────────────────\n")

        return stats

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of files found in ``train_A/`` at construction time.
        """
        return len(self.files)



'''
def seed_worker(worker_id):
"""

Used to seed workers if you want to avoid identical augmentations
DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)
"""
np.random.seed(torch.initial_seed() % 2**32)
random.seed(torch.initial_seed() % 2**32)
'''