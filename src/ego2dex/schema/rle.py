"""COCO RLE (run-length encoding) for instance masks.

Primary path uses ``pycocotools`` (a core dep) and produces COCO's *compressed*
counts string. A pure-python *uncompressed* fallback (column-major run lengths
as a list of ints) keeps ``import ego2dex`` working even if pycocotools is
absent -- it interoperates with ``pycocotools.frPyObjects`` later.

An RLE dict is ``{"size": [h, w], "counts": <str|list[int]>}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

RLE = dict[str, Any]


def _have_pycocotools() -> bool:
    try:
        import pycocotools.mask  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Pure-python (uncompressed) fallback -- column-major (Fortran) run lengths.
# --------------------------------------------------------------------------- #
def _encode_uncompressed(binary: NDArray[np.uint8]) -> RLE:
    h, w = binary.shape
    flat = np.asfortranarray(binary).astype(bool).flatten(order="F")
    # COCO counts start with the length of the initial run of 0s.
    counts: list[int] = []
    prev = False  # implicit leading "0" value
    run = 0
    for val in flat:
        if bool(val) == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = bool(val)
    counts.append(run)
    return {"size": [int(h), int(w)], "counts": counts}


def _decode_uncompressed(rle: RLE) -> NDArray[np.uint8]:
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        if val:
            flat[idx : idx + run] = 1
        idx += run
        val ^= 1
    return np.asarray(flat.reshape((w, h)).T, dtype=np.uint8)  # un-Fortran


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def encode_mask(binary: NDArray) -> RLE:
    """Encode an ``(H, W)`` binary mask to a COCO RLE dict.

    With pycocotools -> compressed ``counts`` as a UTF-8 ``str``.
    Without it     -> uncompressed ``counts`` as ``list[int]``.
    """
    binary = (np.asarray(binary) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"encode_mask expects a 2D mask, got shape {binary.shape}")
    if _have_pycocotools():
        import pycocotools.mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(binary))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}
    return _encode_uncompressed(binary)


def decode_mask(rle: RLE) -> NDArray[np.uint8]:
    """Decode a COCO RLE dict back to an ``(H, W)`` ``uint8`` binary mask."""
    counts = rle["counts"]
    if isinstance(counts, (list, tuple)):
        return _decode_uncompressed(rle)
    if not _have_pycocotools():
        raise ImportError(
            "Decoding compressed RLE requires pycocotools. "
            "Install it with `pip install pycocotools`."
        )
    import pycocotools.mask as mask_utils

    counts_bytes = counts.encode("ascii") if isinstance(counts, str) else counts
    obj = {"size": list(rle["size"]), "counts": counts_bytes}
    return np.asarray(mask_utils.decode(obj), dtype=np.uint8)


def mask_area(rle: RLE) -> int:
    """Number of foreground pixels."""
    if _have_pycocotools() and not isinstance(rle["counts"], (list, tuple)):
        import pycocotools.mask as mask_utils

        counts = rle["counts"]
        counts_bytes = counts.encode("ascii") if isinstance(counts, str) else counts
        return int(mask_utils.area({"size": list(rle["size"]), "counts": counts_bytes}))
    return int(decode_mask(rle).sum())


def mask_to_bbox(rle: RLE) -> list[float]:
    """Tight COCO bbox ``[x, y, w, h]`` of the mask."""
    m = decode_mask(rle)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
