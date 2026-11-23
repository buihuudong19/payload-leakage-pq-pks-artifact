"""Read-only access to the image data.

(1) Synthetic test covers and test messages for the unit tests and the demonstration script. They are generated
    deterministically from a seed, so the tests need no image data; no experiment of the paper uses them.
(2) BOSSbase 1.01 for the experiments. Default folder <repository>/data/external/bossbase, or the environment
    variable PQPKS_BOSSBASE_DIR. Pilot image list: data/external/_pilot/pilot_bossbase_ids.txt (1000 images,
    seed 20261029, excluding the four byte-identical pairs 5341-5344/5536-5539).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_COVER_NAMES = tuple(f"T{k}" for k in range(1, 9))
TEST_MESSAGE_BYTES = (919, 2420)      # lengths of the two binary messages used in E4 and in the tests


class MissingResource(FileNotFoundError):
    """A third-party input that is not redistributed with the repository (images or reference code)."""


# ---------------------------------------------------------------------------- synthetic test data
def synthetic_cover(seed: int, shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Deterministic 8-bit grey-scale test cover: large smooth structures, medium-scale texture and fine noise, with
    values kept in [3, 252] so that no pixel is saturated. For functional tests only."""
    rng = np.random.default_rng(seed)
    h, w = shape

    def layer(cells: int, amp: float) -> np.ndarray:
        g = Image.fromarray(rng.standard_normal((cells, cells)).astype(np.float32), mode="F")
        return amp * np.asarray(g.resize((w, h), Image.BICUBIC), dtype=np.float64)

    x = 128.0 + layer(6, 45.0) + layer(48, 18.0) + rng.normal(0.0, 5.0, size=(h, w))
    return np.clip(np.rint(x), 3, 252).astype(np.uint8)


def synthetic_rgb(seed: int, shape: tuple[int, int] = (512, 512)) -> Image.Image:
    """RGB test image whose three planes are independent synthetic covers."""
    planes = [synthetic_cover(1000 * seed + c, shape) for c in range(3)]
    return Image.fromarray(np.stack(planes, axis=2), mode="RGB")


def cover_pool() -> dict[str, np.ndarray]:
    """Eight synthetic 512 x 512 test covers, named T1 to T8."""
    return {name: synthetic_cover(k) for k, name in enumerate(TEST_COVER_NAMES, start=1)}


def test_messages() -> tuple[bytes, ...]:
    """Two binary test messages of TEST_MESSAGE_BYTES bytes (deterministic pseudorandom bytes)."""
    return tuple(np.random.default_rng(20 + k).integers(0, 256, size=n, dtype=np.uint8).tobytes()
                 for k, n in enumerate(TEST_MESSAGE_BYTES))


# ---------------------------------------------------------------------------- BOSSbase
def bossbase_dir() -> Path:
    env = os.environ.get("PQPKS_BOSSBASE_DIR")
    p = Path(env) if env else PROJECT_ROOT / "data" / "external" / "bossbase"
    if not (p / "1.pgm").is_file():                  # the folder may hold only MANIFEST.sha256
        raise MissingResource(f"BOSSbase not found: {p} (set PQPKS_BOSSBASE_DIR, see docs/data.md)")
    return p


def load_bossbase(image_id: int) -> np.ndarray:
    x = np.asarray(Image.open(bossbase_dir() / f"{image_id}.pgm"), dtype=np.uint8)
    if x.shape != (512, 512):
        raise ValueError(f"BOSSbase {image_id}: unexpected shape {x.shape}")
    return x


def pilot_ids(path: Path | None = None) -> list[int]:
    path = path or PROJECT_ROOT / "data" / "external" / "_pilot" / "pilot_bossbase_ids.txt"
    return [int(t) for t in Path(path).read_text().split()]


# ---------------------------------------------------------------------------- BOWS2, screening set
def bows2_dir() -> Path:
    env = os.environ.get("PQPKS_BOWS2_DIR")
    p = Path(env) if env else PROJECT_ROOT / "data" / "external" / "bows2"
    if not (p / "1.pgm").is_file():                  # the folder may hold only MANIFEST.sha256
        raise MissingResource(f"BOWS2 not found: {p} (set PQPKS_BOWS2_DIR, see docs/data.md)")
    return p


def load_bows2(image_id: int) -> np.ndarray:
    x = np.asarray(Image.open(bows2_dir() / f"{image_id}.pgm"), dtype=np.uint8)
    if x.shape != (512, 512):
        raise ValueError(f"BOWS2 {image_id}: unexpected shape {x.shape}")
    return x


def load_cover(source: str, image_id: int) -> np.ndarray:
    if source == "boss":
        return load_bossbase(image_id)
    if source == "bows2":
        return load_bows2(image_id)
    raise KeyError(source)


def screen_bows2_ids(path: Path | None = None) -> list[int]:
    """1000 BOWS2 images for the screening round (seed 20261105). Every BOWS2 image belongs to the training part of the
    registered split, so the screening round never touches the test set."""
    path = path or PROJECT_ROOT / "data" / "external" / "_pilot" / "screen_bows2_ids.txt"
    return [int(t) for t in Path(path).read_text().split()]


def screening_covers() -> list[tuple[str, int]]:
    """Covers of the screening (decision) round: the 1000 BOSSbase pilot images and 1000 BOWS2 images.
    None of them is in the BOSSbase test set (5000 images drawn from the 9000 non-pilot images)."""
    return [("boss", i) for i in pilot_ids()] + [("bows2", i) for i in screen_bows2_ids()]
