"""Random-oracle bound on epsilon_valid (scripts/theory/epsilon_valid_rom.py): Compress_10 cells and the CBD_2 mass."""
import importlib.util
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "theory" / "epsilon_valid_rom.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_cells_and_bound():
    c = m.cells()
    assert len(c) == 1024 and sum(len(v) == 4 for v in c.values()) == 257 and sum(len(v) == 3 for v in c.values()) == 767
    assert sum(m.cbd(2).values()) == 1
    p, size = m.max_cell_mass()
    assert p == Fraction(15, 16) and size == 4
