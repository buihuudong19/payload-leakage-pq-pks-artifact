"""BL1 provenance: the published SHA-256 list has the nine reference modules, and the checker reports a missing folder."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts" / "setup" / "check_bl1_reference.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_sums_and_checker(tmp_path):
    names = [ln.split()[1] for ln in m.SUMS.read_text().splitlines() if ln.strip()]
    assert sorted(names) == sorted(["BinToText.py", "ECCurve.py", "ECdecrypt.py", "ECencrypt.py", "Embed.py", "Extract.py",
                                    "PicToData.py", "TextToBin.py", "mod_sqrt.py"])
    assert len(m.check(tmp_path)) == 9
