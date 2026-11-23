"""Print the step table of run_all (read directly from scripts/orchestration/pipeline.ps1) as Markdown, so that the table in
the documentation cannot drift from what run_all actually runs.   python scripts/orchestration/plan_table.py > steps.md"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = {"py": "& $Py", "pyg": "& $PyG", "ps": "powershell -ExecutionPolicy Bypass -File"}


def parse(text: str) -> list[dict]:
    steps = []
    for m in re.finditer(r"@\{ Id = '([^']+)'(.*?)(?=\n    @\{ Id = |\n\)\s*$)", text, flags=re.S):
        body = m.group(2)
        get = lambda k: (re.search(rf"{k} = '([^']*)'", body) or [None, ""])[1]
        lst = lambda k: re.findall(r"'([^']*)'", (re.search(rf"{k} = @\(([^)]*)\)", body) or [None, ""])[1])
        steps.append({"id": m.group(1), "lane": get("Lane"), "name": get("Name"), "kind": get("Kind"),
                      "args": lst("Args"), "deps": lst("Deps"), "est": get("Est"), "out": get("Out"),
                      "soft": "Soft = $true" in body, "when": get("WhenText"), "check": "Check = {" in body})
    return steps


def table(steps: list[dict]) -> str:
    rows = ["| # | Step | Lane | Command | Task | Waits for | Estimate | Output |", "|---|---|---|---|---|---|---|---|"]
    prev = {}
    for k, s in enumerate(steps, 1):
        p = prev.get(s["lane"])
        deps = ([("~" if s["soft"] else "") + p] if p else []) + s["deps"]
        cmd = f"`{EXE[s['kind']]} {' '.join(s['args'])}`"
        note = s["name"] + (f" (skipped when: {s['when']})" if s["when"] else "") + (" · checked after running" if s["check"] else "")
        rows.append(f"| {k} | {s['id']} | {s['lane']} | {cmd} | {note} | {', '.join(deps) or '–'} | {s['est']} | `{s['out']}` |")
        prev[s["lane"]] = s["id"]
    return "\n".join(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(table(parse((ROOT / "scripts" / "orchestration" / "pipeline.ps1").read_text(encoding="utf-8-sig"))))
