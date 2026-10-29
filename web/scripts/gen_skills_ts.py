"""Generate web/src/skills.ts from skills_db.py so the two cannot drift."""
import json
import sys

sys.path.insert(0, r"C:/Users/yaheya khan/OneDrive/Documents/hacker/Desktop/AI RESUME ANALYZER")
from skills_db import SKILL_DB  # noqa: E402

out = sys.argv[1]

lines = [
    "// GENERATED FILE - do not edit by hand.",
    "// Source of truth: ../../skills_db.py (regenerate with scripts/gen_skills_ts.py).",
    "",
    "export interface Skill {",
    "  name: string;",
    "  category: string;",
    "  aliases: string[];",
    "  learnUrl: string | null;",
    "}",
    "",
    "export const SKILL_DB: Skill[] = [",
]

for name, category, aliases, url in SKILL_DB:
    lines.append(
        "  { name: %s, category: %s, aliases: %s, learnUrl: %s },"
        % (
            json.dumps(name),
            json.dumps(category),
            json.dumps(aliases),
            json.dumps(url) if url else "null",
        )
    )

lines += [
    "];",
    "",
    "export const CATEGORIES: string[] = [",
]

seen = []
for _n, cat, _a, _u in SKILL_DB:
    if cat not in seen:
        seen.append(cat)
for cat in seen:
    lines.append("  %s," % json.dumps(cat))
lines += ["];", ""]

with open(out, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines))

print(f"wrote {out}: {len(SKILL_DB)} skills, {len(seen)} categories")
