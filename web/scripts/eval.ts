/**
 * Parity check: the TypeScript port against the Python analyzer.
 *
 * Run from web/:  npm run eval
 *
 * Everything lexical - skill detection, requirement parsing, chunking, ATS,
 * years - must match Python exactly. The semantic half is allowed a small
 * tolerance because the browser build runs int8-quantised weights while Python
 * runs fp32.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { analyze } from '../src/analyzer.ts';
import { createEmbedder } from '../src/embed.ts';

const HERE = dirname(fileURLToPath(import.meta.url));

interface BaselineCase {
  resume: string;
  jd: string;
  overall: number;
  skillPercent: number;
  requirementPercent: number;
  verdict: string;
  requirementCount: number;
  resumeChunks: number;
  missingRequired: string[];
  matched: string[];
  statuses: string[];
  yearsRequired: number | null;
  yearsEstimated: number | null;
}

interface Baseline {
  documents: Record<string, string>;
  cases: BaselineCase[];
}

const SEMANTIC_TOLERANCE = 4; // percentage points, int8 vs fp32

const baseline = JSON.parse(readFileSync(join(HERE, 'baseline.json'), 'utf8')) as Baseline;

let failures = 0;
function check(label: string, ok: boolean, detail: string): void {
  if (!ok) failures++;
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}: ${detail}`);
}

const same = (a: string[], b: string[]) => a.length === b.length && a.every((v, i) => v === b[i]);

console.log('loading all-MiniLM-L6-v2 (int8) ...');
const embed = await createEmbedder();

console.log(`\n${'resume'.padEnd(17)}${'job'.padEnd(14)}${'TS'.padStart(5)}${'PY'.padStart(5)}  delta`);
console.log('-'.repeat(52));

const results: Array<{ expected: BaselineCase; overall: number }> = [];

for (const expected of baseline.cases) {
  const resume = baseline.documents[expected.resume]!;
  const jd = baseline.documents[expected.jd]!;
  const got = await analyze(resume, jd, embed);
  results.push({ expected, overall: got.overall });

  const delta = got.overall - expected.overall;
  console.log(
    `${expected.resume.padEnd(17)}${expected.jd.padEnd(14)}` +
      `${String(got.overall).padStart(4)}%${String(expected.overall).padStart(4)}%` +
      `  ${delta > 0 ? '+' : ''}${delta}`
  );

  const tag = `${expected.resume} x ${expected.jd}`;

  // Lexical half: must be identical.
  check(`${tag} skill %`, got.skillPercent === expected.skillPercent,
    `${got.skillPercent}% vs ${expected.skillPercent}%`);
  check(`${tag} requirements parsed`, got.requirements.length === expected.requirementCount,
    `${got.requirements.length} vs ${expected.requirementCount}`);
  check(`${tag} resume chunks`, got.resumeChunks === expected.resumeChunks,
    `${got.resumeChunks} vs ${expected.resumeChunks}`);
  check(`${tag} missing required`, same(got.missingRequired.map((s) => s.name).sort(), expected.missingRequired),
    `[${got.missingRequired.map((s) => s.name).sort().join(', ')}]`);
  check(`${tag} matched skills`, same(got.matched.map((s) => s.name).sort(), expected.matched),
    `${got.matched.length} skills`);
  check(`${tag} years required`, got.yearsRequired === expected.yearsRequired,
    `${got.yearsRequired} vs ${expected.yearsRequired}`);
  check(`${tag} years estimated`, got.yearsEstimated === expected.yearsEstimated,
    `${got.yearsEstimated} vs ${expected.yearsEstimated}`);

  // Semantic half: close enough.
  check(`${tag} overall within ${SEMANTIC_TOLERANCE}pts`, Math.abs(delta) <= SEMANTIC_TOLERANCE,
    `${got.overall}% vs ${expected.overall}% (delta ${delta})`);
  check(`${tag} verdict`, got.verdict === expected.verdict, `${got.verdict} vs ${expected.verdict}`);

  const statusDrift = got.requirements.filter((r, i) => r.status !== expected.statuses[i]).length;
  check(`${tag} requirement statuses`, statusDrift === 0, `${statusDrift} differ`);
  console.log('');
}

// The properties that actually matter, independent of Python.
console.log('Behavioural guarantees');
const byPair = new Map(results.map((r) => [`${r.expected.resume}|${r.expected.jd}`, r.overall]));
const strong = byPair.get('ml_resume|ml_jd')!;
const partial = byPair.get('analyst_resume|ml_jd')!;
const unrelated = byPair.get('frontend_resume|ml_jd')!;

check('strong > partial', strong > partial, `${strong}% vs ${partial}%`);
check('partial > unrelated', partial > unrelated, `${partial}% vs ${unrelated}%`);
check('separation >= 40 pts', strong - unrelated >= 40, `${strong - unrelated} points`);

console.log('\n' + '-'.repeat(52));
if (failures) {
  console.log(`FAILED: ${failures} check(s)`);
  process.exit(1);
}
console.log('TypeScript port matches the Python baseline.');
