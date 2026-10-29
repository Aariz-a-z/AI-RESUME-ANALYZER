/**
 * Resume / job-description fit analysis.
 *
 * A direct port of analyzer.py. The thresholds, regexes and scoring formula are
 * intentionally identical so the two produce the same numbers; scripts/eval.ts
 * checks that against the Python baseline.
 *
 * Why not one whole-document similarity: MiniLM truncates at 256 word pieces and
 * a real two-page resume runs 800-1500, so a document-level embedding never sees
 * most of it. Requirements are matched against resume chunks instead.
 */

import { SKILL_DB } from './skills.ts';

export const COVERED_T = 0.5;
export const PARTIAL_T = 0.34;
export const SIM_FLOOR = 0.1;
export const SIM_CEIL = 0.55;
export const W_SKILLS = 0.55;
export const W_REQS = 0.45;
export const PREFERRED_WEIGHT = 0.4;
export const MAX_REQUIREMENTS = 24;
export const CHUNK_WORDS = 45;
export const CHUNK_OVERLAP = 12;

const PUNCT = /[^a-z0-9+#./\- ]+/g;
const WS = /\s+/g;

/** Lowercase and strip punctuation, preserving the characters that carry meaning
 *  in technology names: c++, c#, .net, ci/cd, node.js. */
export function normalize(text: string): string {
  const t = text
    .toLowerCase()
    .replace(/–|—/g, '-')
    .replace(/’/g, "'")
    .replace(PUNCT, ' ');
  return t.replace(WS, ' ').trim();
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function aliasPattern(aliases: string[]): RegExp {
  // Longest first so "machine learning" wins over a bare "machine".
  const parts = aliases.map(escapeRe).sort((a, b) => b.length - a.length);
  return new RegExp(`(?<![a-z0-9+#])(?:${parts.join('|')})(?![a-z0-9+#])`);
}

interface CompiledSkill {
  name: string;
  category: string;
  pattern: RegExp;
  learnUrl: string | null;
}

const SKILLS: CompiledSkill[] = SKILL_DB.map((s) => ({
  name: s.name,
  category: s.category,
  pattern: aliasPattern(s.aliases),
  learnUrl: s.learnUrl,
}));

export interface SkillRef {
  category: string;
  learnUrl: string | null;
}

export function findSkills(normalized: string): Map<string, SkillRef> {
  const found = new Map<string, SkillRef>();
  for (const s of SKILLS) {
    if (s.pattern.test(normalized)) {
      found.set(s.name, { category: s.category, learnUrl: s.learnUrl });
    }
  }
  return found;
}

const words = (s: string) => s.split(/\s+/).filter(Boolean);

// --- Job description parsing ------------------------------------------------

const BULLET = /^\s*(?:[-*•▪◦‣·o]|\d+[.)])\s+/;
const SENTENCE = /(?<=[.;!?])\s+/;

const REQ_CUE =
  /\b(experience[d]?|proficien\w*|knowledge|familiar\w*|skill\w*|abilit\w*|able to|degree|years?|bachelor\w*|master\w*|phd|understand\w*|expertise|background|competen\w*|you will|you'll|responsible|work with|build|develop|design|deploy|maintain|collaborat\w*|strong|solid|hands.on|demonstrated|track record|required|preferred|must)\b/;
const REQUIRED_CUE =
  /\b(must have|must|required|require[sd]?|essential|mandatory|minimum|at least|strong|solid|proven|demonstrated)\b/;
const PREFERRED_CUE =
  /\b(preferred|nice to have|nice-to-have|plus|bonus|desirable|ideally|advantage|familiarity|would be great|good to have)\b/;
const SECTION_PREFERRED =
  /^\s*(preferred|nice to have|bonus|desirable|good to have|pluses)\b[^.]{0,40}:?\s*$/;
const SECTION_REQUIRED =
  /^\s*(requirements?|qualifications?|must have|what you.{0,4}ll need|responsibilit\w+|who you are|about you)\b[^.]{0,40}:?\s*$/;

export type Priority = 'required' | 'preferred';
export type Status = 'covered' | 'partial' | 'missing';

export class Requirement {
  text: string;
  priority: Priority;
  score = 0;
  evidence = '';
  status: Status = 'missing';
  skillsPresent: string[] = [];
  skillsMissing: string[] = [];

  constructor(text: string, priority: Priority) {
    this.text = text;
    this.priority = priority;
  }

  get semanticCredit(): number {
    const raw = (this.score - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR);
    return Math.min(1, Math.max(0, raw));
  }

  /** Share of the technologies this requirement names that the resume contains,
   *  or null when it names none. */
  get lexicalCredit(): number | null {
    const total = this.skillsPresent.length + this.skillsMissing.length;
    return total ? this.skillsPresent.length / total : null;
  }

  /**
   * A skills list embeds poorly against prose: "Hands-on experience deploying
   * services with Docker on AWS" scores only ~0.27 against a resume whose skills
   * line plainly reads "Docker, Kubernetes, AWS". Literal evidence of the named
   * technology is the stronger signal, so whichever source is more convincing wins.
   */
  get credit(): number {
    const lexical = this.lexicalCredit;
    return lexical === null ? this.semanticCredit : Math.max(this.semanticCredit, lexical);
  }
}

export interface SplitResult {
  requirements: Requirement[];
  requiredBlob: string;
  preferredBlob: string;
}

export function splitRequirements(jdText: string): SplitResult {
  const requirements: Requirement[] = [];
  const requiredLines: string[] = [];
  const preferredLines: string[] = [];
  const seen = new Set<string>();
  let section: Priority = 'required';

  for (const rawLine of jdText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;

    const low = line.toLowerCase();
    if (SECTION_PREFERRED.test(low)) {
      section = 'preferred';
      continue;
    }
    if (SECTION_REQUIRED.test(low)) {
      section = 'required';
      continue;
    }

    const isBullet = BULLET.test(line);
    const body = line.replace(BULLET, '').trim();

    // A paragraph can hold several requirements; a bullet is already one.
    const pieces = isBullet || words(body).length <= 35 ? [body] : body.split(SENTENCE);

    for (const raw of pieces) {
      const piece = raw.replace(/^[\s\t;:\-•]+|[\s\t;:\-•]+$/g, '');
      const n = words(piece).length;
      if (n < 4 || n > 60) continue;

      const lowPiece = piece.toLowerCase();
      if (!(isBullet || REQ_CUE.test(lowPiece))) continue;

      let priority: Priority;
      if (PREFERRED_CUE.test(lowPiece)) priority = 'preferred';
      else if (REQUIRED_CUE.test(lowPiece)) priority = 'required';
      else priority = section;

      (priority === 'preferred' ? preferredLines : requiredLines).push(piece);

      const key = normalize(piece);
      if (seen.has(key)) continue;
      seen.add(key);
      requirements.push(new Requirement(piece, priority));
    }
  }

  // Fall back to sentence splitting when the JD has no line structure at all.
  if (requirements.length === 0) {
    for (const raw of jdText.replace(/\r?\n/g, ' ').split(SENTENCE)) {
      const piece = raw.trim();
      const n = words(piece).length;
      if (n >= 4 && n <= 60) {
        requirements.push(new Requirement(piece, 'required'));
        requiredLines.push(piece);
      }
    }
  }

  return {
    requirements: requirements.slice(0, MAX_REQUIREMENTS),
    requiredBlob: normalize(requiredLines.join(' ')),
    preferredBlob: normalize(preferredLines.join(' ')),
  };
}

// --- Resume parsing ---------------------------------------------------------

/** Split a resume into overlapping chunks that stay well under the model's 256
 *  word-piece limit, so nothing past the first page is silently dropped. */
export function chunkResume(resumeText: string, maxWords = CHUNK_WORDS): string[] {
  const units: string[] = [];
  for (const rawLine of resumeText.split(/\r?\n/)) {
    const line = rawLine.trim().replace(BULLET, '').trim();
    const n = words(line).length;
    if (n < 3) continue;
    if (n > maxWords) {
      for (const s of line.split(SENTENCE)) {
        const t = s.trim();
        if (t) units.push(t);
      }
    } else {
      units.push(line);
    }
  }

  const chunks: string[] = [];
  let buffer: string[] = [];
  for (const unit of units) {
    buffer.push(...words(unit));
    if (buffer.length >= maxWords) {
      chunks.push(buffer.slice(0, maxWords).join(' '));
      buffer = buffer.slice(maxWords - CHUNK_OVERLAP);
    }
  }
  if (buffer.length >= 3) chunks.push(buffer.join(' '));

  return chunks.length ? chunks : [resumeText.slice(0, 500)];
}

// --- Experience -------------------------------------------------------------

const JD_YEARS = /(\d{1,2})\s*\+?\s*(?:-|to|–)?\s*(?:\d{1,2})?\s*\+?\s*years?/g;
const DATE_RANGE =
  /((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)/gi;

export function requiredYears(jdText: string): number | null {
  const values: number[] = [];
  for (const m of jdText.toLowerCase().matchAll(JD_YEARS)) {
    const v = parseInt(m[1]!, 10);
    if (v > 0 && v <= 20) values.push(v);
  }
  return values.length ? Math.max(...values) : null;
}

/** Rough span of employment implied by date ranges. An estimate only:
 *  overlapping roles are merged, undated work is invisible. */
export function estimatedYears(resumeText: string): number | null {
  const thisYear = new Date().getFullYear();
  const spans: Array<[number, number]> = [];
  for (const m of resumeText.matchAll(DATE_RANGE)) {
    const s = parseInt(m[1]!, 10);
    const endRaw = m[2]!.toLowerCase();
    const e = ['present', 'current', 'now'].includes(endRaw) ? thisYear : parseInt(endRaw, 10);
    if (s > 1960 && s <= e && e <= thisYear) spans.push([s, e]);
  }
  if (!spans.length) return null;

  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const merged: Array<[number, number]> = [[spans[0]![0], spans[0]![1]]];
  for (const [s, e] of spans.slice(1)) {
    const last = merged[merged.length - 1]!;
    if (s <= last[1]) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  return merged.reduce((acc, [s, e]) => acc + (e - s), 0);
}

// --- ATS checks -------------------------------------------------------------

const EMAIL = /[\w.+-]+@[\w-]+\.[\w.]+/;
const PHONE = /\+?\d[\d\s().-]{7,}\d/;
const SECTION_WORDS: Array<[string, RegExp]> = [
  ['experience', /\b(experience|employment|work history)\b/],
  ['education', /\b(education|academic|university|college|b\.?e\.?|b\.?tech|degree)\b/],
  ['skills', /\b(skills?|technologies|technical)\b/],
];

export interface AtsCheck {
  ok: boolean;
  message: string;
}

export function atsReport(resumeText: string): AtsCheck[] {
  const checks: AtsCheck[] = [];
  const n = words(resumeText).length;
  const low = resumeText.toLowerCase();

  if (n < 50) {
    checks.push({
      ok: false,
      message: `Only ${n} words extracted - the PDF is likely image-based or heavily formatted, which ATS parsers cannot read.`,
    });
  } else if (n > 1200) {
    checks.push({
      ok: false,
      message: `${n} words is long for a resume; most screeners expect under about 800.`,
    });
  } else {
    checks.push({ ok: true, message: `${n} words extracted cleanly.` });
  }

  const hasEmail = EMAIL.test(resumeText);
  checks.push({
    ok: hasEmail,
    message: hasEmail ? 'Email address found.' : 'No email address detected - parsers often key on it.',
  });

  const hasPhone = PHONE.test(resumeText);
  checks.push({
    ok: hasPhone,
    message: hasPhone ? 'Phone number found.' : 'No phone number detected.',
  });

  const missing = SECTION_WORDS.filter(([, re]) => !re.test(low)).map(([name]) => name);
  checks.push(
    missing.length
      ? { ok: false, message: `Missing standard section heading(s): ${missing.sort().join(', ')}.` }
      : { ok: true, message: 'Standard sections present (experience, education, skills).' }
  );

  return checks;
}

// --- Result types -----------------------------------------------------------

export interface SkillGap {
  name: string;
  category: string;
  priority: Priority;
  inResume: boolean;
  learnUrl: string | null;
  related: string[];
}

export const gapWeight = (g: SkillGap) => (g.priority === 'required' ? 1 : PREFERRED_WEIGHT);

export interface Analysis {
  overall: number;
  verdict: string;
  skillPercent: number;
  requirementPercent: number;
  skills: SkillGap[];
  requirements: Requirement[];
  categoryCoverage: Map<string, [number, number]>;
  ats: AtsCheck[];
  yearsRequired: number | null;
  yearsEstimated: number | null;
  resumeChunks: number;
  missingRequired: SkillGap[];
  missingPreferred: SkillGap[];
  matched: SkillGap[];
}

function verdictFor(score: number): string {
  if (score >= 75) return 'Strong match';
  if (score >= 58) return 'Good match';
  if (score >= 40) return 'Partial match';
  return 'Weak match';
}

export type Embedder = (texts: string[]) => Promise<number[][]>;

const dot = (a: number[], b: number[]) => {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += a[i]! * b[i]!;
  return total;
};

export async function analyze(
  resumeText: string,
  jdText: string,
  embed: Embedder
): Promise<Analysis> {
  const normResume = normalize(resumeText);
  const { requirements, requiredBlob, preferredBlob } = splitRequirements(jdText);
  const chunks = chunkResume(resumeText);

  const jdSkills = findSkills(normalize(jdText));
  const resumeSkills = findSkills(normResume);

  let requirementScore = 0;
  if (requirements.length) {
    const reqVecs = await embed(requirements.map((r) => r.text));
    const chunkVecs = await embed(chunks);

    requirements.forEach((requirement, i) => {
      let best = 0;
      let bestScore = -Infinity;
      for (let j = 0; j < chunkVecs.length; j++) {
        const s = dot(reqVecs[i]!, chunkVecs[j]!);
        if (s > bestScore) {
          bestScore = s;
          best = j;
        }
      }
      requirement.score = bestScore;
      requirement.evidence = chunks[best]!;

      const named = findSkills(normalize(requirement.text));
      requirement.skillsPresent = [...named.keys()].filter((n) => resumeSkills.has(n)).sort();
      requirement.skillsMissing = [...named.keys()].filter((n) => !resumeSkills.has(n)).sort();

      const lexical = requirement.lexicalCredit;
      if ((lexical !== null && lexical >= 0.75) || requirement.score >= COVERED_T) {
        requirement.status = 'covered';
      } else if ((lexical !== null && lexical > 0) || requirement.score >= PARTIAL_T) {
        requirement.status = 'partial';
      } else {
        requirement.status = 'missing';
      }
    });

    let weighted = 0;
    let totalWeight = 0;
    for (const r of requirements) {
      const w = r.priority === 'required' ? 1 : PREFERRED_WEIGHT;
      weighted += w * r.credit;
      totalWeight += w;
    }
    requirementScore = weighted / totalWeight;
  }

  const skills: SkillGap[] = [];
  for (const name of [...jdSkills.keys()].sort()) {
    const ref = jdSkills.get(name)!;
    const compiled = SKILLS.find((s) => s.name === name)!;
    // A skill only mentioned in the "nice to have" band is weighted down.
    const priority: Priority =
      !compiled.pattern.test(requiredBlob) && compiled.pattern.test(preferredBlob)
        ? 'preferred'
        : 'required';
    const inResume = resumeSkills.has(name);
    const related = inResume
      ? []
      : [...resumeSkills.entries()]
          .filter(([, v]) => v.category === ref.category)
          .map(([n]) => n)
          .sort()
          .slice(0, 3);
    skills.push({ name, category: ref.category, priority, inResume, learnUrl: ref.learnUrl, related });
  }

  let skillScore = 0;
  let overall: number;
  if (skills.length) {
    const totalW = skills.reduce((acc, s) => acc + gapWeight(s), 0);
    skillScore = skills.filter((s) => s.inResume).reduce((acc, s) => acc + gapWeight(s), 0) / totalW;
    overall = W_SKILLS * skillScore + W_REQS * requirementScore;
  } else {
    overall = requirementScore;
  }

  const categoryCoverage = new Map<string, [number, number]>();
  for (const s of skills) {
    const [hit, total] = categoryCoverage.get(s.category) ?? [0, 0];
    categoryCoverage.set(s.category, [hit + (s.inResume ? 1 : 0), total + 1]);
  }

  const score = Math.round(Math.min(1, Math.max(0, overall)) * 100);
  return {
    overall: score,
    verdict: verdictFor(score),
    skillPercent: Math.round(skillScore * 100),
    requirementPercent: Math.round(requirementScore * 100),
    skills,
    requirements,
    categoryCoverage,
    ats: atsReport(resumeText),
    yearsRequired: requiredYears(jdText),
    yearsEstimated: estimatedYears(resumeText),
    resumeChunks: chunks.length,
    missingRequired: skills.filter((s) => !s.inResume && s.priority === 'required'),
    missingPreferred: skills.filter((s) => !s.inResume && s.priority === 'preferred'),
    matched: skills.filter((s) => s.inResume),
  };
}
