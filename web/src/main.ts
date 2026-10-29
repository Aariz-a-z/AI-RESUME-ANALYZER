import './styles.css';

import { analyze, type Analysis, type Requirement, type SkillGap } from './analyzer.ts';
import { createEmbedder, type LoadProgress } from './embed.ts';
import { extractPdfText } from './pdf.ts';
import { gaugeSvg, radarSvg, scoreColor } from './charts.ts';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const els = {
  jd: $<HTMLTextAreaElement>('jd'),
  file: $<HTMLInputElement>('file'),
  drop: $('drop'),
  filename: $('filename'),
  analyze: $<HTMLButtonElement>('analyze'),
  status: $('status'),
  progress: $('progress'),
  bar: $('bar'),
  empty: $('empty'),
  results: $('results'),
  gauge: $('gauge'),
  mSkills: $('m-skills'),
  mReqs: $('m-reqs'),
  mGaps: $('m-gaps'),
  headline: $('headline'),
  pReqs: $('p-reqs'),
  pSkills: $('p-skills'),
  pAts: $('p-ats'),
  download: $<HTMLButtonElement>('download'),
  extracted: $('extracted'),
};

const STATUS_ICON = { covered: '✅', partial: '🟡', missing: '❌' } as const;
const STATUS_LABEL = { covered: 'Covered', partial: 'Partial', missing: 'Not found' } as const;

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

let selectedFile: File | null = null;
let resumeText = '';
let current: Analysis | null = null;

// --- file selection ---------------------------------------------------------

function setFile(file: File | null): void {
  if (file && file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
    setStatus('That file is not a PDF.', true);
    return;
  }
  selectedFile = file;
  els.filename.textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : '';
  if (file) setStatus('');
}

els.drop.addEventListener('click', () => els.file.click());
els.drop.addEventListener('keydown', (e) => {
  const key = (e as KeyboardEvent).key;
  if (key === 'Enter' || key === ' ') {
    e.preventDefault();
    els.file.click();
  }
});
els.file.addEventListener('change', () => setFile(els.file.files?.[0] ?? null));

for (const type of ['dragenter', 'dragover'] as const) {
  els.drop.addEventListener(type, (e) => {
    e.preventDefault();
    els.drop.classList.add('is-over');
  });
}
for (const type of ['dragleave', 'drop'] as const) {
  els.drop.addEventListener(type, (e) => {
    e.preventDefault();
    els.drop.classList.remove('is-over');
  });
}
els.drop.addEventListener('drop', (e) => {
  const dropped = (e as DragEvent).dataTransfer?.files?.[0];
  if (dropped) setFile(dropped);
});

function setStatus(message: string, isError = false): void {
  els.status.textContent = message;
  els.status.classList.toggle('error', isError);
}

function setProgress(fraction: number | null): void {
  if (fraction === null) {
    els.progress.hidden = true;
    return;
  }
  els.progress.hidden = false;
  els.bar.style.width = `${Math.round(fraction * 100)}%`;
}

// --- analysis ---------------------------------------------------------------

let embedderPromise: ReturnType<typeof createEmbedder> | null = null;

function getEmbedder() {
  embedderPromise ??= createEmbedder((p: LoadProgress) => {
    if (p.status === 'progress' && typeof p.progress === 'number') {
      setStatus('Downloading the language model (one time, then cached)…');
      setProgress(p.progress / 100);
    } else if (p.status === 'ready' || p.status === 'done') {
      setProgress(null);
    }
  });
  return embedderPromise;
}

els.analyze.addEventListener('click', async () => {
  const jd = els.jd.value.trim();
  if (!selectedFile) return setStatus('Choose a PDF resume first.', true);
  if (!jd) return setStatus('Paste a job description first.', true);

  els.analyze.disabled = true;
  try {
    setStatus('Reading your resume…');
    resumeText = await extractPdfText(selectedFile);

    if (resumeText.split(/\s+/).filter(Boolean).length < 20) {
      setStatus(
        'Almost no text came out of that PDF — it is probably a scan or an image. An ATS cannot read it either.',
        true
      );
      return;
    }

    setStatus('Loading the language model…');
    const embed = await getEmbedder();

    setStatus('Scoring each requirement…');
    current = await analyze(resumeText, jd, embed);

    render(current);
    setStatus('');
    setProgress(null);
    els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    console.error(error);
    setStatus(`Something went wrong: ${(error as Error).message}`, true);
    setProgress(null);
  } finally {
    els.analyze.disabled = false;
  }
});

// --- rendering --------------------------------------------------------------

function renderRequirements(result: Analysis): string {
  if (!result.requirements.length) {
    return '<p class="empty-note">No distinct requirements were found — paste the full posting, bullets included.</p>';
  }

  const counts = { covered: 0, partial: 0, missing: 0 };
  for (const r of result.requirements) counts[r.status]++;

  const rows = result.requirements
    .map((req: Requirement) => {
      const tag = req.priority === 'required' ? 'must have' : 'nice to have';
      const present = req.skillsPresent.length
        ? `<div><strong>Found in your resume:</strong> ${escapeHtml(req.skillsPresent.join(', '))}</div>`
        : '';
      const missing = req.skillsMissing.length
        ? `<div><strong>Not found:</strong> ${escapeHtml(req.skillsMissing.join(', '))}</div>`
        : '';
      const body =
        req.status === 'missing'
          ? '<p class="req-meta">Nothing in your resume addresses this. If you have the experience, add a bullet that names it explicitly.</p>'
          : `<p class="evidence">${escapeHtml(req.evidence)}</p>`;

      return `
<details class="req"${req.status === 'missing' ? ' open' : ''}>
  <summary><span class="req-icon">${STATUS_ICON[req.status]}</span><span>${escapeHtml(req.text)}</span></summary>
  <div class="req-body">
    <div class="req-meta">${STATUS_LABEL[req.status]} · <span class="pill ${req.priority}">${tag}</span> · similarity ${req.score.toFixed(2)}</div>
    ${present}${missing}${body}
  </div>
</details>`;
    })
    .join('');

  return `<p class="summary-line">${counts.covered} covered · ${counts.partial} partial · ${counts.missing} not found, across ${result.requirements.length} requirements</p>${rows}`;
}

function renderSkills(result: Analysis): string {
  const gap = (s: SkillGap) => `
<div class="gap">
  <span class="gap-name">${escapeHtml(s.name)}</span> — ${escapeHtml(s.category)}${
    s.learnUrl ? ` · <a href="${s.learnUrl}" target="_blank" rel="noopener noreferrer">Learn</a>` : ''
  }
  ${s.related.length ? `<span class="gap-related">Related experience you already have: ${escapeHtml(s.related.join(', '))}</span>` : ''}
</div>`;

  const required = result.missingRequired.length
    ? result.missingRequired.map(gap).join('')
    : '<p class="empty-note">No must-have gaps — every required skill in this posting appears in your resume.</p>';

  const preferred = result.missingPreferred.length
    ? `<section><h3>Nice-to-have gaps</h3><div class="chips">${result.missingPreferred
        .map((s) => `<span class="chip">${escapeHtml(s.name)}</span>`)
        .join('')}</div></section>`
    : '';

  const matched = result.matched.length
    ? `<section><h3>Matched skills</h3><div class="chips">${result.matched
        .map((s) => `<span class="chip ok">${escapeHtml(s.name)}</span>`)
        .join('')}</div></section>`
    : '';

  const radar = radarSvg(result.categoryCoverage);

  return `
<div class="two-col">
  <div>
    <section><h3>Must-have gaps</h3>${required}</section>
    ${preferred}
    ${matched}
  </div>
  <div>
    <h3>Coverage by category</h3>
    ${radar ? `<div class="radar">${radar}</div>` : '<p class="empty-note">Not enough distinct skill categories in this posting to chart.</p>'}
  </div>
</div>`;
}

function renderAts(result: Analysis): string {
  const checks = result.ats
    .map(
      (c) =>
        `<div class="check ${c.ok ? 'ok' : 'warn'}"><span class="mark">${c.ok ? '✓' : '!'}</span><span>${escapeHtml(c.message)}</span></div>`
    )
    .join('');

  let experience: string;
  if (result.yearsRequired === null) {
    experience = '<p class="empty-note">The posting does not state a years-of-experience requirement.</p>';
  } else if (result.yearsEstimated === null) {
    experience = `<p class="empty-note">The posting asks for about ${result.yearsRequired} years. No date ranges were detected in your resume, so this could not be checked.</p>`;
  } else {
    const ok = result.yearsEstimated + 0.5 >= result.yearsRequired;
    experience = `<div class="check ${ok ? 'ok' : 'warn'}"><span class="mark">${ok ? '✓' : '!'}</span><span>Posting asks for ~${result.yearsRequired} years; your dates suggest about ${result.yearsEstimated.toFixed(0)}.</span></div>
      <p class="empty-note" style="margin-top:8px">Estimated from date ranges, with overlapping roles merged.</p>`;
  }

  return `<section class="ats"><h3>ATS readability</h3>${checks}</section>
    <section class="ats" style="margin-top:20px"><h3>Experience</h3>${experience}</section>`;
}

function render(result: Analysis): void {
  els.empty.hidden = true;
  els.results.hidden = false;

  els.gauge.innerHTML = `${gaugeSvg(result.overall)}<div style="font-weight:600;color:${scoreColor(result.overall)}">${result.verdict}</div>`;
  els.mSkills.textContent = `${result.skillPercent}%`;
  els.mReqs.textContent = `${result.requirementPercent}%`;
  els.mGaps.textContent = String(result.missingRequired.length);

  const gaps = result.missingRequired;
  els.headline.className = `headline ${gaps.length ? 'bad' : 'good'}`;
  els.headline.innerHTML = gaps.length
    ? `<strong>Biggest gaps:</strong> ${escapeHtml(gaps.slice(0, 6).map((s) => s.name).join(', '))}${gaps.length > 6 ? ` (+${gaps.length - 6} more)` : ''}`
    : 'Every required skill named in this posting appears in your resume.';

  els.pReqs.innerHTML = renderRequirements(result);
  els.pSkills.innerHTML = renderSkills(result);
  els.pAts.innerHTML = renderAts(result);
  els.extracted.textContent = resumeText.slice(0, 8000);
}

// --- tabs -------------------------------------------------------------------

for (const tab of document.querySelectorAll<HTMLButtonElement>('.tab')) {
  tab.addEventListener('click', () => {
    for (const other of document.querySelectorAll<HTMLButtonElement>('.tab')) {
      const panel = document.getElementById(other.dataset.panel!)!;
      const active = other === tab;
      other.classList.toggle('is-active', active);
      panel.hidden = !active;
    }
  });
}

// --- report -----------------------------------------------------------------

function buildReport(result: Analysis): string {
  const lines = [
    'RESUME FIT ANALYSIS',
    '='.repeat(60),
    `Overall fit:            ${result.overall}%  (${result.verdict})`,
    `  Skill coverage:       ${result.skillPercent}%`,
    `  Requirement coverage: ${result.requirementPercent}%`,
    '',
    'MISSING REQUIRED SKILLS',
    '-'.repeat(60),
  ];

  if (result.missingRequired.length) {
    for (const s of result.missingRequired) {
      lines.push(`  - ${s.name} (${s.category})`);
      if (s.related.length) lines.push(`      related experience you do have: ${s.related.join(', ')}`);
      if (s.learnUrl) lines.push(`      learn: ${s.learnUrl}`);
    }
  } else {
    lines.push('  None - every required skill in the posting appears in your resume.');
  }

  lines.push('', 'MISSING NICE-TO-HAVE SKILLS', '-'.repeat(60));
  lines.push('  ' + (result.missingPreferred.map((s) => s.name).join(', ') || 'None.'));

  lines.push('', 'REQUIREMENT BY REQUIREMENT', '-'.repeat(60));
  for (const req of result.requirements) {
    lines.push(`  [${STATUS_LABEL[req.status].padEnd(9)}] (${req.priority}) ${req.text}`);
    if (req.skillsMissing.length) lines.push(`      not found in resume: ${req.skillsMissing.join(', ')}`);
    if (req.status !== 'missing' && req.evidence) lines.push(`      matched on: ${req.evidence.slice(0, 150)}`);
  }

  lines.push('', 'ATS READABILITY', '-'.repeat(60));
  for (const c of result.ats) lines.push(`  [${c.ok ? 'OK  ' : 'WARN'}] ${c.message}`);

  if (result.yearsRequired !== null) {
    const estimate = result.yearsEstimated === null ? 'could not estimate' : result.yearsEstimated.toFixed(0);
    lines.push('', 'EXPERIENCE', '-'.repeat(60),
      `  Posting asks for ~${result.yearsRequired} years; resume suggests ${estimate}.`);
  }

  lines.push('', 'Scores blend literal skill matching with semantic similarity.',
    'Treat them as guidance, not a hiring decision.');
  return lines.join('\n');
}

els.download.addEventListener('click', () => {
  if (!current) return;
  const blob = new Blob([buildReport(current)], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'resume_fit_report.txt';
  a.click();
  URL.revokeObjectURL(url);
});
