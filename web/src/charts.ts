/** Hand-rolled inline SVG so the page carries no charting library. */

const polar = (cx: number, cy: number, r: number, deg: number): [number, number] => {
  const a = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
};

function arc(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const [x1, y1] = polar(cx, cy, r, startDeg);
  const [x2, y2] = polar(cx, cy, r, endDeg);
  const large = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export function scoreColor(score: number): string {
  if (score >= 75) return 'var(--good)';
  if (score >= 58) return '#65a30d';
  if (score >= 40) return 'var(--warn)';
  return 'var(--bad)';
}

/** Half-dial, 180deg (left) to 360deg (right). */
export function gaugeSvg(score: number): string {
  const cx = 120;
  const cy = 122;
  const r = 92;
  const end = 180 + (Math.min(100, Math.max(0, score)) / 100) * 180;

  return `
<svg viewBox="0 0 240 150" role="img" aria-label="Overall fit ${score} percent">
  <path d="${arc(cx, cy, r, 180, 360)}" fill="none" stroke="var(--border)" stroke-width="16" stroke-linecap="round" />
  ${score > 0 ? `<path d="${arc(cx, cy, r, 180, end)}" fill="none" stroke="${scoreColor(score)}" stroke-width="16" stroke-linecap="round" />` : ''}
  <text x="${cx}" y="${cy - 6}" text-anchor="middle" font-size="42" font-weight="700" fill="var(--text)">${score}<tspan font-size="22">%</tspan></text>
  <text x="18" y="146" font-size="11" fill="var(--muted)">0</text>
  <text x="222" y="146" font-size="11" fill="var(--muted)" text-anchor="end">100</text>
</svg>`.trim();
}

/** Break a long category name onto two lines so axis labels never clip. */
function wrap(label: string): string[] {
  if (label.length <= 14) return [label];
  const mid = Math.floor(label.length / 2);
  let best = -1;
  for (let i = 0; i < label.length; i++) {
    if (label[i] === ' ' && (best === -1 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  }
  if (best === -1) return [label];
  return [label.slice(0, best), label.slice(best + 1)];
}

export function radarSvg(coverage: Map<string, [number, number]>): string {
  const entries = [...coverage.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  if (entries.length < 3) return '';

  const cx = 220;
  const cy = 190;
  const R = 112;
  const n = entries.length;
  const angleAt = (i: number) => -90 + (360 * i) / n;

  const rings = [25, 50, 75, 100]
    .map((pct) => {
      const pts = entries
        .map((_, i) => polar(cx, cy, (R * pct) / 100, angleAt(i)).map((v) => v.toFixed(1)).join(','))
        .join(' ');
      return `<polygon points="${pts}" fill="none" stroke="var(--border)" stroke-width="1" />`;
    })
    .join('');

  const spokes = entries
    .map((_, i) => {
      const [x, y] = polar(cx, cy, R, angleAt(i));
      return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-width="1" />`;
    })
    .join('');

  const valuePts = entries
    .map(([, [hit, total]], i) => {
      const pct = total ? (100 * hit) / total : 0;
      return polar(cx, cy, (R * pct) / 100, angleAt(i)).map((v) => v.toFixed(1)).join(',');
    })
    .join(' ');

  const dots = entries
    .map(([, [hit, total]], i) => {
      const pct = total ? (100 * hit) / total : 0;
      const [x, y] = polar(cx, cy, (R * pct) / 100, angleAt(i));
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="var(--accent)" />`;
    })
    .join('');

  const labels = entries
    .map(([name, [hit, total]], i) => {
      const deg = angleAt(i);
      const [x, y] = polar(cx, cy, R + 26, deg);
      const cos = Math.cos((deg * Math.PI) / 180);
      const anchor = cos > 0.3 ? 'start' : cos < -0.3 ? 'end' : 'middle';
      const lines = wrap(name);
      const startY = y - (lines.length - 1) * 6;
      const tspans = lines
        .map((line, li) => `<tspan x="${x.toFixed(1)}" dy="${li === 0 ? 0 : 12}">${line}</tspan>`)
        .join('');
      return `<text x="${x.toFixed(1)}" y="${startY.toFixed(1)}" text-anchor="${anchor}" font-size="11" fill="var(--muted)">${tspans}<title>${name}: ${hit} of ${total}</title></text>`;
    })
    .join('');

  return `
<svg viewBox="0 0 440 380" role="img" aria-label="Skill coverage by category">
  ${rings}${spokes}
  <polygon points="${valuePts}" fill="color-mix(in srgb, var(--accent) 22%, transparent)" stroke="var(--accent)" stroke-width="2" />
  ${dots}${labels}
</svg>`.trim();
}
