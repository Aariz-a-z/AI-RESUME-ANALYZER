/**
 * PDF text extraction.
 *
 * pdf.js hands back positioned text fragments, not lines. The analyzer depends
 * on line structure (bullets mark requirements, blank-ish lines separate
 * sections), so fragments are regrouped by vertical position first - this is
 * what PyMuPDF's get_text("text") gave us on the Python side.
 */

import * as pdfjs from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import type { TextItem } from 'pdfjs-dist/types/src/display/api';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

/** Fragments whose baselines differ by less than this belong to one line. */
const LINE_TOLERANCE = 3;

export async function extractPdfText(file: File): Promise<string> {
  const data = await file.arrayBuffer();
  // v6 puts teardown on the loading task, not on the document proxy.
  const loadingTask = pdfjs.getDocument({ data });
  const doc = await loadingTask.promise;
  const pages: string[] = [];

  try {
    for (let pageNo = 1; pageNo <= doc.numPages; pageNo++) {
      const page = await doc.getPage(pageNo);
      const content = await page.getTextContent();

      const rows: Array<{ y: number; items: Array<{ x: number; text: string }> }> = [];
      for (const raw of content.items) {
        const item = raw as TextItem;
        if (!item.str || !item.transform) continue;
        const x = item.transform[4] as number;
        const y = item.transform[5] as number;

        const row = rows.find((r) => Math.abs(r.y - y) <= LINE_TOLERANCE);
        if (row) row.items.push({ x, text: item.str });
        else rows.push({ y, items: [{ x, text: item.str }] });
      }

      // Top of the page downwards, then left to right within each line.
      rows.sort((a, b) => b.y - a.y);
      const lines = rows.map((row) =>
        row.items
          .sort((a, b) => a.x - b.x)
          .map((i) => i.text)
          .join('')
          .replace(/\s+/g, ' ')
          .trim()
      );

      pages.push(lines.filter(Boolean).join('\n'));
      page.cleanup();
    }
  } finally {
    await loadingTask.destroy();
  }

  return pages.join('\n');
}
