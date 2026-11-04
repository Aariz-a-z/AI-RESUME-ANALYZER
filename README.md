# AI Resume Fit Analyzer

**Live: https://aariz-a-z.github.io/AI-RESUME-ANALYZER/**

Scores a resume against a job description requirement by requirement, and names
the skills that are missing. The live version runs entirely in the browser, so
no resume is ever uploaded anywhere.

## Features
- Overall fit score blending literal skill matching with semantic similarity
- Requirement-by-requirement coverage, each marked covered / partial / missing,
  with the resume line that evidenced it
- Skill gap list split into must-haves and nice-to-haves, across 136 skills
- ATS readability checks and a years-of-experience estimate

## How it works
The job description is split into individual requirements and the resume into
overlapping chunks, then each requirement is matched against its best chunk.
This matters: `all-MiniLM-L6-v2` truncates at 256 word pieces, so embedding a
whole two-page resume as one vector never sees past roughly its first quarter.

Scoring is hybrid. A comma-separated skills list embeds poorly against prose, so
literal skill evidence and semantic similarity compete and the stronger wins.

## Two implementations

| | Browser (live) | Python (reference) |
|---|---|---|
| Entry point | `web/` | `app_full.py` |
| Model | transformers.js, int8 | sentence-transformers, fp32 |
| PDF | pdf.js | PyMuPDF |
| Hosting | GitHub Pages (static) | local |

They score the same: `web/scripts/eval.ts` checks the TypeScript port against
scores generated from the Python analyzer. Everything lexical matches exactly
and the overall score lands within a point, the difference being int8 weights
in the browser against fp32 in Python.

## Run the Python version locally

```
pip install -r requirements-full.txt
streamlit run app_full.py
python eval_analyzer.py        # scoring checks
```

## Run the browser version locally

```
cd web
npm install
npm run dev
npm run eval                   # parity against the Python baseline
```

## Note on app.py
`app.py` is a redirect stub. Streamlit Community Cloud is pointed at that entry
point and it cannot be changed from the repository, so it forwards visitors to
the live site. The real Streamlit app is `app_full.py`.
