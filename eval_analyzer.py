"""Evaluation harness for the fit scorer.

Run: python eval_analyzer.py

Checks three properties that the previous whole-document cosine similarity
failed:

1. Separation   - a matching candidate must outscore a mismatched one by a wide
                  margin, not the ~15 points the old scorer produced.
2. Ordering     - strong > partial > unrelated, for the same job description.
3. No truncation - skills listed at the END of a long resume must still be seen.
                  all-MiniLM-L6-v2 cuts off at 256 word pieces, so a
                  document-level embedding never reaches them.
"""

from __future__ import annotations

import sys

from sentence_transformers import SentenceTransformer

import analyzer

# --- Fixtures ----------------------------------------------------------------

ML_RESUME = """Priya Raman
Senior Machine Learning Engineer | priya@example.com | +91 98765 43210 | Bengaluru

SUMMARY
Machine learning engineer with six years building and shipping production models.

EXPERIENCE
Senior ML Engineer, Northwind Analytics (2021 - present)
- Built and deployed a churn prediction system serving 4 million users daily.
- Led migration of the training stack to distributed GPU training, cutting
  model iteration time from eighteen hours to under three.
- Designed the feature store backing all production models in the company.
- Mentored four junior engineers through model design and code review.
- Owned on-call for the inference service and its latency budget.

ML Engineer, Kestrel Software (2019 - 2021)
- Developed document classification pipelines over scanned contracts.
- Built retrieval pipelines over a corpus of twelve million legal documents.
- Automated retraining and rollback with continuous integration workflows.
- Reduced inference cost 40 percent by quantising and batching requests.

EDUCATION
B.E. Computer Science, Anna University, 2019

TECHNICAL SKILLS
Python, PyTorch, scikit-learn, natural language processing, SQL, PostgreSQL,
Docker, Kubernetes, AWS, Apache Spark, Airflow, data structures and algorithms
"""

FRONTEND_RESUME = """Arjun Mehta
Frontend Engineer | arjun@example.com | +91 91234 56789 | Pune

SUMMARY
Frontend engineer focused on design systems and accessible interfaces.

EXPERIENCE
Frontend Engineer, Lumen Retail (2021 - present)
- Rebuilt the storefront in React and TypeScript, improving load time by half.
- Built a component library adopted by three product teams.
- Drove the accessibility audit to WCAG AA compliance across checkout.
- Cut bundle size 35 percent through route-level code splitting.

Junior Developer, Tanish Web (2019 - 2021)
- Built responsive marketing pages and an internal CMS.
- Migrated legacy jQuery widgets to modern component patterns.

EDUCATION
B.Tech Information Technology, Pune University, 2019

TECHNICAL SKILLS
React, TypeScript, JavaScript, Next.js, CSS, Tailwind CSS, Redux, HTML,
responsive design, Jest, Git, REST APIs
"""

ANALYST_RESUME = """Sneha Kulkarni
Data Analyst | sneha@example.com | +91 99887 76655 | Hyderabad

SUMMARY
Data analyst supporting growth and finance teams with reporting and insight.

EXPERIENCE
Data Analyst, Harbor Logistics (2021 - present)
- Built the weekly executive reporting pack used by the leadership team.
- Wrote SQL models over the warehouse powering all revenue dashboards.
- Ran A/B tests on pricing changes and reported statistical significance.
- Automated a manual reconciliation process, saving six hours per week.

Analyst, Finserv Partners (2020 - 2021)
- Maintained forecasting spreadsheets and quarterly variance analysis.

EDUCATION
B.Sc Statistics, Osmania University, 2020

TECHNICAL SKILLS
SQL, Excel, Tableau, Python, pandas, statistics, data visualization,
hypothesis testing, PostgreSQL
"""

ML_JD = """Machine Learning Engineer

About the role:
You will build and deploy production machine learning systems that serve
millions of users.

Requirements:
- 4+ years of experience building machine learning models in Python
- Strong experience with PyTorch or TensorFlow in production
- Solid understanding of data structures and algorithms
- Experience writing performant SQL against relational databases
- Hands-on experience deploying services with Docker on AWS
- Experience with natural language processing pipelines

Preferred qualifications:
- Familiarity with Kubernetes for orchestration
- Experience with Apache Spark for large scale data processing
- Exposure to large language models is a plus
"""

FRONTEND_JD = """Senior Frontend Engineer

Requirements:
- 3+ years building production web applications with React
- Strong TypeScript and modern JavaScript experience
- Experience with CSS architecture and responsive design
- Ability to consume REST APIs and manage client state
- Experience with component testing

Preferred qualifications:
- Familiarity with Next.js server rendering
- Experience with accessibility standards is a plus
"""


def band(score: int) -> str:
    return "#" * (score // 5)


def main() -> int:
    print("loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    cases = [
        ("ML resume",       ML_RESUME,       "ML JD",       ML_JD),
        ("Analyst resume",  ANALYST_RESUME,  "ML JD",       ML_JD),
        ("Frontend resume", FRONTEND_RESUME, "ML JD",       ML_JD),
        ("Frontend resume", FRONTEND_RESUME, "Frontend JD", FRONTEND_JD),
        ("ML resume",       ML_RESUME,       "Frontend JD", FRONTEND_JD),
    ]

    results: dict[tuple[str, str], analyzer.Analysis] = {}
    print(f"\n{'resume':<17}{'job':<14}{'score':>6}{'skills':>8}{'reqs':>7}  verdict")
    print("-" * 78)
    for rname, resume, jname, jd in cases:
        a = analyzer.analyze(resume, jd, model)
        results[(rname, jname)] = a
        print(f"{rname:<17}{jname:<14}{a.overall:>5}%{a.skill_percent:>7}%{a.requirement_percent:>6}%  {a.verdict}")
        print(f"{'':17}{'':14}{band(a.overall)}")

    strong = results[("ML resume", "ML JD")]
    partial = results[("Analyst resume", "ML JD")]
    unrelated = results[("Frontend resume", "ML JD")]
    fe_strong = results[("Frontend resume", "Frontend JD")]

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    print("\nOrdering")
    check("strong > partial", strong.overall > partial.overall,
          f"{strong.overall}% vs {partial.overall}%")
    check("partial > unrelated", partial.overall > unrelated.overall,
          f"{partial.overall}% vs {unrelated.overall}%")

    print("\nSeparation")
    gap = strong.overall - unrelated.overall
    check("match vs mismatch gap >= 40 pts", gap >= 40, f"{gap} points")
    check("strong match scores >= 70", strong.overall >= 70, f"{strong.overall}%")
    check("mismatch scores <= 35", unrelated.overall <= 35, f"{unrelated.overall}%")
    check("frontend pairing >= 70", fe_strong.overall >= 70, f"{fe_strong.overall}%")

    print("\nNo truncation (skills listed at the end of the resume)")
    # The ML resume's TECHNICAL SKILLS block sits past the 256 word-piece cutoff,
    # so a document-level embedding never reaches it.
    tail_skills = {"Apache Spark", "Airflow", "Kubernetes", "Data Structures and Algorithms"}
    detected = set(analyzer.find_skills(analyzer.normalize(ML_RESUME)))
    missed = tail_skills - detected
    check("tail-of-resume skills detected", not missed,
          "all found" if not missed else f"missed {sorted(missed)}")
    # ...and the ones this JD actually asks for must be credited as matches.
    jd_tail = {"Apache Spark", "Kubernetes", "Data Structures and Algorithms"}
    unmatched = jd_tail - {s.name for s in strong.matched}
    check("tail skills credited against the JD", not unmatched,
          "all credited" if not unmatched else f"missed {sorted(unmatched)}")
    check("resume split into multiple chunks", strong.resume_chunks >= 3,
          f"{strong.resume_chunks} chunks")

    print("\nGap reporting")
    check("mismatch reports missing required skills", len(unrelated.missing_required) >= 3,
          f"{len(unrelated.missing_required)} missing")
    check("strong match reports few required gaps", len(strong.missing_required) <= 1,
          f"{len(strong.missing_required)} missing: {[s.name for s in strong.missing_required]}")
    check("requirements were extracted", len(strong.requirements) >= 6,
          f"{len(strong.requirements)} requirements")
    check("preferred tier detected", any(r.priority == "preferred" for r in strong.requirements),
          f"{sum(1 for r in strong.requirements if r.priority == 'preferred')} preferred")

    print("\nSymbol-preserving skill matching")
    for text, expected in [
        ("Built services in C++ and C#", "C++"),
        ("Experience with Node.js required", "Node.js"),
        ("Owns the CI/CD pipeline", "CI/CD"),
        ("Maintains ASP.NET services", ".NET"),
    ]:
        hits = analyzer.find_skills(analyzer.normalize(text))
        check(f"{expected!r} in {text!r}", expected in hits, ", ".join(sorted(hits)) or "nothing")

    print("\n" + "-" * 78)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
