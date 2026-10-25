import re

import plotly.graph_objects as go
import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer

import analyzer

st.set_page_config(page_title="AI Resume Fit Analyzer", page_icon="📄", layout="wide")

STATUS_STYLE = {
    "covered": ("✅", "Covered"),
    "partial": ("🟡", "Partial"),
    "missing": ("❌", "Not found"),
}


@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file) -> str:
    """Keep line breaks: bullet structure is what the JD/resume parsing relies on."""
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


def score_color(score: int) -> str:
    if score >= 75:
        return "#16a34a"
    if score >= 58:
        return "#65a30d"
    if score >= 40:
        return "#d97706"
    return "#dc2626"


def gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            # The number sits inside the arc, so it needs vertical room of its
            # own; the verdict is rendered as text below rather than as a title.
            number={"suffix": "%", "font": {"size": 40}},
            domain={"x": [0, 1], "y": [0, 0.82]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                # A thin bar keeps the centre of the dial clear for the number.
                "bar": {"color": score_color(score), "thickness": 0.32},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 58], "color": "#fef3c7"},
                    {"range": [58, 75], "color": "#ecfccb"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=30, r=30, t=10, b=0))
    return fig


def coverage_radar(category_coverage: dict[str, tuple[int, int]]) -> go.Figure | None:
    if len(category_coverage) < 3:
        return None

    labels, values, hover = [], [], []
    for category, (hit, total) in sorted(category_coverage.items()):
        # Wrap long category names so the axis labels do not clip.
        labels.append(category.replace(" & ", "<br>& "))
        values.append(round(100 * hit / total))
        hover.append(f"{category}: {hit} of {total}")

    fig = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name="Coverage",
            line_color="#0ea5e9",
            fillcolor="rgba(14,165,233,0.25)",
            text=hover + hover[:1],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=False,
        height=380,
        margin=dict(l=70, r=70, t=40, b=40),
    )
    return fig


def build_report(result: analyzer.Analysis) -> str:
    lines = [
        "AI RESUME FIT ANALYSIS",
        "=" * 60,
        f"Overall fit:            {result.overall}%  ({result.verdict})",
        f"  Skill coverage:       {result.skill_percent}%",
        f"  Requirement coverage: {result.requirement_percent}%",
        "",
        "MISSING REQUIRED SKILLS",
        "-" * 60,
    ]
    if result.missing_required:
        for skill in result.missing_required:
            lines.append(f"  - {skill.name} ({skill.category})")
            if skill.related:
                lines.append(f"      related experience you do have: {', '.join(skill.related)}")
            if skill.learn_url:
                lines.append(f"      learn: {skill.learn_url}")
    else:
        lines.append("  None - every required skill in the posting appears in your resume.")

    lines += ["", "MISSING NICE-TO-HAVE SKILLS", "-" * 60]
    lines.append(
        "  " + (", ".join(s.name for s in result.missing_preferred) or "None.")
    )

    lines += ["", "REQUIREMENT BY REQUIREMENT", "-" * 60]
    for req in result.requirements:
        _, label = STATUS_STYLE[req.status]
        lines.append(f"  [{label:<9}] ({req.priority}) {req.text}")
        if req.skills_missing:
            lines.append(f"      not found in resume: {', '.join(req.skills_missing)}")
        if req.status != "missing" and req.evidence:
            lines.append(f"      matched on: {req.evidence[:150]}")

    lines += ["", "ATS READABILITY", "-" * 60]
    for ok, message in result.ats:
        lines.append(f"  [{'OK  ' if ok else 'WARN'}] {message}")

    if result.years_required:
        estimate = (
            f"{result.years_estimated:.0f}" if result.years_estimated is not None else "could not estimate"
        )
        lines += ["", "EXPERIENCE", "-" * 60,
                  f"  Posting asks for ~{result.years_required} years; resume suggests {estimate}."]

    lines += ["", "Scores blend literal skill matching with semantic similarity.",
              "Treat them as guidance, not a hiring decision."]
    return "\n".join(lines)


# --- Sidebar -----------------------------------------------------------------

st.sidebar.header("📝 Inputs")
uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF)", type="pdf")
job_description = st.sidebar.text_area(
    "Paste the job description", height=280,
    placeholder="Paste the full posting, including the Requirements and Preferred sections.",
)
analyze_clicked = st.sidebar.button("Analyze Fit", type="primary", use_container_width=True)

st.sidebar.caption(
    "Keeping the posting's bullet structure gives the best results - "
    "each bullet is scored as its own requirement."
)

# --- Run ---------------------------------------------------------------------

st.title("🚀 AI Resume & Career Fit Analyzer")
st.markdown(
    "Scores your resume against a job description requirement by requirement, "
    "and tells you exactly which skills are missing."
)

if analyze_clicked:
    if uploaded_file is None or not job_description.strip():
        st.warning("Upload a PDF resume and paste a job description to begin.")
    else:
        with st.spinner("Reading your resume and scoring each requirement..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            model = load_model()
            # Stored so later interactions (tabs, downloads) do not wipe the results.
            st.session_state["result"] = analyzer.analyze(resume_text, job_description, model)
            st.session_state["resume_text"] = resume_text

result: analyzer.Analysis | None = st.session_state.get("result")

if result is None:
    st.info("👈 Upload your resume and paste a job description in the sidebar to get started.")
    st.subheader("What you get")
    st.markdown(
        """
        - **An overall fit score** that blends literal skill matching with semantic similarity.
        - **Requirement-by-requirement coverage** - every bullet in the posting, marked
          covered, partial or missing, with the line from your resume that matched it.
        - **A concrete skill gap list**, split into must-haves and nice-to-haves, with
          the related experience you already have.
        - **ATS readability checks** beyond a word count.
        """
    )
else:
    missing_required = result.missing_required

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        st.plotly_chart(gauge(result.overall), use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;margin-top:-18px;font-size:1.05rem;"
            f"font-weight:600;color:{score_color(result.overall)}'>{result.verdict}</div>",
            unsafe_allow_html=True,
        )
    c2.metric("Skill coverage", f"{result.skill_percent}%",
              help="Share of the technologies named in the posting that appear in your resume.")
    c3.metric("Requirements met", f"{result.requirement_percent}%",
              help="How well your resume evidences each requirement, semantically and literally.")
    c4.metric("Must-have gaps", len(missing_required),
              delta=None if not missing_required else f"-{len(missing_required)}",
              delta_color="inverse")

    if missing_required:
        st.error(
            "**Biggest gaps:** " + ", ".join(s.name for s in missing_required[:6])
            + ("" if len(missing_required) <= 6 else f" (+{len(missing_required) - 6} more)")
        )
    else:
        st.success("Every required skill named in this posting appears in your resume.")

    tab_reqs, tab_skills, tab_ats = st.tabs(
        ["📋 Requirements", "🧩 Skill gaps", "🖨️ ATS & experience"]
    )

    with tab_reqs:
        if not result.requirements:
            st.info("No distinct requirements were found - paste the full posting, bullets included.")
        else:
            counts = {k: sum(1 for r in result.requirements if r.status == k) for k in STATUS_STYLE}
            st.caption(
                f"{counts['covered']} covered · {counts['partial']} partial · "
                f"{counts['missing']} not found, across {len(result.requirements)} requirements"
            )
            for req in result.requirements:
                icon, label = STATUS_STYLE[req.status]
                tag = "must have" if req.priority == "required" else "nice to have"
                with st.expander(f"{icon} {req.text}", expanded=req.status == "missing"):
                    st.caption(f"{label} · {tag} · similarity {req.score:.2f}")
                    if req.skills_present:
                        st.markdown("**Found in your resume:** " + ", ".join(req.skills_present))
                    if req.skills_missing:
                        st.markdown("**Not found:** " + ", ".join(req.skills_missing))
                    if req.evidence and req.status != "missing":
                        st.markdown("**Closest line in your resume**")
                        st.info(req.evidence)
                    elif req.status == "missing":
                        st.markdown(
                            "Nothing in your resume addresses this. If you have the "
                            "experience, add a bullet that names it explicitly."
                        )

    with tab_skills:
        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("Must-have gaps")
            if missing_required:
                for skill in missing_required:
                    link = f" · [Learn]({skill.learn_url})" if skill.learn_url else ""
                    st.warning(f"**{skill.name}** — {skill.category}{link}")
                    if skill.related:
                        st.caption(f"Related experience you already have: {', '.join(skill.related)}")
            else:
                st.success("No must-have gaps.")

            if result.missing_preferred:
                st.subheader("Nice-to-have gaps")
                st.caption(", ".join(s.name for s in result.missing_preferred))

            if result.matched:
                st.subheader("Matched skills")
                st.caption(", ".join(s.name for s in result.matched))

        with right:
            st.subheader("Coverage by category")
            radar = coverage_radar(result.category_coverage)
            if radar is None:
                st.caption("Not enough distinct skill categories in this posting to chart.")
            else:
                st.plotly_chart(radar, use_container_width=True)

    with tab_ats:
        st.subheader("ATS readability")
        for ok, message in result.ats:
            (st.success if ok else st.warning)(message)

        st.subheader("Experience")
        if result.years_required:
            if result.years_estimated is None:
                st.info(
                    f"The posting asks for about {result.years_required} years. "
                    "No date ranges were detected in your resume, so this could not be checked."
                )
            elif result.years_estimated + 0.5 >= result.years_required:
                st.success(
                    f"Posting asks for ~{result.years_required} years; your dates suggest "
                    f"about {result.years_estimated:.0f}."
                )
            else:
                st.warning(
                    f"Posting asks for ~{result.years_required} years; your dates suggest "
                    f"about {result.years_estimated:.0f}."
                )
            st.caption("Estimated from date ranges, with overlapping roles merged.")
        else:
            st.caption("The posting does not state a years-of-experience requirement.")

    st.divider()
    d1, d2 = st.columns([1, 3])
    d1.download_button(
        "📥 Download report",
        data=build_report(result),
        file_name="resume_fit_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    with d2.expander("View the text extracted from your PDF"):
        st.text(st.session_state.get("resume_text", "")[:8000])
