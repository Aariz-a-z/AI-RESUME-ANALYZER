"""Landing stub for the old Streamlit Community Cloud deployment.

The analyzer now runs as a static site on GitHub Pages. Streamlit Cloud is
pointed at app.py and that entry point cannot be changed from the repository,
so app.py forwards visitors instead. The real Streamlit app is app_full.py and
still runs locally:

    pip install -r requirements-full.txt
    streamlit run app_full.py

This does not redirect automatically, and cannot. st.components.v1.html renders
inside a sandboxed iframe without allow-top-navigation, so window.top.location
is blocked by the browser, and Streamlit strips <script> from st.markdown. A
link the visitor clicks is the only thing that actually works here.

Keeping this stub dependency-free is deliberate. The previous deployment kept
falling over loading torch and the embedding model inside the free tier's
memory; a page that only needs Streamlit itself has nothing left to crash.
"""

import streamlit as st

NEW_URL = "https://aariz-a-z.github.io/AI-RESUME-ANALYZER/"

st.set_page_config(page_title="Moved - Resume Fit Analyzer", page_icon="📄")

st.title("📄 This app has moved")

st.markdown(
    "The **AI Resume Fit Analyzer** now runs as a static site. It is faster, "
    "works in every browser, and your resume never leaves your machine."
)

st.link_button(
    "Open the Resume Fit Analyzer →",
    NEW_URL,
    type="primary",
    use_container_width=True,
)

st.markdown(f"Or copy this address: `{NEW_URL}`")

st.divider()
st.caption(
    "The analyzer runs entirely in your browser - nothing is uploaded. "
    "The first visit downloads about 30 MB of model and runtime, then caches it."
)
