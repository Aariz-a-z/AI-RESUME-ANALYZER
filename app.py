"""Redirect stub for the old Streamlit Community Cloud deployment.

The analyzer now runs as a static site on GitHub Pages. Streamlit Cloud is
pointed at app.py and that entry point cannot be changed from the repository,
so app.py forwards visitors instead. The real Streamlit app is app_full.py and
still runs locally:

    pip install -r requirements-full.txt
    streamlit run app_full.py

Keeping this stub dependency-free is deliberate. The previous deployment kept
falling over loading torch and the embedding model inside the free tier's
memory; a redirect that only needs Streamlit itself has nothing left to crash.
"""

import streamlit as st
import streamlit.components.v1 as components

NEW_URL = "https://aariz-a-z.github.io/AI-RESUME-ANALYZER/"

st.set_page_config(page_title="Moved - Resume Fit Analyzer", page_icon="📄")

# Streamlit serves the app inside a same-origin iframe, so the parent frame is
# reachable; fall back to this frame if that ever stops being true.
components.html(
    f"""
    <script>
      const url = {NEW_URL!r};
      try {{ window.top.location.replace(url); }}
      catch (e) {{ window.location.replace(url); }}
    </script>
    """,
    height=0,
)

st.title("📄 This app has moved")
st.markdown(f"## [Open the Resume Fit Analyzer]({NEW_URL})")
st.info("You should be redirected automatically. If not, use the link above.")
st.caption(
    "The analyzer now runs entirely in your browser - your resume is never "
    "uploaded anywhere."
)
