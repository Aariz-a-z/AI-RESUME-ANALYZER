import streamlit as st
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go # <-- NEW: For the Radar Chart

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Resume Fit Analyzer", page_icon="📄", layout="wide")

# --- 2. Load the AI Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 3. Dictionaries & Logic ---
skill_ontology = {
    "Data Structures and Algorithms": ["dsa", "data structures", "algorithms"],
    "Machine Learning": ["machine learning", "ml", "predictive modeling"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Python": ["python", "python3"],
    "SQL": ["sql", "mysql", "postgresql", "databases"],
    "Amazon Web Services": ["aws", "amazon web services", "cloud"]
}

resource_links = {
    "Data Structures and Algorithms": "https://www.coursera.org/specializations/data-structures-algorithms",
    "Machine Learning": "https://www.coursera.org/learn/machine-learning",
    "Natural Language Processing": "https://www.deeplearning.ai/courses/natural-language-processing-specialization/",
    "Amazon Web Services": "https://aws.amazon.com/training/"
}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    return text.lower()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ats_check(text):
    word_count = len(text.split())
    if word_count < 50:
        return False, "⚠️ ATS Warning: Less than 50 words extracted. Formatting may be too complex."
    return True, f"✅ ATS Check Passed: {word_count} words extracted."

# --- 4. The Streamlit Dashboard UI ---
st.title("🚀 AI-Powered Resume & Career Fit Analyzer")
st.markdown("Analyze your resume against any Job Description using Deep Semantic NLP.")

with st.sidebar:
    st.header("📝 Upload Details")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
    st.markdown("---")
    job_description = st.text_area("Paste the Job Description here:", height=300)
    analyze_button = st.button("Analyze Fit", type="primary", use_container_width=True)

# Main Dashboard Area
if analyze_button:
    if uploaded_file is not None and job_description:
        with st.spinner('Analyzing semantics and plotting data...'):
            
            raw_resume = extract_text_from_pdf(uploaded_file)
            cleaned_resume = clean_text(raw_resume)
            cleaned_jd = clean_text(job_description.lower())
            
            ats_passed, ats_message = ats_check(cleaned_resume)
            
            embeddings1 = model.encode(cleaned_resume, convert_to_tensor=True)
            embeddings2 = model.encode(cleaned_jd, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            
            # The clamped percentage fix!
            raw_score = int(cosine_scores.item() * 100)
            match_percentage = max(0, min(100, raw_score))
            
            # --- NEW: Advanced Skill Mapping for the Radar Chart ---
            jd_skills_found = []
            resume_skills_matched = []
            missing_skills = []
            
            for standard_name, aliases in skill_ontology.items():
                is_in_jd = any(re.search(r'\b' + re.escape(alias) + r'\b', cleaned_jd) for alias in aliases)
                if is_in_jd:
                    jd_skills_found.append(standard_name)
                    is_in_resume = any(re.search(r'\b' + re.escape(alias) + r'\b', cleaned_resume) for alias in aliases)
                    
                    if is_in_resume:
                        resume_skills_matched.append(100) # 100% matched
                    else:
                        resume_skills_matched.append(0)   # 0% matched
                        missing_skills.append(standard_name)

            # --- Display Results ---
            if ats_passed:
                st.success(ats_message)
            else:
                st.error(ats_message)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Semantic Match Score", value=f"{match_percentage}%")
                st.progress(match_percentage / 100)
            
            with col2:
                st.metric(label="Critical Skills Missing", value=len(missing_skills))

            st.markdown("---")

            # --- NEW: Dashboard Layout for Charts & Insights ---
            dash_col1, dash_col2 = st.columns([1, 1])
            
            with dash_col1:
                st.subheader("🔍 Skill Gap Analysis")
                if missing_skills:
                    for skill in missing_skills:
                        if skill in resource_links:
                            st.warning(f"**Missing:** {skill} → [Learn Here]({resource_links[skill]})")
                        else:
                            st.warning(f"**Missing:** {skill}")
                else:
                    st.info("🎯 Excellent! Your resume covers all mapped skills.")
                    
            with dash_col2:
                st.subheader("📊 Skill Alignment Map")
                # Draw the Radar Chart if JD has skills mapped in our ontology
                if jd_skills_found:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=resume_skills_matched,
                        theta=jd_skills_found,
                        fill='toself',
                        name='Resume',
                        line_color='#00CC96'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Not enough standard skills detected in the JD to generate a map.")

            st.markdown("---")

            # --- NEW: Download Report Generation ---
            report_content = f"""
            AI RESUME ANALYSIS REPORT
            -------------------------
            Semantic Match Score: {match_percentage}%
            ATS Check: {ats_message}
            
            MISSING SKILLS:
            {', '.join(missing_skills) if missing_skills else 'None - Perfect Match!'}
            """
            
            st.download_button(
                label="📥 Download Analysis Report",
                data=report_content,
                file_name="resume_analysis_report.txt",
                mime="text/plain"
            )

            with st.expander("View Extracted Resume Text"):
                st.write(cleaned_resume)
            
    else:
        st.warning("Please upload a PDF and paste a Job Description to begin.")
else:
    st.info("👈 Please upload your resume and paste a job description in the sidebar to get started.")