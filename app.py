
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

from interview_monitoring import show_interview_page


st.set_page_config(
    page_title="HIRE BATH",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# DESIGN / CUSTOM STYLE
# ==============================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #172554;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: transparent;
        padding: 9px 12px;
        border-radius: 8px;
    }

    .main-header {
        background: linear-gradient(120deg, #172554, #2563EB);
        padding: 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .main-header p {
        color: #DBEAFE;
        font-size: 17px;
    }

    .section-title {
        color: #172554;
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .service-card {
        background-color: white;
        padding: 24px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        min-height: 190px;
        margin-bottom: 15px;
    }

    .service-card h3 {
        color: #172554;
        font-size: 20px;
    }

    .service-card p {
        color: #64748B;
        font-size: 15px;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
    }

    h1, h2, h3 {
        color: #172554;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        padding: 25px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# NAVIGATION
# ==============================

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"

st.sidebar.markdown("""
<div style="text-align:center; padding:15px 0 25px 0;">
    <div style="font-size:38px;">💼</div>
    <h2 style="color:white; margin:5px;">HIRE BATH</h2>
    <p style="color:#BFDBFE; font-size:12px;">
        Your Career Starts Here
    </p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "MAIN MENU",
    ["Home", "CV Analysis", "AI Mock Interview"],
    key="navigation"
)

st.sidebar.markdown("---")
st.sidebar.caption("Smart Job Matching Platform")


# ==============================
# JOBS DATABASE
# ==============================

jobs = [
    {
        "Job": "Front-End Developer",
        "Requirements": [
            "HTML", "CSS", "JavaScript", "React",
            "Web Development"
        ]
    },
    {
        "Job": "Python Back-End Developer",
        "Requirements": [
            "Python", "Django", "Flask", "SQL", "REST APIs"
        ]
    },
    {
        "Job": "Data Analyst",
        "Requirements": [
            "Python", "Pandas", "SQL",
            "Data Visualization", "Excel"
        ]
    },
    {
        "Job": "Machine Learning Engineer",
        "Requirements": [
            "Python", "Machine Learning",
            "TensorFlow", "Deep Learning"
        ]
    },
    {
        "Job": "UI/UX Designer",
        "Requirements": [
            "Figma", "User Research",
            "Wireframing", "Prototyping"
        ]
    }
]


# ==============================
# SKILL ALIASES
# ==============================

skill_aliases = {
    "JavaScript": ["javascript", "js"],
    "HTML": ["html"],
    "CSS": ["css"],
    "React": ["react", "react.js"],
    "Web Development": ["web development", "web developer"],
    "Python": ["python"],
    "Django": ["django"],
    "Flask": ["flask"],
    "SQL": ["sql", "sql server", "mysql", "postgresql"],
    "REST APIs": ["rest api", "rest apis", "restful api"],
    "Pandas": ["pandas"],
    "Data Visualization": [
        "data visualization", "data visualisation"
    ],
    "Excel": ["excel", "microsoft excel"],
    "Machine Learning": ["machine learning"],
    "TensorFlow": ["tensorflow"],
    "Deep Learning": ["deep learning"],
    "Figma": ["figma"],
    "User Research": ["user research"],
    "Wireframing": ["wireframing", "wireframes"],
    "Prototyping": ["prototyping", "prototype"]
}


# ==============================
# HELPER FUNCTIONS
# ==============================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return " ".join(text.split())


def skill_is_present(skill, cv_text):
    aliases = skill_aliases.get(skill, [skill])
    normalized_cv = " " + normalize_text(cv_text) + " "

    for alias in aliases:
        normalized_alias = normalize_text(alias)

        if " " + normalized_alias + " " in normalized_cv:
            return True

    return False


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ==============================
# HOME PAGE
# ==============================

if page == "Home":

    st.markdown("""
    <div class="main-header">
        <h1>Welcome to HIRE BATH 👋</h1>
        <p>
            Smart Job Matching & Skill Gap Analysis
        </p>
        <p>
            Discover your skills, find suitable career opportunities,
            and prepare yourself for the job market.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Explore Our Services</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service-card">
            <h3>📄 CV Analysis</h3>
            <p>
                Upload your CV and let our AI analyze your skills
                and discover suitable job opportunities.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start CV Analysis", use_container_width=True):
            st.session_state["navigation"] = "CV Analysis"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="service-card">
            <h3>🎥 AI Mock Interview</h3>
            <p>
                Practice your interview skills, answer questions,
                and get ready for real job interviews.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Mock Interview", use_container_width=True):
            st.session_state["navigation"] = "AI Mock Interview"
            st.rerun()

    st.markdown("---")

    st.markdown(
        '<div class="section-title">How HIRE BATH Works</div>',
        unsafe_allow_html=True
    )

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        <div class="service-card">
            <h3>01. Upload</h3>
            <p>Upload your CV to start your career analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="service-card">
            <h3>02. Analyze</h3>
            <p>Discover your skills and match them with job requirements.</p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="service-card">
            <h3>03. Improve</h3>
            <p>Identify missing skills and prepare for your next opportunity.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        HIRE BATH | Smart Job Matching Platform
    </div>
    """, unsafe_allow_html=True)


# ==============================
# PAGE 1: CV ANALYSIS
# ==============================

elif page == "CV Analysis":

    st.markdown("""
    <div class="main-header">
        <h1>📄 CV Analysis</h1>
        <p>
            Analyze your CV, discover your strengths,
            and find the jobs that match your skills.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload Your CV")

    uploaded_file = st.file_uploader(
        "Choose your CV in PDF format",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "🚀 Analyze My CV",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Reading and analyzing your CV..."
            ):

                try:

                    reader = PdfReader(uploaded_file)

                    cv_text = ""

                    for page_pdf in reader.pages:
                        cv_text += (
                            page_pdf.extract_text() or ""
                        ) + "\n"

                    if not cv_text.strip():

                        st.error(
                            "No readable text was found in this PDF. "
                            "Please upload a text-based PDF."
                        )

                        st.stop()

                    model = load_model()

                    job_texts = [
                        " ".join(job["Requirements"])
                        for job in jobs
                    ]

                    all_texts = [cv_text] + job_texts

                    vectors = model.encode(all_texts)

                    cv_vector = vectors[0].reshape(1, -1)

                    job_vectors = vectors[1:]

                    results = []

                    for index, job in enumerate(jobs):

                        requirements = job["Requirements"]

                        matched_skills = [
                            skill
                            for skill in requirements
                            if skill_is_present(
                                skill,
                                cv_text
                            )
                        ]

                        missing_skills = [
                            skill
                            for skill in requirements
                            if skill not in matched_skills
                        ]

                        match_score = (
                            len(matched_skills)
                            / len(requirements)
                        ) * 100

                        similarity_score = float(
                            cosine_similarity(
                                cv_vector,
                                job_vectors[index].reshape(1, -1)
                            )[0][0]
                        ) * 100

                        results.append({

                            "Job": job["Job"],

                            "Match Score (%)": round(
                                match_score, 1
                            ),

                            "Matched Skills": ", ".join(
                                matched_skills
                            ) or "None detected",

                            "Missing Skills": ", ".join(
                                missing_skills
                            ) or "None detected",

                            "AI Text Similarity (%)": round(
                                similarity_score, 1
                            )

                        })

                    df = pd.DataFrame(results)

                    df = df.sort_values(
                        "Match Score (%)",
                        ascending=False
                    ).reset_index(drop=True)

                    st.session_state["analysis_results"] = df

                    st.session_state["selected_job"] = df.iloc[0]["Job"]

                    st.success(
                        "CV analysis completed!"
                    )

                except Exception as error:

                    st.error(
                        "Something went wrong while analyzing your CV."
                    )

                    st.code(str(error))


    # ==============================
    # DISPLAY SAVED RESULTS
    # ==============================

    if "analysis_results" in st.session_state:

        df = st.session_state["analysis_results"]

        st.markdown("---")

        st.subheader("🏆 Your Best Job Match")

        best_job = df.iloc[0]

        st.metric(
            label=best_job["Job"],
            value=f'{best_job["Match Score (%)"]}%'
        )

        st.subheader("📊 Job Matching Results")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Match Score is based on detected required skills. "
            "AI Text Similarity compares the CV text with job "
            "requirements. Neither score guarantees employment."
        )

        st.markdown("---")

        st.subheader("🎯 Skill Gap Analysis")

        selected_job = st.selectbox(
            "Choose a job to inspect",
            df["Job"].tolist(),
            key="selected_job"
        )

        selected_result = df[
            df["Job"] == selected_job
        ].iloc[0]

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### ✅ Matched Skills")

            matched = selected_result["Matched Skills"]

            if matched == "None detected":

                st.write(
                    "No required skills detected yet."
                )

            else:

                for skill in matched.split(", "):
                    st.success(skill)

        with col2:

            st.markdown("### 📚 Skills to Learn")

            missing = selected_result["Missing Skills"]

            if missing == "None detected":

                st.success(
                    "All listed skills were detected!"
                )

            else:

                for skill in missing.split(", "):
                    st.info(skill)

        st.markdown("---")

        st.subheader("🚀 Your Next Step")

        if selected_result["Missing Skills"] == "None detected":

            st.success(
                "Great! Your CV contains all the listed "
                "skills for this role. Keep building projects "
                "to demonstrate your abilities."
            )

        else:

            st.write(
                "Focus on learning and practicing these skills:"
            )

            st.warning(
                selected_result["Missing Skills"]
            )

            st.write(
                "Build a small project using these skills "
                "and add it to your portfolio."
            )


# ==============================
# PAGE 2: AI MOCK INTERVIEW
# ==============================

elif page == "AI Mock Interview":

    show_interview_page()
