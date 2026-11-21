
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
# NAVIGATION - FIXED
# ==============================

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"

# Apply page navigation before creating the radio widget
if "next_page" in st.session_state:
    next_page = st.session_state.pop("next_page")
    st.session_state["navigation"] = next_page


# ==============================
# DESIGN
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

    .main-header {
        background: linear-gradient(120deg, #172554, #2563EB);
        padding: 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        color: white;
        font-size: 36px;
    }

    .main-header p {
        color: #DBEAFE;
        font-size: 16px;
    }

    .service-card {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        min-height: 170px;
        margin-bottom: 15px;
    }

    .service-card h3 {
        color: #172554;
    }

    .service-card p {
        color: #64748B;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        padding: 25px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# SIDEBAR NAVIGATION
# ==============================

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
# ATS KEYWORDS
# ==============================

ats_job_keywords = {
    "Front-End Developer": [
        "HTML", "CSS", "JavaScript", "React",
        "Responsive Design", "Git", "API"
    ],
    "Python Back-End Developer": [
        "Python", "Django", "Flask", "SQL",
        "REST API", "Git", "Backend"
    ],
    "Data Analyst": [
        "Python", "Pandas", "SQL", "Excel",
        "Power BI", "Data Visualization", "Statistics"
    ],
    "Machine Learning Engineer": [
        "Python", "Machine Learning", "TensorFlow",
        "Deep Learning", "Scikit-learn", "Data Science"
    ],
    "UI/UX Designer": [
        "Figma", "User Research", "Wireframing",
        "Prototyping", "UI Design", "UX Design"
    ]
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


def detect_target_job(cv_text):

    normalized_cv = normalize_text(cv_text)

    job_aliases = {
        "Front-End Developer": [
            "front end developer",
            "frontend developer",
            "front end"
        ],
        "Python Back-End Developer": [
            "python back end developer",
            "python backend developer",
            "backend developer",
            "back end developer"
        ],
        "Data Analyst": [
            "data analyst",
            "data analysis"
        ],
        "Machine Learning Engineer": [
            "machine learning engineer",
            "ml engineer"
        ],
        "UI/UX Designer": [
            "ui ux designer",
            "ui designer",
            "ux designer"
        ]
    }

    scores = {}

    for job, aliases in job_aliases.items():
        scores[job] = sum(
            1 for alias in aliases
            if alias in normalized_cv
        )

    best_job = max(scores, key=scores.get)

    if scores[best_job] == 0:
        return None

    return best_job


def calculate_ats_score(cv_text, target_job):

    keywords = ats_job_keywords[target_job]

    matched_keywords = [
        keyword for keyword in keywords
        if skill_is_present(keyword, cv_text)
    ]

    missing_keywords = [
        keyword for keyword in keywords
        if keyword not in matched_keywords
    ]

    sections = {
        "Contact Information": [
            "email", "phone", "linkedin"
        ],
        "Education": [
            "education", "university", "degree"
        ],
        "Experience": [
            "experience", "work experience", "employment"
        ],
        "Skills": [
            "skills", "technical skills"
        ],
        "Projects": [
            "projects", "personal projects"
        ]
    }

    normalized_cv = normalize_text(cv_text)

    found_sections = [
        section for section, aliases in sections.items()
        if any(
            normalize_text(alias) in normalized_cv
            for alias in aliases
        )
    ]

    missing_sections = [
        section for section in sections
        if section not in found_sections
    ]

    keyword_score = (
        len(matched_keywords) / len(keywords)
    ) * 60

    section_score = (
        len(found_sections) / len(sections)
    ) * 30

    readability_score = (
        10 if len(cv_text.strip()) >= 300 else 0
    )

    total_score = round(
        keyword_score + section_score + readability_score
    )

    return {
        "score": total_score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "found_sections": found_sections,
        "missing_sections": missing_sections
    }


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
        <p>Smart Job Matching & Skill Gap Analysis</p>
        <p>
            Discover your skills, find suitable career opportunities,
            and prepare yourself for the job market.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore Our Services")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="service-card">
            <h3>📄 CV Analysis & ATS</h3>
            <p>
                Analyze your CV, check ATS compatibility,
                and discover suitable jobs.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Start CV Analysis",
            use_container_width=True
        ):
            st.session_state["next_page"] = "CV Analysis"
            st.rerun()

    with col2:

        st.markdown("""
        <div class="service-card">
            <h3>🎥 AI Mock Interview</h3>
            <p>
                Practice interview questions and prepare
                for real job interviews.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Start Mock Interview",
            use_container_width=True
        ):
            st.session_state["next_page"] = "AI Mock Interview"
            st.rerun()

    st.markdown("---")
    st.subheader("How HIRE BATH Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        <div class="service-card">
            <h3>01. Upload</h3>
            <p>Upload your CV in PDF format.</p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="service-card">
            <h3>02. Analyze</h3>
            <p>Check your skills, ATS score, and job matching.</p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="service-card">
            <h3>03. Improve</h3>
            <p>Discover missing skills and improve your CV.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">HIRE BATH | Smart Job Matching Platform</div>',
        unsafe_allow_html=True
    )


# ==============================
# PAGE 1: CV ANALYSIS + ATS
# ==============================

elif page == "CV Analysis":

    st.markdown("""
    <div class="main-header">
        <h1>📄 CV Analysis & ATS Checker</h1>
        <p>
            Analyze your CV, discover matching jobs,
            and check your ATS compatibility.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload Your CV")

    uploaded_file = st.file_uploader(
        "Choose your CV in PDF format",
        type=["pdf"],
        key="cv_upload"
    )

    if uploaded_file is not None:

        if st.button(
            "🚀 Analyze My CV",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Reading and analyzing your CV..."):

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
                            if skill_is_present(skill, cv_text)
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
                            "Match Score (%)": round(match_score, 1),
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
                    st.session_state["cv_text"] = cv_text
                    st.session_state["selected_job"] = df.iloc[0]["Job"]

                    detected_job = detect_target_job(cv_text)

                    st.session_state["detected_ats_job"] = detected_job

                    st.success("CV analysis completed!")

                except Exception as error:

                    st.error(
                        "Something went wrong while analyzing your CV."
                    )

                    st.code(str(error))


    # ==============================
    # DISPLAY RESULTS
    # ==============================

    if "analysis_results" in st.session_state:

        df = st.session_state["analysis_results"]
        cv_text = st.session_state["cv_text"]

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
                st.write("No required skills detected yet.")
            else:
                for skill in matched.split(", "):
                    st.success(skill)

        with col2:

            st.markdown("### 📚 Skills to Learn")

            missing = selected_result["Missing Skills"]

            if missing == "None detected":
                st.success("All listed skills were detected!")
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

            st.warning(selected_result["Missing Skills"])

            st.write(
                "Build a small project using these skills "
                "and add it to your portfolio."
            )


        # ==============================
        # ATS CHECKER
        # ==============================

        st.markdown("---")

        st.markdown("""
        <div class="main-header">
            <h1>🤖 ATS Resume Checker</h1>
            <p>
                Check how well your CV matches the target job
                and discover ways to improve it.
            </p>
        </div>
        """, unsafe_allow_html=True)

        detected_job = st.session_state.get("detected_ats_job")

        if detected_job:

            st.success(
                f"Target job detected from your CV: {detected_job}"
            )

            target_job = detected_job

        else:

            st.warning(
                "We couldn't identify a target job from your CV. "
                "Please select the intended job."
            )

            target_job = st.selectbox(
                "Select your target job",
                list(ats_job_keywords.keys()),
                key="ats_manual_job"
            )

        ats_result = calculate_ats_score(cv_text, target_job)

        st.subheader("Your ATS Results")

        score_col, keyword_col, section_col = st.columns(3)

        with score_col:
            st.metric(
                "ATS Score",
                f'{ats_result["score"]}/100'
            )

        with keyword_col:
            st.metric(
                "Matched Keywords",
                f'{len(ats_result["matched_keywords"])}/'
                f'{len(ats_job_keywords[target_job])}'
            )

        with section_col:
            st.metric(
                "Detected Sections",
                f'{len(ats_result["found_sections"])}/5'
            )

        st.progress(ats_result["score"] / 100)

        if ats_result["score"] >= 80:

            st.success(
                "Your CV has a good result according to this basic ATS check."
            )

        elif ats_result["score"] >= 60:

            st.warning(
                "Your CV needs some improvements to better match the target job."
            )

        else:

            st.error(
                "Your CV needs improvement before applying for this job."
            )

        st.markdown("---")

        ats_col1, ats_col2 = st.columns(2)

        with ats_col1:

            st.markdown("### ✅ Matched Keywords")

            if ats_result["matched_keywords"]:

                for keyword in ats_result["matched_keywords"]:
                    st.success(keyword)

            else:

                st.write("No target keywords were detected.")

        with ats_col2:

            st.markdown("### 📌 Missing Keywords")

            if ats_result["missing_keywords"]:

                for keyword in ats_result["missing_keywords"]:
                    st.warning(keyword)

            else:

                st.success("All target keywords were detected!")

        st.markdown("---")

        section_col1, section_col2 = st.columns(2)

        with section_col1:

            st.markdown("### ✅ Detected CV Sections")

            for section in ats_result["found_sections"]:
                st.success(section)

        with section_col2:

            st.markdown("### 📚 Missing CV Sections")

            for section in ats_result["missing_sections"]:
                st.warning(section)

        st.markdown("---")
        st.subheader("💡 ATS Improvement Suggestions")

        suggestions = []

        if ats_result["missing_keywords"]:

            suggestions.append(
                "Add relevant missing job keywords only when "
                "they accurately describe your real skills."
            )

        if ats_result["missing_sections"]:

            suggestions.append(
                "Consider adding these sections to your CV: "
                + ", ".join(ats_result["missing_sections"])
            )

        if len(cv_text.strip()) < 300:

            suggestions.append(
                "Your CV text is quite short. Add more relevant "
                "details about your education, skills, and projects."
            )

        if not suggestions:

            suggestions.append(
                "Your CV covers the main checks in this basic ATS review. "
                "Keep your information accurate and up to date."
            )

        for suggestion in suggestions:
            st.info(suggestion)

        st.caption(
            "Note: This is an educational ATS-style checker based on "
            "text extraction, keyword matching, and section detection. "
            "It is not connected to a real employer ATS and does not "
            "guarantee that a CV will pass every company's screening system."
        )


# ==============================
# PAGE 2: AI MOCK INTERVIEW
# ==============================

elif page == "AI Mock Interview":

    show_interview_page()

