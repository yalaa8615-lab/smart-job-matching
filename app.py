
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="HIRE BATH",
    page_icon="💼",
    layout="wide"
)

st.title("💼 HIRE BATH")
st.subheader("Smart Job Matching & Skill Gap Analysis")
st.write(
    "Upload your CV to discover suitable jobs "
    "and internship opportunities."
)

jobs = [
    {
        "Job": "Front-End Developer",
        "Requirements": ["HTML", "CSS", "JavaScript", "React", "Web Development"]
    },
    {
        "Job": "Python Back-End Developer",
        "Requirements": ["Python", "Django", "Flask", "SQL", "REST APIs"]
    },
    {
        "Job": "Data Analyst",
        "Requirements": ["Python", "Pandas", "SQL", "Data Visualization", "Excel"]
    },
    {
        "Job": "Machine Learning Engineer",
        "Requirements": ["Python", "Machine Learning", "TensorFlow", "Deep Learning"]
    },
    {
        "Job": "UI/UX Designer",
        "Requirements": ["Figma", "User Research", "Wireframing", "Prototyping"]
    }
]

uploaded_file = st.file_uploader(
    "Upload your CV (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Analyze My CV"):
        with st.spinner("Analyzing your CV..."):
            reader = PdfReader(uploaded_file)
            cv_text = ""

            for page in reader.pages:
                cv_text += page.extract_text() or ""

            if not cv_text.strip():
                st.error(
                    "Could not extract text from this PDF. "
                    "Please try a text-based PDF."
                )
            else:
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )

                cv_vector = model.encode([cv_text])
                results = []

               for job in jobs:
    requirements = job["Requirements"]

    matched_skills = [
        skill for skill in requirements
        if skill.lower() in cv_text.lower()
    ]

    missing_skills = [
        skill for skill in requirements
        if skill.lower() not in cv_text.lower()
    ]

    job_vector = model.encode([" ".join(requirements)])
    score = cosine_similarity(
        cv_vector, job_vector
    )[0][0]

    results.append({
        "Job": job["Job"],
        "Match Score (%)": round(float(score) * 100, 2),
        "Matched Skills": ", ".join(matched_skills) or "None",
        "Missing Skills": ", ".join(missing_skills) or "None"
    })

                df = pd.DataFrame(results)
                df = df.sort_values(
                    "Match Score (%)",
                    ascending=False
                )

                st.success("CV analysis completed!")
                st.subheader("Your Job Matches")
                st.dataframe(df, use_container_width=True)

                st.info(
                    "These scores measure text similarity, "
                    "not a guaranteed probability of getting hired."
                )
