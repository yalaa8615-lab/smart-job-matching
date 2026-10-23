
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(
    page_title="HIRE BATH",
    page_icon="💼",
    layout="wide"
)

st.title("💼 HIRE BATH")
st.subheader("Smart Job Matching & Skill Gap Analysis")
st.write(
    "Find suitable jobs, discover missing skills, "
    "and plan your next career step."
)

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


uploaded_file = st.file_uploader(
    "Upload your CV (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Analyze My CV", type="primary"):
        with st.spinner("Reading and analyzing your CV..."):
            try:
                reader = PdfReader(uploaded_file)
                cv_text = ""

                for page in reader.pages:
                    cv_text += (page.extract_text() or "") + "\n"

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
                        skill for skill in requirements
                        if skill_is_present(skill, cv_text)
                    ]

                    missing_skills = [
                        skill for skill in requirements
                        if skill not in matched_skills
                    ]

                    match_score = (
                        len(matched_skills) / len(requirements)
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

                st.success("CV analysis completed!")

                best_job = df.iloc[0]

                st.subheader("Your Best Job Match")
                st.metric(
                    label=best_job["Job"],
                    value=f'{best_job["Match Score (%)"]}%'
                )

                st.subheader("Job Matching Results")
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

                st.subheader("Skill Gap Analysis")

                selected_job = st.selectbox(
                    "Choose a job to inspect",
                    df["Job"].tolist()
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
                            st.write(f"✅ {skill}")

                with col2:
                    st.markdown("### 📚 Skills to Learn")
                    missing = selected_result["Missing Skills"]

                    if missing == "None detected":
                        st.write("All listed skills were detected!")
                    else:
                        for skill in missing.split(", "):
                            st.write(f"📌 {skill}")

                st.subheader("Your Next Step")

                if selected_result["Missing Skills"] == "None detected":
                    st.info(
                        "Great! Your CV contains all the listed "
                        "skills for this role. Keep building projects "
                        "to demonstrate your abilities."
                    )
                else:
                    st.write(
                        "Focus on learning and practicing these skills:"
                    )
                    st.write(selected_result["Missing Skills"])
                    st.write(
                        "Build a small project using these skills "
                        "and add it to your portfolio."
                    )

            except Exception as error:
                st.error(
                    "Something went wrong while analyzing your CV."
                )
                st.code(str(error))
