
import os
import time
import threading
from datetime import datetime
from io import BytesIO
from html import escape

import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage
)

# Optional Arabic PDF support
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_landmarker (3).task"
)

ICE_SERVERS = [
    {"urls": ["stun:stun.relay.metered.ca:80"]},
    {
        "urls": ["turn:global.relay.metered.ca:80"],
        "username": "f363667191b616bf8197b5b1",
        "credential": "13r6iWwyq/QoJYO6"
    },
    {
        "urls": ["turn:global.relay.metered.ca:80?transport=tcp"],
        "username": "f363667191b616bf8197b5b1",
        "credential": "13r6iWwyq/QoJYO6"
    },
    {
        "urls": ["turn:global.relay.metered.ca:443"],
        "username": "f363667191b616bf8197b5b1",
        "credential": "13r6iWwyq/QoJYO6"
    },
    {
        "urls": ["turns:global.relay.metered.ca:443?transport=tcp"],
        "username": "f363667191b616bf8197b5b1",
        "credential": "13r6iWwyq/QoJYO6"
    }
]

RTC_CONFIGURATION = {"iceServers": ICE_SERVERS}


# =====================================
# 2. JOB QUESTION BANK
# =====================================

QUESTION_BANK = {
    "Front-End Developer": [
        {
            "question": "What is the difference between HTML, CSS, and JavaScript?",
            "keywords": ["html", "css", "javascript", "structure", "style", "behavior"]
        },
        {
            "question": "What is the difference between Flexbox and CSS Grid?",
            "keywords": ["flexbox", "grid", "layout", "one-dimensional", "two-dimensional"]
        },
        {
            "question": "What are React components and props?",
            "keywords": ["react", "components", "props", "reusable", "data"]
        },
        {
            "question": "What is the Virtual DOM in React?",
            "keywords": ["virtual dom", "react", "real dom", "update", "performance"]
        },
        {
            "question": "How do you make a website responsive?",
            "keywords": ["responsive", "media queries", "flexible", "mobile", "screen"]
        }
    ],

    "Python Back-End Developer": [
        {
            "question": "What are the main features of Python?",
            "keywords": ["python", "readable", "interpreted", "dynamic", "libraries"]
        },
        {
            "question": "What is the difference between a list and a tuple?",
            "keywords": ["list", "tuple", "mutable", "immutable", "elements"]
        },
        {
            "question": "What is Django and why is it used?",
            "keywords": ["django", "framework", "web", "database", "security"]
        },
        {
            "question": "What is a REST API?",
            "keywords": ["rest", "api", "http", "request", "response", "json"]
        },
        {
            "question": "What is the purpose of SQL in back-end development?",
            "keywords": ["sql", "database", "query", "data", "tables"]
        }
    ],

    "Data Analyst": [
        {
            "question": "What is the difference between data analysis and data visualization?",
            "keywords": ["data", "analysis", "visualization", "insights", "charts"]
        },
        {
            "question": "How do you handle missing values in a dataset?",
            "keywords": ["missing", "null", "delete", "imputation", "mean", "median"]
        },
        {
            "question": "What is the difference between INNER JOIN and LEFT JOIN?",
            "keywords": ["inner join", "left join", "matching", "rows", "tables"]
        },
        {
            "question": "How do you use Pandas in Python?",
            "keywords": ["pandas", "dataframe", "data", "cleaning", "analysis"]
        },
        {
            "question": "What makes a good Power BI dashboard?",
            "keywords": ["power bi", "dashboard", "kpi", "visualization", "filters", "insights"]
        }
    ],

    "Machine Learning Engineer": [
        {
            "question": "What is the difference between supervised and unsupervised learning?",
            "keywords": ["supervised", "unsupervised", "labeled", "unlabeled", "data"]
        },
        {
            "question": "What is overfitting in machine learning?",
            "keywords": ["overfitting", "training", "testing", "generalization", "model"]
        },
        {
            "question": "What is the purpose of train-test split?",
            "keywords": ["train", "test", "split", "evaluate", "model", "data"]
        },
        {
            "question": "What is a neural network?",
            "keywords": ["neural network", "layers", "neurons", "weights", "learning"]
        },
        {
            "question": "How do you evaluate a classification model?",
            "keywords": ["accuracy", "precision", "recall", "f1", "confusion matrix"]
        }
    ],

    "UI/UX Designer": [
        {
            "question": "What is the difference between UI and UX?",
            "keywords": ["ui", "ux", "interface", "experience", "user"]
        },
        {
            "question": "What is a user persona?",
            "keywords": ["persona", "user", "goals", "needs", "research"]
        },
        {
            "question": "What is wireframing?",
            "keywords": ["wireframe", "layout", "structure", "design", "prototype"]
        },
        {
            "question": "Why is usability testing important?",
            "keywords": ["usability", "testing", "users", "feedback", "improve"]
        },
        {
            "question": "How do you use Figma in a design project?",
            "keywords": ["figma", "design", "prototype", "components", "collaboration"]
        }
    ]
}


# =====================================
# 3. MONITORING STORAGE
# =====================================

class InterviewMonitor:

    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def add_event(self, image, timestamp, duration):

        with self.lock:
            event_id = len(self.events)

            self.events.append({
                "id": event_id,
                "time": timestamp,
                "duration": duration,
                "image": image,
                "finished": False
            })

            return event_id

    def finish_event(self, event_id, duration):

        if event_id is None:
            return

        with self.lock:
            if 0 <= event_id < len(self.events):
                self.events[event_id]["duration"] = duration
                self.events[event_id]["finished"] = True

    def get_events(self):

        with self.lock:
            return [event.copy() for event in self.events]


# =====================================
# 4. VIDEO PROCESSOR
# =====================================

class InterviewProcessor(VideoProcessorBase):

    def __init__(self, monitor):

        self.monitor = monitor
        self.away_since = None
        self.last_capture = 0
        self.last_timestamp = 0
        self.current_event_id = None
        self.landmarker = None

        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=MODEL_PATH
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.landmarker = (
                vision.FaceLandmarker.create_from_options(options)
            )

        except Exception as error:
            self.initialization_error = str(error)

    def close_away_event(self, current_time):

        if self.away_since is None:
            return

        duration = round(current_time - self.away_since, 1)

        self.monitor.finish_event(
            self.current_event_id,
            duration
        )

        self.away_since = None
        self.current_event_id = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        if self.landmarker is None:
            cv2.putText(
                img,
                "Model initialization failed",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            return type(frame).from_ndarray(img, format="bgr24")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            current_time = time.monotonic()

            timestamp_ms = max(
                int(current_time * 1000),
                self.last_timestamp + 1
            )
            self.last_timestamp = timestamp_ms

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            results = self.landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            status = "Looking at screen"
            color = (0, 255, 0)

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]
                left_side = landmarks[234]
                right_side = landmarks[454]
                nose = landmarks[1]

                face_width = abs(right_side.x - left_side.x)
                face_center = (left_side.x + right_side.x) / 2

                if face_width > 0:

                    deviation = abs(nose.x - face_center) / face_width

                    if deviation > 0.12:

                        status = "Looking away indicator"
                        color = (0, 165, 255)

                        if self.away_since is None:
                            self.away_since = current_time

                        duration = current_time - self.away_since

                        if (
                            duration >= 3
                            and self.current_event_id is None
                            and current_time - self.last_capture >= 10
                        ):

                            success, encoded = cv2.imencode(".jpg", img)

                            if success:
                                timestamp = datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )

                                self.current_event_id = (
                                    self.monitor.add_event(
                                        encoded.tobytes(),
                                        timestamp,
                                        round(duration, 1)
                                    )
                                )

                                self.last_capture = current_time

                    else:
                        self.close_away_event(current_time)

            else:
                status = "Face not detected"
                color = (0, 0, 255)
                self.close_away_event(current_time)

            cv2.putText(
                img,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        except Exception:
            cv2.putText(
                img,
                "Processing frame...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

        return type(frame).from_ndarray(img, format="bgr24")

    def __del__(self):

        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass


# =====================================
# 5. ANSWER SCORING
# =====================================

def evaluate_answer(answer, keywords):

    answer = (answer or "").strip().lower()

    if not answer:
        return 0, [], "No answer provided"

    matched = [
        keyword for keyword in keywords
        if keyword.lower() in answer
    ]

    word_count = len(answer.split())

    keyword_score = (
        len(matched) / len(keywords) * 70
        if keywords else 0
    )

    length_score = min(word_count / 50, 1) * 30

    score = min(100, round(keyword_score + length_score))

    if score >= 75:
        feedback = "Good keyword coverage and answer length"
    elif score >= 45:
        feedback = "Partially covered; add more relevant details"
    else:
        feedback = "Needs a more detailed and relevant answer"

    return score, matched, feedback


# =====================================
# 6. PDF REPORT
# =====================================

def prepare_pdf_text(value):

    value = str(value or "")

    if ARABIC_SUPPORT:
        try:
            value = get_display(arabic_reshaper.reshape(value))
        except Exception:
            pass

    return escape(value).replace("\n", "<br/>")


def create_pdf_report(job, answers, events):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#24527A"),
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=7
    )

    story = []

    story.append(Paragraph("HIRE BATH", title_style))
    story.append(Paragraph("AI Mock Interview Report", title_style))

    story.append(Paragraph(
        f"<b>Job Position:</b> {prepare_pdf_text(job)}",
        body_style
    ))

    story.append(Paragraph(
        f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body_style
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph("Interview Answers", heading_style))

    question_list = QUESTION_BANK[job]
    total_score = 0
    answered_count = 0

    for index, item in enumerate(question_list):

        answer = answers.get(index, "")

        score, matched, feedback = evaluate_answer(
            answer,
            item["keywords"]
        )

        if answer.strip():
            total_score += score
            answered_count += 1

        story.append(Paragraph(
            f"<b>Question {index + 1}:</b> "
            f"{prepare_pdf_text(item['question'])}",
            body_style
        ))

        story.append(Paragraph(
            f"<b>Answer:</b> {prepare_pdf_text(answer or 'Not answered')}",
            body_style
        ))

        story.append(Paragraph(
            f"<b>Preliminary Score:</b> {score}/100",
            body_style
        ))

        story.append(Paragraph(
            f"<b>Matched Keywords:</b> "
            f"{prepare_pdf_text(', '.join(matched) or 'None')}",
            body_style
        ))

        story.append(Paragraph(
            f"<b>Feedback:</b> {prepare_pdf_text(feedback)}",
            body_style
        ))

        story.append(Spacer(1, 8))

    average_score = (
        round(total_score / answered_count)
        if answered_count else 0
    )

    story.append(Paragraph("Answer Summary", heading_style))

    story.append(Paragraph(
        f"<b>Questions Answered:</b> {answered_count}/{len(question_list)}",
        body_style
    ))

    story.append(Paragraph(
        f"<b>Average Preliminary Score:</b> {average_score}/100",
        body_style
    ))

    story.append(Paragraph(
        "Note: This is a basic keyword-based indicator, not an AI or human assessment.",
        body_style
    ))

    story.append(Paragraph("Camera Monitoring", heading_style))

    completed_events = [
        event for event in events if event["finished"]
    ]

    total_away_time = sum(
        event["duration"] for event in completed_events
    )

    story.append(Paragraph(
        f"<b>Looking Away Events:</b> {len(events)}",
        body_style
    ))

    story.append(Paragraph(
        f"<b>Total Recorded Away Time:</b> {total_away_time:.1f} seconds",
        body_style
    ))

    for index, event in enumerate(events):

        duration_text = (
            f"{event['duration']:.1f} seconds"
            if event["finished"]
            else "Ongoing"
        )

        story.append(Paragraph(
            f"<b>Event {index + 1}</b> - "
            f"{prepare_pdf_text(event['time'])} - {duration_text}",
            body_style
        ))

        try:
            image = PDFImage(
                BytesIO(event["image"]),
                width=10 * cm,
                height=6 * cm
            )
            story.append(image)
            story.append(Spacer(1, 8))
        except Exception:
            story.append(Paragraph(
                "Screenshot could not be added.",
                body_style
            ))

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# =====================================
# 7. INTERVIEW QUESTIONS PAGE
# =====================================

def show_questions_section():

    st.divider()
    st.header("AI Interview Questions")

    st.write(
        "Choose a job position to practice role-specific interview questions."
    )

    jobs = list(QUESTION_BANK.keys())

    selected_job = st.selectbox(
        "Choose Job Position",
        jobs,
        key="interview_selected_job"
    )

    if "interview_answers" not in st.session_state:
        st.session_state.interview_answers = {}

    if "interview_question_index" not in st.session_state:
        st.session_state.interview_question_index = 0

    if "interview_previous_job" not in st.session_state:
        st.session_state.interview_previous_job = selected_job

    if st.session_state.interview_previous_job != selected_job:

        st.session_state.interview_question_index = 0
        st.session_state.interview_previous_job = selected_job

    answers_by_job = st.session_state.interview_answers

    if selected_job not in answers_by_job:
        answers_by_job[selected_job] = {}

    answers = answers_by_job[selected_job]
    questions = QUESTION_BANK[selected_job]

    index = st.session_state.interview_question_index
    index = max(0, min(index, len(questions) - 1))

    st.progress(
        (index + 1) / len(questions)
    )

    st.caption(
        f"Question {index + 1} of {len(questions)}"
    )

    question = questions[index]

    st.subheader(question["question"])

    answer_key = f"interview_answer_{selected_job}_{index}"

    if answer_key not in st.session_state:
        st.session_state[answer_key] = answers.get(index, "")

    with st.form(key=f"question_form_{selected_job}_{index}"):

        st.text_area(
            "Your Answer",
            key=answer_key,
            height=160,
            placeholder="Write your answer here..."
        )

        submitted = st.form_submit_button(
            "Save Answer & Next"
            if index < len(questions) - 1
            else "Save Final Answer"
        )

        if submitted:

            current_answer = st.session_state[answer_key].strip()

            answers[index] = current_answer

            if index < len(questions) - 1:
                st.session_state.interview_question_index += 1

            st.rerun()

    col1, col2 = st.columns(2)

    with col1:

        if index > 0:

            if st.button("Previous Question"):

                st.session_state.interview_question_index -= 1
                st.rerun()

    with col2:

        if index < len(questions) - 1:

            if st.button("Next Question"):

                st.session_state.interview_question_index += 1
                st.rerun()

    answered_count = sum(
        1 for value in answers.values()
        if value.strip()
    )

    st.caption(
        f"Answers saved: {answered_count}/{len(questions)}"
    )

    if answered_count == len(questions):

        st.success("You have answered all interview questions!")

    if answered_count > 0:

        st.subheader("Preliminary Answer Evaluation")

        for question_index, item in enumerate(questions):

            answer = answers.get(question_index, "")

            if answer.strip():

                score, matched, feedback = evaluate_answer(
                    answer,
                    item["keywords"]
                )

                with st.expander(
                    f"Question {question_index + 1} - Score: {score}/100"
                ):

                    st.write(feedback)

                    st.write(
                        "Matched keywords:",
                        ", ".join(matched) if matched else "None"
                    )


# =====================================
# 8. MAIN INTERVIEW PAGE
# =====================================

def show_interview_page():

    st.title("AI Mock Interview")

    st.write(
        "Welcome to HIRE BATH Interview Monitoring System"
    )

    st.info(
        "The system monitors face direction. "
        "Looking away is an indicator, not proof of cheating."
    )

    if not os.path.isfile(MODEL_PATH):

        st.error(
            "Face Landmarker model file was not found."
        )
        return

    if "interview_monitor" not in st.session_state:

        st.session_state.interview_monitor = InterviewMonitor()

    monitor = st.session_state.interview_monitor

    consent = st.checkbox(
        "I consent to camera processing and screenshot "
        "capture when a looking-away indicator is detected."
    )

    if consent:

        st.subheader("Live Interview Camera")

        webrtc_ctx = webrtc_streamer(
            key="hire_bath_interview",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: InterviewProcessor(
                monitor
            ),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        if webrtc_ctx.state.playing:
            st.success("Camera is running.")
        else:
            st.warning(
                "Camera is not connected yet. "
                "Allow camera permission and start it."
            )

        st.caption(
            "Stop the camera when the interview ends."
        )

    else:

        st.warning(
            "Please provide consent before starting camera monitoring."
        )

    # Questions
    show_questions_section()

    # Monitoring report
    st.divider()
    st.header("Interview Monitoring Report")

    events = monitor.get_events()

    completed_events = [
        event for event in events if event["finished"]
    ]

    total_duration = sum(
        event["duration"] for event in completed_events
    )

    average_duration = (
        total_duration / len(completed_events)
        if completed_events else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Looking Away Events",
        len(events)
    )

    col2.metric(
        "Total Away Time",
        f"{total_duration:.1f} sec"
    )

    col3.metric(
        "Average Duration",
        f"{average_duration:.1f} sec"
    )

    if events:

        report_data = []

        for index, event in enumerate(events):

            report_data.append({
                "Event": index + 1,
                "Date & Time": event["time"],
                "Duration": (
                    f"{event['duration']:.1f} sec"
                    if event["finished"]
                    else "Ongoing"
                ),
                "Status": (
                    "Completed"
                    if event["finished"]
                    else "Ongoing"
                )
            })

        st.dataframe(
            pd.DataFrame(report_data),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Captured Screenshots")

        for index, event in enumerate(events):

            with st.expander(
                f"Event {index + 1} - {event['time']}"
            ):

                st.write(
                    "Duration: "
                    + (
                        f"{event['duration']:.1f} seconds"
                        if event["finished"]
                        else "Ongoing"
                    )
                )

                st.image(
                    event["image"],
                    caption="Captured interview frame",
                    width=450
                )

    else:

        st.success(
            "No looking-away events have been recorded."
        )

    # PDF REPORT
    st.divider()
    st.subheader("Download Final Interview Report")

    selected_job = st.session_state.get(
        "interview_selected_job",
        list(QUESTION_BANK.keys())[0]
    )

    answers = st.session_state.get(
        "interview_answers",
        {}
    ).get(selected_job, {})

    pdf_bytes = create_pdf_report(
        selected_job,
        answers,
        events
    )

    st.download_button(
        label="Download Interview Report (PDF)",
        data=pdf_bytes,
        file_name="HIRE_BATH_Interview_Report.pdf",
        mime="application/pdf"
    )

    st.caption(
        "The report and screenshots are temporarily stored in "
        "application memory during this session."
    )


