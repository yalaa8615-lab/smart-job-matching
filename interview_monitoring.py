
import streamlit as st
import mediapipe as mp
import time
import threading
import numpy as np
import time
import threading
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

class InterviewMonitor:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def add_event(self, image, timestamp):
        with self.lock:
            self.events.append({
                "time": timestamp,
                "image": image
            })

    def get_events(self):
        with self.lock:
            return list(self.events)


class InterviewProcessor(VideoProcessorBase):

    def __init__(self, monitor):
        self.monitor = monitor
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.away_since = None
        self.last_capture = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        status = "Looking at screen"

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose = landmarks[1]

            eye_distance = abs(
                right_eye.x - left_eye.x
            )

            eye_center = (
                left_eye.x + right_eye.x
            ) / 2

            if eye_distance > 0:

                deviation = abs(
                    nose.x - eye_center
                ) / eye_distance

                looking_away = deviation > 0.20

                if looking_away:

                    status = "Looking away indicator"

                    if self.away_since is None:
                        self.away_since = time.monotonic()

                    duration = (
                        time.monotonic() - self.away_since
                    )

                    if (
                        duration >= 3
                        and time.monotonic() - self.last_capture >= 10
                    ):

                        success, encoded = cv2.imencode(
                            ".jpg", img
                        )

                        if success:
                            image_bytes = encoded.tobytes()

                            self.monitor.add_event(
                                image_bytes,
                                time.strftime("%H:%M:%S")
                            )

                            self.last_capture = time.monotonic()

                else:
                    self.away_since = None

        else:
            status = "Face not detected"
            self.away_since = None

        cv2.putText(
            img,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        return type(frame).from_ndarray(
            img,
            format="bgr24"
        )


def show_interview_page():

    st.title("🎥 AI Mock Interview")

    st.write(
        "Welcome to HIRE BATH Interview Monitoring System"
    )

    st.info(
        "This system monitors head direction during the interview. "
        "Looking away is only an indicator, not proof of cheating."
    )

    consent = st.checkbox(
        "I consent to live camera processing and capturing "
        "screenshots when a looking-away indicator is detected."
    )

    if consent:

        if "interview_monitor" not in st.session_state:
            st.session_state.interview_monitor = InterviewMonitor()

        monitor = st.session_state.interview_monitor

        st.subheader("Live Interview Camera")

        webrtc_streamer(
            key="hire_bath_interview",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: InterviewProcessor(
                monitor
            ),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.caption(
            "Start the camera and allow browser camera access. "
            "Stop the camera when the interview ends."
        )

        st.divider()

        if st.button("Refresh Interview Report"):
            
            st.rerun()

        events = monitor.get_events()

        st.subheader("Interview Monitoring Report")

        st.metric(
            "Looking Away Events",
            len(events)
        )

        if events:

            for index, event in enumerate(events):

                st.write(
                    f"Event {index + 1} - {event['time']}"
                )

                st.image(
                    event["image"],
                    caption="Captured interview frame",
                    width=400
                )

        else:
            st.success(
                "No looking-away events have been recorded."
            )

        st.caption(
            "Screenshots are stored temporarily in application "
            "memory during this session. They are not permanently "
            "saved."
        )

    else:
        st.warning(
            "Please provide consent before starting the camera."
        )
