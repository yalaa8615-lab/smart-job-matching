
import os
import time
import threading
import zipfile

import cv2
import mediapipe as mp
import streamlit as st

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker (3).task"
)

# Metered TURN configuration
ICE_SERVERS = [
    {
        "urls": ["stun:stun.relay.metered.ca:80"]
    },
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
        self.away_since = None
        self.last_capture = 0
        self.last_timestamp = 0

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        try:
            self.landmarker = (
                vision.FaceLandmarker.create_from_options(options)
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load Face Landmarker model. "
                f"Original error: {error}"
            ) from error

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        current_time = time.monotonic()

        timestamp_ms = int(current_time * 1000)
        timestamp_ms = max(
            timestamp_ms,
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

        if results.face_landmarks:

            landmarks = results.face_landmarks[0]

            left_side = landmarks[234]
            right_side = landmarks[454]
            nose = landmarks[1]

            face_width = abs(
                right_side.x - left_side.x
            )

            face_center = (
                left_side.x + right_side.x
            ) / 2

            if face_width > 0:

                deviation = (
                    abs(nose.x - face_center) / face_width
                )

                if deviation > 0.12:

                    status = "Looking away indicator"

                    if self.away_since is None:
                        self.away_since = current_time

                    duration = current_time - self.away_since

                    if (
                        duration >= 3
                        and current_time - self.last_capture >= 10
                    ):

                        success, encoded = cv2.imencode(
                            ".jpg", img
                        )

                        if success:
                            self.monitor.add_event(
                                encoded.tobytes(),
                                time.strftime("%H:%M:%S")
                            )

                            self.last_capture = current_time

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
            f"Model file not found: {os.path.basename(MODEL_PATH)}"
        )
        st.stop()

    if not zipfile.is_zipfile(MODEL_PATH):
        st.error(
            "The model file is not a valid task archive."
        )
        st.stop()

    consent = st.checkbox(
        "I consent to camera processing and screenshot capture "
        "when a looking-away indicator is detected."
    )

    if not consent:
        st.warning(
            "Please provide consent before starting the camera."
        )
        return

    if "interview_monitor" not in st.session_state:
        st.session_state.interview_monitor = InterviewMonitor()

    monitor = st.session_state.interview_monitor

    st.subheader("Live Interview Camera")

    webrtc_streamer(
        key="hire_bath_interview",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": ICE_SERVERS
        },
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
        "Allow camera access in your browser. "
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
        "Screenshots are temporarily stored in application "
        "memory during this session."
    )



