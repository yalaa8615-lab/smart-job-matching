
import os
import time
import threading
from datetime import datetime

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


# =========================
# 1. MODEL CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_landmarker (3).task"
)


# =========================
# 2. WEBRTC CONFIGURATION
# =========================

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

RTC_CONFIGURATION = {
    "iceServers": ICE_SERVERS
}


# =========================
# 3. MONITORING STORAGE
# =========================

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


# =========================
# 4. VIDEO PROCESSOR
# =========================

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
                vision.FaceLandmarker.create_from_options(
                    options
                )
            )

        except Exception as error:

            self.initialization_error = str(error)

    def close_away_event(self, current_time):

        if self.away_since is None:
            return

        duration = round(
            current_time - self.away_since, 1
        )

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

            return type(frame).from_ndarray(
                img,
                format="bgr24"
            )

        try:

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

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

                face_width = abs(
                    right_side.x - left_side.x
                )

                face_center = (
                    left_side.x + right_side.x
                ) / 2

                if face_width > 0:

                    deviation = (
                        abs(nose.x - face_center)
                        / face_width
                    )

                    if deviation > 0.12:

                        status = "Looking away indicator"
                        color = (0, 165, 255)

                        if self.away_since is None:
                            self.away_since = current_time

                        duration = (
                            current_time - self.away_since
                        )

                        # Record one event per looking-away episode.
                        if (
                            duration >= 3
                            and self.current_event_id is None
                            and current_time - self.last_capture >= 10
                        ):

                            success, encoded = cv2.imencode(
                                ".jpg",
                                img
                            )

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

        except Exception as error:

            cv2.putText(
                img,
                "Processing frame...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

        return type(frame).from_ndarray(
            img,
            format="bgr24"
        )

    def __del__(self):

        try:

            if self.landmarker is not None:
                self.landmarker.close()

        except Exception:
            pass


# =========================
# 5. INTERVIEW PAGE
# =========================

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

    if not consent:

        st.warning(
            "Please provide consent before starting the camera."
        )

        return

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
            "Camera is not connected yet."
        )

    st.caption(
        "Stop the camera when the interview ends."
    )

    st.divider()

    # =========================
    # 6. INTERVIEW REPORT
    # =========================

    st.subheader("Interview Monitoring Report")

    events = monitor.get_events()

    total_events = len(events)

    completed_events = [
        event for event in events
        if event["finished"]
    ]

    total_duration = sum(
        event["duration"]
        for event in completed_events
    )

    average_duration = (
        total_duration / len(completed_events)
        if completed_events else 0
    )

    # KPI CARDS

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Looking Away Events",
            total_events
        )

    with col2:

        st.metric(
            "Total Away Time",
            f"{total_duration:.1f} sec"
        )

    with col3:

        st.metric(
            "Average Duration",
            f"{average_duration:.1f} sec"
        )

    st.divider()

    # EVENT TABLE

    st.subheader("Event Details")

    if events:

        report_data = []

        for index, event in enumerate(events):

            duration_text = (
                f"{event['duration']:.1f} seconds"
                if event["finished"]
                else "Still looking away"
            )

            report_data.append({
                "Event": index + 1,
                "Date & Time": event["time"],
                "Duration": duration_text,
                "Status": (
                    "Completed"
                    if event["finished"]
                    else "Ongoing"
                )
            })

        report_df = pd.DataFrame(report_data)

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        # CSV DOWNLOAD

        csv_data = report_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="Download Interview Report (CSV)",
            data=csv_data,
            file_name="interview_monitoring_report.csv",
            mime="text/csv"
        )

        # SCREENSHOTS

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
                        else "Still looking away"
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

    st.caption(
        "Screenshots and report data are temporarily stored "
        "in application memory during this session."
    )



