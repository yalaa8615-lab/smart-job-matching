import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from audio_module import AudioTranscriber, draw_transcript  # NEW (audio)

# Settings
MODEL_PATH = "face_landmarker.task"
SAVE_DIR = "cheating_captures"

HEAD_TURN_LIMIT = 0.20
GAZE_LIMIT = 0.10
SUSPICIOUS_SECONDS = 1.0
COOLDOWN_SECONDS = 2.0
GRACE_SECONDS = 0.7

# Audio settings (NEW)
AUDIO_ENABLED = True                          # False = video only, as before
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # folder of the Vosk speech model
AUDIO_OUTPUT_DIR = "interview_audio"          # transcripts + audio are saved here

os.makedirs(SAVE_DIR, exist_ok=True)

# Load AI model
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=2
)

landmarker = vision.FaceLandmarker.create_from_options(options)


def analyze_face(lm):
    left = lm[234].x
    right = lm[454].x
    nose = lm[1].x

    face_w = max(right - left, 1e-6)
    head_offset = ((nose - left) / face_w) - 0.5

    def eye_ratio(iris, c1, c2):
        a = lm[c1].x
        b = lm[c2].x
        w = max(abs(b - a), 1e-6)
        return ((lm[iris].x - min(a, b)) / w) - 0.5

    gaze_offset = (
        eye_ratio(468, 33, 133) +
        eye_ratio(473, 362, 263)
    ) / 2

    return head_offset, gaze_offset


def save_capture(frame, reason):
    name = time.strftime("%Y%m%d_%H%M%S")
    name += f"_{reason}.jpg"

    path = os.path.join(SAVE_DIR, name)
    cv2.imwrite(path, frame)
    print("Saved:", path)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    landmarker.close()
    raise SystemExit

print("Privacy Notice: Webcam monitoring will start.")
answer = input("Do you consent? (y/n): ")

if answer.strip().lower() != "y":
    print("Consent not given.")
    cap.release()
    landmarker.close()
    raise SystemExit

# ---- Audio (NEW): the microphone starts ONLY after its own explicit consent ----
audio = None

if AUDIO_ENABLED:
    mic_answer = input(
        "Also record your voice and convert it to text (microphone)? (y/n): "
    )

    if mic_answer.strip().lower() == "y":
        audio = AudioTranscriber(
            model_path=VOSK_MODEL_PATH,
            output_dir=AUDIO_OUTPUT_DIR
        )

        if not audio.start():
            print("Audio disabled:", audio.error)
            print("Continuing with video monitoring only.")
            audio = None
    else:
        print("Microphone consent not given. Video monitoring only.")

print("Monitoring started. Press Q to quit.")

suspicious_since = None
last_saved = 0
last_seen_bad = 0
events_count = 0

while True:
    ok, frame = cap.read()

    if not ok:
        print("Could not read camera frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)
    faces = result.face_landmarks

    reason = None

    if len(faces) == 0:
        reason = "no_face"

    elif len(faces) > 1:
        reason = "multiple_faces"

    else:
        head, gaze = analyze_face(faces[0])

        cv2.putText(
            frame,
            f"head={head:.2f} gaze={gaze:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        if abs(head) > HEAD_TURN_LIMIT:
            reason = "head_turned"

        elif abs(gaze) > GAZE_LIMIT:
            reason = "looking_away"

    now = time.time()

    if reason:
        last_seen_bad = now

        if suspicious_since is None:
            suspicious_since = now

        elapsed = now - suspicious_since

        cv2.putText(
            frame,
            f"WARNING: {reason}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if (
            elapsed >= SUSPICIOUS_SECONDS
            and now - last_saved >= COOLDOWN_SECONDS
        ):
            save_capture(frame, reason)
            last_saved = now
            events_count += 1

    else:
        if (
            suspicious_since is not None
            and now - last_seen_bad > GRACE_SECONDS
        ):
            suspicious_since = None

        cv2.putText(
            frame,
            "OK",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 0),
            2
        )

    cv2.putText(
        frame,
        f"Potential events: {events_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 128, 255),
        2
    )

    # NEW (audio): drawn after any snapshot was saved, so snapshots are unchanged
    if audio is not None:
        draw_transcript(frame, audio)

    cv2.imshow("Interview Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

# NEW (audio): stop the microphone together with the camera
if audio is not None:
    audio.stop()

cv2.destroyAllWindows()
landmarker.close()

print("\nSession summary")
print("Potential events:", events_count)
print("Snapshots folder:", SAVE_DIR)

# NEW (audio): where to find the saved transcript
if audio is not None:
    print("Transcript saved to:", audio.transcript_txt)
    print("Audio recording:", audio.wav_path)
print("These events need human review; they are not proof of misconduct.")
