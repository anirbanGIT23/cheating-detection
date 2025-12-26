import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)


def get_head_pose(image, landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[287].x * img_w, landmarks[287].y * img_h),
        (landmarks[57].x * img_w, landmarks[57].y * img_h)
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = 0

    return roll, pitch, yaw


def estimate_gaze(landmarks):
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    center_eye = (left_eye + right_eye) / 2
    gaze_vector = right_eye - left_eye
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
    return float(gaze_vector[0]), float(gaze_vector[1])


def load_reference():
    if not os.path.exists("reference.json"):
        print("⚠️ No baseline found! Please run baseline capture first.")
        return None
    with open("reference.json", "r") as f:
        return json.load(f)


def compute_deviation(current, reference):
    roll_diff = current[0] - reference["head_pose"]["roll"]
    pitch_diff = current[1] - reference["head_pose"]["pitch"]
    yaw_diff = current[2] - reference["head_pose"]["yaw"]
    angle_deviation = np.sqrt(roll_diff ** 2 + pitch_diff ** 2 + yaw_diff ** 2)
    return angle_deviation, (roll_diff, pitch_diff, yaw_diff)


def load_questions(folder_path="questions"):
    """Load all question images from the specified folder"""
    if not os.path.exists(folder_path):
        print(f"⚠️ Questions folder '{folder_path}' not found!")
        return []

    question_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    questions = []

    for qfile in question_files:
        img_path = os.path.join(folder_path, qfile)
        img = cv2.imread(img_path)
        if img is not None:
            questions.append((qfile, img))

    print(f"✅ Loaded {len(questions)} questions")
    return questions


def create_rnn_model(sequence_length=10):
    """Create a simple RNN model for pattern detection"""
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, 1)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def analyze_timing_pattern(timings, model, sequence_length=10):
    """Analyze timing patterns using RNN to detect suspicious behavior"""
    if len(timings) < sequence_length:
        return False, 0.0

    # Normalize timings
    recent_timings = list(timings)[-sequence_length:]
    normalized = np.array(recent_timings).reshape(-1, 1)
    mean_time = np.mean(normalized)
    std_time = np.std(normalized) + 1e-6
    normalized = (normalized - mean_time) / std_time

    # Reshape for RNN input
    X = normalized.reshape(1, sequence_length, 1)

    # Predict
    prediction = model.predict(X, verbose=0)[0][0]

    # Heuristic-based detection (since we don't have pre-trained weights)
    # Detect patterns: very consistent times or unusually fast responses
    time_variance = np.var(recent_timings)
    mean_response_time = np.mean(recent_timings)

    # Suspicious if: very low variance (bot-like) or very fast responses
    is_suspicious = time_variance < 1.0 or mean_response_time < 3.0
    confidence = prediction

    return is_suspicious, float(confidence)


def save_session_data(timings, deviations, malpractice_flags):
    """Save session data for analysis"""
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "timings": timings,
        "deviations": deviations,
        "malpractice_detected": malpractice_flags,
        "total_questions": len(timings) + 1
    }

    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(session_data, f, indent=4)
    print(f"\n📊 Session data saved to {filename}")


def main():
    mode = input("Enter mode: (1) Capture Baseline  (2) Deviation Check + Exam → ").strip()

    cap = cv2.VideoCapture(0)

    # Initialize for mode 2
    questions = []
    current_q_idx = 0
    question_start_time = None
    timings = deque(maxlen=20)  # Store last 20 transition times
    deviation_log = []
    malpractice_flags = []
    rnn_model = None

    if mode == "2":
        reference = load_reference()
        if not reference:
            cap.release()
            return

        questions = load_questions()
        if not questions:
            print("⚠️ No questions found. Please add JPG images to 'questions' folder.")
            cap.release()
            return

        question_start_time = time.time()
        print("➡ Exam mode started. Use RIGHT ARROW to navigate. Press ESC to exit.")

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        if mode == "1":
            print("➡ Look straight and press SPACE to capture baseline.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            h, w, _ = frame.shape

            # Create display frame
            if mode == "2" and questions:
                # Resize question image to fit beside camera feed
                question_img = questions[current_q_idx][1].copy()
                q_height, q_width = question_img.shape[:2]

                # Scale question to match camera height
                scale = h / q_height
                new_q_width = int(q_width * scale)
                new_q_height = h
                question_img_resized = cv2.resize(question_img, (new_q_width, new_q_height))

                # Combine camera feed and question side by side
                display_frame = np.hstack([frame, question_img_resized])
            else:
                display_frame = frame.copy()

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                head_pose = get_head_pose(frame, face_landmarks, w, h)
                gaze_vec = estimate_gaze(face_landmarks)

                if mode == "1":
                    cv2.putText(display_frame, "Press SPACE to capture baseline", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                elif mode == "2" and head_pose:
                    deviation, (rd, pd, yd) = compute_deviation(head_pose, reference)
                    status = "✅ Aligned" if deviation < 10 else "⚠️ Deviation!"
                    color = (0, 255, 0) if deviation < 10 else (0, 0, 255)

                    deviation_log.append(deviation)

                    # Display deviation info on camera feed
                    cv2.putText(display_frame, f"Deviation: {deviation:.1f}° | {status}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display_frame, f"Q {current_q_idx + 1}/{len(questions)}", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Check for malpractice pattern
                    if len(timings) >= 3:
                        is_suspicious, confidence = analyze_timing_pattern(timings)
                        if is_suspicious:
                            cv2.putText(display_frame, f"⚠️ SUSPICIOUS PATTERN! ({confidence:.2f})", (20, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            if not any(f["question"] == current_q_idx for f in malpractice_flags):
                                malpractice_flags.append({
                                    "question": current_q_idx,
                                    "time": time.time(),
                                    "confidence": confidence
                                })

            # Display instructions
            if mode == "2" and questions:
                cv2.putText(display_frame, "Z KEY SCORE: Next Question | ESC: Exit",
                            (20, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Exam Proctoring System", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                if mode == "2":
                    save_session_data(list(timings), deviation_log, malpractice_flags)
                break

            if mode == "1" and key == 32:  # SPACE
                if results.multi_face_landmarks and head_pose:
                    ref_data = {
                        "head_pose": {"roll": head_pose[0], "pitch": head_pose[1], "yaw": head_pose[2]},
                        "gaze_vector": {"x_g": gaze_vec[0], "y_g": gaze_vec[1]}
                    }
                    with open("reference.json", "w") as f:
                        json.dump(ref_data, f, indent=4)
                    print("\n✅ Baseline saved to reference.json")
                    print(ref_data)
                    break

            elif mode == "2" and key == 122:  # Z key
                if questions and current_q_idx < len(questions) - 1:
                    # Record time taken
                    time_taken = time.time() - question_start_time
                    timings.append(time_taken)
                    print(f"⏱️ Question {current_q_idx + 1} time: {time_taken:.2f}s")

                    # Move to next question
                    current_q_idx += 1
                    question_start_time = time.time()
                elif questions and current_q_idx == len(questions) - 1:
                    print("\n✅ All questions completed!")
                    save_session_data(list(timings), deviation_log, malpractice_flags)
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()