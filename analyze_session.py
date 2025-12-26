import json
import numpy as np
import glob
import os
from datetime import datetime
from tensorflow import keras

# Global variables for trained model
TRAINED_MODEL = None
NORMALIZATION_PARAMS = None


def load_trained_model():
    """Load the trained RNN model and normalization parameters"""
    global TRAINED_MODEL, NORMALIZATION_PARAMS

    if not os.path.exists('rnn_cheating_detector.h5'):
        print("⚠️ Warning: Trained model 'rnn_cheating_detector.h5' not found!")
        print("   RNN predictions will use heuristics only.")
        print("   Run 'train_rnn_detector.py' first to train the model.\n")
        return False

    if not os.path.exists('rnn_normalization.npz'):
        print("⚠️ Warning: Normalization parameters 'rnn_normalization.npz' not found!")
        print("   RNN predictions will use heuristics only.\n")
        return False

    try:
        TRAINED_MODEL = keras.models.load_model('rnn_cheating_detector.h5')
        NORMALIZATION_PARAMS = np.load('rnn_normalization.npz')
        print("✅ Loaded trained RNN model successfully\n")
        return True
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        print("   RNN predictions will use heuristics only.\n")
        return False


def prepare_rnn_features(timings):
    """Prepare sequential features for RNN analysis"""
    if len(timings) < 2:
        return None

    if NORMALIZATION_PARAMS is None:
        # Fallback to local normalization if params not loaded
        features = []
        cumulative = 0

        for i, t in enumerate(timings):
            cumulative += t
            time_diff = t - timings[i - 1] if i > 0 else 0
            features.append([t, cumulative, time_diff])

        features = np.array(features, dtype=np.float32)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        return features.reshape(1, -1, 3)

    # Use saved normalization parameters
    features = []
    cumulative = 0

    for i, t in enumerate(timings):
        cumulative += t
        time_diff = t - timings[i - 1] if i > 0 else 0
        features.append([t, cumulative, time_diff])

    features = np.array(features, dtype=np.float32)

    # Apply saved normalization
    mean = NORMALIZATION_PARAMS['mean']
    std = NORMALIZATION_PARAMS['std']
    features = (features - mean) / std

    # Pad to max_length if necessary
    max_length = int(NORMALIZATION_PARAMS['max_length'])
    if len(features) < max_length:
        padding = np.zeros((max_length - len(features), 3), dtype=np.float32)
        features = np.vstack([features, padding])
    elif len(features) > max_length:
        features = features[:max_length]

    return features.reshape(1, max_length, 3)


def rnn_pattern_detection(timings):
    """Use trained RNN to detect sequential cheating patterns"""
    if len(timings) < 2:
        return 0.0

    features = prepare_rnn_features(timings)

    if features is None:
        return 0.0

    # Use trained model if available
    if TRAINED_MODEL is not None:
        try:
            prediction = TRAINED_MODEL.predict(features, verbose=0)[0][0]
        except Exception as e:
            print(f"⚠️ RNN prediction failed: {e}")
            prediction = 0.5  # Default fallback
    else:
        # Fallback to heuristics if model not loaded
        timings_array = np.array(timings)
        cv = np.std(timings_array) / (np.mean(timings_array) + 1e-6)

        if cv < 0.20:
            prediction = 0.7
        elif cv < 0.30:
            prediction = 0.5
        else:
            prediction = 0.3

    # Enhance prediction with heuristics
    timings_array = np.array(timings)
    cv = np.std(timings_array) / (np.mean(timings_array) + 1e-6)

    if cv < 0.20:  # Very consistent pacing
        prediction = max(prediction, 0.7)

    if len(timings) >= 3:
        acceleration = np.diff(timings)
        if np.all(acceleration < 0):  # Continuously getting faster
            prediction = max(prediction, 0.65)

    return float(prediction)


def calculate_timing_suspicion(timings):
    """Enhanced timing suspicion with total time range check and RNN"""
    if len(timings) < 2:
        return 0.0

    timings_array = np.array(timings)
    total_time = np.sum(timings_array)
    mean_time = np.mean(timings_array)
    std_time = np.std(timings_array)
    cv = std_time / (mean_time + 1e-6)

    suspicion_score = 0.0

    # Check if total time falls in suspicious range (20-30s)
    if 20 <= total_time <= 30:
        suspicion_score += 0.45  # Strong indicator of cheating
    elif 18 <= total_time <= 35:
        suspicion_score += 0.25  # Moderate suspicion

    # Coefficient of variation check (consistency)
    if cv < 0.12:  # More sensitive
        suspicion_score += 0.35
    elif cv < 0.20:
        suspicion_score += 0.20

    # Mean response time check
    if mean_time < 4.0:  # More sensitive
        suspicion_score += 0.25
    elif mean_time < 6.0:
        suspicion_score += 0.15

    # Variance check
    variance = np.var(timings_array)
    if variance < 1.5:  # More sensitive
        suspicion_score += 0.20
    elif variance < 3.0:
        suspicion_score += 0.12

    # RNN-based pattern detection
    rnn_score = rnn_pattern_detection(timings)
    suspicion_score += rnn_score * 0.3  # Weight RNN contribution at 30%

    # Acceleration pattern (getting faster over time)
    if len(timings) >= 3:
        first_half = np.mean(timings_array[:len(timings) // 2])
        second_half = np.mean(timings_array[len(timings) // 2:])
        if first_half > second_half * 1.4:  # More sensitive
            suspicion_score += 0.20

    # Check for uniform/duplicate response times
    unique_times = len(np.unique(np.round(timings_array, 1)))
    if unique_times <= len(timings) * 0.5:  # 50%+ duplicate times
        suspicion_score += 0.25

    return min(suspicion_score, 1.0)


def calculate_deviation_suspicion(deviations, threshold=10):
    """Calculate suspicion score based on head pose deviations (0-1)"""
    if not deviations:
        return 0.0

    deviations_array = np.array(deviations)

    high_deviation_ratio = np.sum(deviations_array > threshold) / len(deviations_array)

    mean_deviation = np.mean(deviations_array)

    suspicion_score = 0.0

    if high_deviation_ratio > 0.5:
        suspicion_score += 0.5
    elif high_deviation_ratio > 0.3:
        suspicion_score += 0.3
    elif high_deviation_ratio > 0.15:
        suspicion_score += 0.15

    if mean_deviation > 15:
        suspicion_score += 0.3
    elif mean_deviation > 10:
        suspicion_score += 0.15

    return min(suspicion_score, 1.0)


def calculate_malpractice_suspicion(malpractice_flags, total_questions):
    """Calculate suspicion score based on flagged incidents (0-1)"""
    if not malpractice_flags or total_questions == 0:
        return 0.0

    flag_ratio = len(malpractice_flags) / total_questions

    avg_confidence = np.mean([f.get("confidence", 0.5) for f in malpractice_flags])

    suspicion_score = flag_ratio * 0.6 + avg_confidence * 0.4

    return min(suspicion_score, 1.0)


def calculate_combined_suspicion(session_data):
    """Calculate overall combined suspicion score"""
    timings = session_data.get("timings", [])
    deviations = session_data.get("deviations", [])
    malpractice_flags = session_data.get("malpractice_detected", [])
    total_questions = session_data.get("total_questions", len(timings))

    timing_score = calculate_timing_suspicion(timings)
    deviation_score = calculate_deviation_suspicion(deviations)
    malpractice_score = calculate_malpractice_suspicion(malpractice_flags, total_questions)

    combined_score = (
            timing_score * 0.4 +
            deviation_score * 0.3 +
            malpractice_score * 0.3
    )

    return {
        "combined_suspicion_score": round(combined_score, 3),
        "timing_suspicion": round(timing_score, 3),
        "deviation_suspicion": round(deviation_score, 3),
        "malpractice_suspicion": round(malpractice_score, 3),
        "risk_level": get_risk_level(combined_score)
    }


def get_risk_level(score):
    """Categorize risk level based on suspicion score"""
    if score >= 0.7:
        return "HIGH RISK"
    elif score >= 0.5:
        return "MODERATE RISK"
    elif score >= 0.3:
        return "LOW RISK"
    else:
        return "MINIMAL RISK"


def analyze_session_file(filename):
    """Analyze a single session file"""
    try:
        with open(filename, 'r') as f:
            session_data = json.load(f)

        analysis = calculate_combined_suspicion(session_data)

        analysis["session_file"] = filename
        analysis["timestamp"] = session_data.get("timestamp", "Unknown")
        analysis["total_questions"] = session_data.get("total_questions", 0)
        analysis["timings"] = session_data.get("timings", [])

        return analysis

    except Exception as e:
        print(f"❌ Error analyzing {filename}: {e}")
        return None


def main():
    print("🔍 Session Suspicion Score Analyzer\n")
    print("=" * 70)

    # Load trained model at startup
    load_trained_model()

    session_files = glob.glob("session_*.json")

    if not session_files:
        print("⚠️ No session files found. Run the exam first to generate session data.")
        return

    print(f"Found {len(session_files)} session file(s)\n")

    if len(session_files) == 1:
        target_file = session_files[0]
    else:
        print("Available sessions:")
        for i, f in enumerate(session_files, 1):
            print(f"  {i}. {f}")
        print(f"  {len(session_files) + 1}. Analyze all sessions")

        choice = input(f"\nSelect session (1-{len(session_files) + 1}): ").strip()

        try:
            choice_idx = int(choice) - 1
            if choice_idx == len(session_files):
                target_file = None
            else:
                target_file = session_files[choice_idx]
        except:
            print("Invalid choice. Analyzing most recent session.")
            target_file = max(session_files, key=os.path.getctime)

    if target_file:
        analysis = analyze_session_file(target_file)
        if analysis:
            display_analysis(analysis)
    else:
        for session_file in session_files:
            analysis = analyze_session_file(session_file)
            if analysis:
                display_analysis(analysis)
                print("\n" + "=" * 70 + "\n")


def display_analysis(analysis):
    """Display analysis results in formatted output"""
    print(f"📄 Session: {analysis['session_file']}")
    print(f"🕒 Timestamp: {analysis['timestamp']}")
    print(f"📝 Questions: {analysis['total_questions']}")
    print(f"\n⏱️  Response Times: {[f'{t:.1f}s' for t in analysis['timings']]}")
    print(f"\n{'=' * 70}")
    print(f"📊 SUSPICION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Timing Pattern Score:     {analysis['timing_suspicion']:.3f}")
    print(f"  Head Deviation Score:     {analysis['deviation_suspicion']:.3f}")
    print(f"  Malpractice Flag Score:   {analysis['malpractice_suspicion']:.3f}")
    print(f"{'-' * 70}")
    print(f"  🎯 COMBINED SCORE:         {analysis['combined_suspicion_score']:.3f}")
    print(f"  ⚠️  RISK LEVEL:             {analysis['risk_level']}")
    print(f"{'=' * 70}")

    print(f"\n💡 Interpretation:")
    score = analysis['combined_suspicion_score']
    if score >= 0.7:
        print("   Strong indicators of potential malpractice detected.")
        print("   Recommend manual review of session footage.")
    elif score >= 0.5:
        print("   Moderate suspicious patterns detected.")
        print("   Consider flagging for secondary review.")
    elif score >= 0.3:
        print("   Minor anomalies detected but likely within normal range.")
    else:
        print("   Behavior appears normal with minimal concerns.")


if __name__ == "__main__":
    main()
