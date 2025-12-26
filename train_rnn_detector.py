import json
import numpy as np
import glob
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def load_all_session_data():
    """Load all real and synthetic session data"""
    all_data = []
    all_labels = []

    # Load original sessions (if they have labels)
    session_files = glob.glob("session_*.json")
    for file in session_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if 'label' in data:  # Only use if manually labeled
                    timings = data.get('timings', [])
                    label = data.get('label', 0)
                    if len(timings) > 0:
                        all_data.append(timings)
                        all_labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    # Load synthetic data
    synthetic_files = glob.glob("synthetic_data/synthetic_*.json")
    for file in synthetic_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                timings = data.get('timings', [])
                label = data.get('label', 0)
                if len(timings) > 0:
                    all_data.append(timings)
                    all_labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    # Load original labeled files
    if os.path.exists('cheating.json'):
        with open('cheating.json', 'r') as f:
            data = json.load(f)
            timings = data.get('timings', [])
            if len(timings) > 0:
                all_data.append(timings)
                all_labels.append(1)

    if os.path.exists('no_cheating.json'):
        with open('no_cheating.json', 'r') as f:
            data = json.load(f)
            timings = data.get('timings', [])
            if len(timings) > 0:
                all_data.append(timings)
                all_labels.append(0)

    return all_data, all_labels


def prepare_rnn_features(timings_list):
    """Convert timing sequences into RNN-ready features with padding"""
    max_length = max(len(t) for t in timings_list)

    features = []
    for timings in timings_list:
        seq_features = []
        cumulative = 0

        for i, t in enumerate(timings):
            cumulative += t
            time_diff = t - timings[i - 1] if i > 0 else 0
            seq_features.append([t, cumulative, time_diff])

        # Pad sequences to max_length
        while len(seq_features) < max_length:
            seq_features.append([0, 0, 0])

        features.append(seq_features)

    features = np.array(features, dtype=np.float32)

    # Normalize
    mean = np.mean(features, axis=(0, 1))
    std = np.std(features, axis=(0, 1)) + 1e-6
    features = (features - mean) / std

    return features, max_length, mean, std


def build_rnn_model(sequence_length, n_features=3):
    """Build LSTM-based RNN for cheating detection"""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.Masking(mask_value=0.0),  # Ignore padded values
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rnn_training_history.png', dpi=150)
    print("📊 Saved training visualization: rnn_training_history.png")


def main():
    print("🧠 RNN Cheating Detector Training Script\n")
    print("=" * 70)

    # Load data
    print("\n📂 Loading training data...")
    timings_list, labels = load_all_session_data()

    if len(timings_list) == 0:
        print("❌ No training data found!")
        print("   Please run 'generate_synthetic_data.py' first or ensure")
        print("   'cheating.json' and 'no_cheating.json' exist.")
        return

    print(f"✅ Loaded {len(timings_list)} sessions")
    print(f"   Cheating: {sum(labels)}")
    print(f"   Non-cheating: {len(labels) - sum(labels)}")

    # Prepare features
    print(f"\n🔧 Preparing RNN features...")
    X, max_length, mean, std = prepare_rnn_features(timings_list)
    y = np.array(labels)

    print(f"✅ Feature shape: {X.shape}")
    print(f"✅ Max sequence length: {max_length}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")

    # Build model
    print(f"\n🏗️ Building RNN model...")
    model = build_rnn_model(max_length)
    model.summary()

    # Train model
    print(f"\n🚀 Training model...")
    print("=" * 70)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    print(f"\n📈 Final Evaluation:")
    print("=" * 70)
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Save model and normalization parameters
    model.save('rnn_cheating_detector.h5')

    np.savez('rnn_normalization.npz',
             mean=mean,
             std=std,
             max_length=max_length)

    print(f"\n💾 Saved model and parameters:")
    print(f"   - rnn_cheating_detector.h5")
    print(f"   - rnn_normalization.npz")

    # Plot training history
    plot_training_history(history)

    print(f"\n{'=' * 70}")
    print("✅ Training complete!")
    print(f"{'=' * 70}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review training metrics in 'rnn_training_history.png'")
    print(f"   2. Use 'rnn_cheating_detector.h5' in your proctoring system")
    print(f"   3. The model is ready for real-time detection!")


if __name__ == "__main__":
    main()
