import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os


class TimingGAN:
    def __init__(self, sequence_length=5, latent_dim=32):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        """Generator: noise -> synthetic timing sequences"""
        model = keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.sequence_length, activation='linear')  # Output: timing sequence
        ], name='generator')
        return model

    def build_discriminator(self):
        """Discriminator: timing sequence -> real/fake"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Output: probability of being real
        ], name='discriminator')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        """Combined GAN model"""
        self.discriminator.trainable = False
        gan_input = keras.Input(shape=(self.latent_dim,))
        generated_sequence = self.generator(gan_input)
        gan_output = self.discriminator(generated_sequence)
        model = keras.Model(gan_input, gan_output, name='gan')
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, real_data, epochs=2000, batch_size=16):
        """Train the GAN on real data"""
        real_data = np.array(real_data)

        # Normalize data
        self.mean = np.mean(real_data)
        self.std = np.std(real_data) + 1e-6
        real_data_normalized = (real_data - self.mean) / self.std

        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, real_data_normalized.shape[0], batch_size)
            real_sequences = real_data_normalized[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_sequences = self.generator.predict(noise, verbose=0)

            d_loss_real = self.discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if epoch % 200 == 0:
                print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

        return d_losses, g_losses

    def generate_samples(self, n_samples):
        """Generate synthetic timing sequences"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        synthetic_normalized = self.generator.predict(noise, verbose=0)

        # Denormalize
        synthetic = synthetic_normalized * self.std + self.mean

        # Ensure positive times
        synthetic = np.abs(synthetic)

        return synthetic


def load_training_data():
    """Load real cheating and non-cheating data"""
    try:
        with open('cheating.json', 'r') as f:
            cheating_data = json.load(f)
        with open('no_cheating.json', 'r') as f:
            no_cheating_data = json.load(f)

        cheating_timings = cheating_data.get('timings', [])
        no_cheating_timings = no_cheating_data.get('timings', [])

        return cheating_timings, no_cheating_timings

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure 'cheating.json' and 'no_cheating.json' exist in the current directory.")
        return None, None


def save_synthetic_data(synthetic_cheating, synthetic_no_cheating):
    """Save synthetic data to files"""
    os.makedirs('synthetic_data', exist_ok=True)

    for i, seq in enumerate(synthetic_cheating):
        session_data = {
            "timestamp": f"synthetic_cheating_{i}",
            "timings": seq.tolist(),
            "label": 1,  # 1 = cheating
            "synthetic": True
        }
        filename = f"synthetic_data/synthetic_cheating_{i}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=4)

    for i, seq in enumerate(synthetic_no_cheating):
        session_data = {
            "timestamp": f"synthetic_no_cheating_{i}",
            "timings": seq.tolist(),
            "label": 0,  # 0 = no cheating
            "synthetic": True
        }
        filename = f"synthetic_data/synthetic_no_cheating_{i}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=4)

    print(f"\n✅ Generated {len(synthetic_cheating)} cheating samples")
    print(f"✅ Generated {len(synthetic_no_cheating)} non-cheating samples")
    print(f"📁 Saved to 'synthetic_data/' folder")


def plot_comparison(real_data, synthetic_data, title):
    """Plot comparison between real and synthetic data"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f'Real {title} Data')
    for seq in real_data[:5]:
        plt.plot(seq, marker='o', alpha=0.6)
    plt.xlabel('Question Number')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title(f'Synthetic {title} Data')
    for seq in synthetic_data[:5]:
        plt.plot(seq, marker='o', alpha=0.6)
    plt.xlabel('Question Number')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'gan_{title.lower().replace(" ", "_")}_comparison.png', dpi=150)
    print(f"📊 Saved visualization: gan_{title.lower().replace(' ', '_')}_comparison.png")


def main():
    print("🎲 GAN-Based Synthetic Exam Data Generator\n")
    print("=" * 70)

    # Load real data
    print("\n📂 Loading real session data...")
    cheating_timings, no_cheating_timings = load_training_data()

    if cheating_timings is None or no_cheating_timings is None:
        return

    print(f"✅ Loaded {len(cheating_timings)} cheating timings")
    print(f"✅ Loaded {len(no_cheating_timings)} non-cheating timings")

    # Prepare data (pad/trim to same length)
    max_len = max(len(cheating_timings), len(no_cheating_timings))

    cheating_array = np.array(cheating_timings + [cheating_timings[-1]] * (max_len - len(cheating_timings))).reshape(1,
                                                                                                                     -1)
    no_cheating_array = np.array(
        no_cheating_timings + [no_cheating_timings[-1]] * (max_len - len(no_cheating_timings))).reshape(1, -1)

    sequence_length = cheating_array.shape[1]

    # Train GAN for cheating behavior
    print(f"\n🔧 Training GAN for cheating behavior...")
    print("=" * 70)
    gan_cheating = TimingGAN(sequence_length=sequence_length, latent_dim=32)
    gan_cheating.train(cheating_array, epochs=2000, batch_size=1)

    # Train GAN for non-cheating behavior
    print(f"\n🔧 Training GAN for non-cheating behavior...")
    print("=" * 70)
    gan_no_cheating = TimingGAN(sequence_length=sequence_length, latent_dim=32)
    gan_no_cheating.train(no_cheating_array, epochs=2000, batch_size=1)

    # Generate synthetic samples
    print(f"\n🎯 Generating synthetic data...")
    n_synthetic = 100  # Generate 100 samples of each type

    synthetic_cheating = gan_cheating.generate_samples(n_synthetic)
    synthetic_no_cheating = gan_no_cheating.generate_samples(n_synthetic)

    # Save synthetic data
    save_synthetic_data(synthetic_cheating, synthetic_no_cheating)

    # Plot comparisons
    print(f"\n📊 Creating visualizations...")
    plot_comparison([cheating_timings], synthetic_cheating, "Cheating")
    plot_comparison([no_cheating_timings], synthetic_no_cheating, "Non-Cheating")

    # Save GAN models
    gan_cheating.generator.save('gan_cheating_generator.h5')
    gan_no_cheating.generator.save('gan_no_cheating_generator.h5')
    print(f"\n💾 Saved GAN models:")
    print(f"   - gan_cheating_generator.h5")
    print(f"   - gan_no_cheating_generator.h5")

    print(f"\n{'=' * 70}")
    print("✅ Data generation complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
