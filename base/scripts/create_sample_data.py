import pandas as pd
from pathlib import Path
import torch
import numpy as np

# Use soundfile instead of torchaudio for better compatibility
try:
    import soundfile as sf
    USE_SOUNDFILE = True
except ImportError:
    import torchaudio
    USE_SOUNDFILE = False
    print("Warning: soundfile not found, using torchaudio (may have issues)")

# Create directories
data_dir = Path('data')
audio_dir = data_dir / 'audio'
audio_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating sample data in: {data_dir}")

# Generate dummy audio files (1 second each at 16kHz)
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)

print("Generating audio files...")
for i in range(1, 6):
    # Generate random audio
    waveform = torch.randn(1, num_samples) * 0.1
    
    # Save
    audio_path = audio_dir / f'sample_{i:03d}.wav'
    
    if USE_SOUNDFILE:
        # Use soundfile (more reliable)
        sf.write(str(audio_path), waveform.squeeze().numpy(), sample_rate)
    else:
        # Use torchaudio
        torchaudio.save(str(audio_path), waveform, sample_rate)
    
    print(f"  ✓ Created {audio_path.name}")

# Create metadata
train_data = pd.DataFrame({
    'audio_path': [f'sample_{i:03d}.wav' for i in range(1, 4)],
    'text': [
        'A dog barking loudly',
        'Ocean waves crashing',
        'Piano melody playing'
    ]
})

val_data = pd.DataFrame({
    'audio_path': [f'sample_{i:03d}.wav' for i in range(4, 6)],
    'text': [
        'Car horn honking',
        'Rain falling softly'
    ]
})

# Save CSVs
train_csv = data_dir / 'train_data.csv'
val_csv = data_dir / 'val_data.csv'

train_data.to_csv(train_csv, index=False)
val_data.to_csv(val_csv, index=False)

print("\n✓ Sample data created successfully!")
print(f"  Train CSV: {train_csv} ({len(train_data)} samples)")
print(f"  Val CSV: {val_csv} ({len(val_data)} samples)")
print(f"  Audio files: {audio_dir} (5 files)")