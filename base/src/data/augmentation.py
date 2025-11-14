import numpy as np
import librosa

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return np.clip(augmented_audio, -1.0, 1.0)

def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate)

def pitch_shift(audio, sampling_rate, n_steps):
    return librosa.effects.pitch_shift(audio, sampling_rate, n_steps)

def shift_audio(audio, shift):
    return np.roll(audio, shift)