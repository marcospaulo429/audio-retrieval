import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union


class AudioTextDataset(Dataset):
    """
    PyTorch Dataset for audio-text pairs.
    Compatible with the training pipeline.
    """
    
    def __init__(
        self,
        data_path: Union[str, pd.DataFrame],
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_audio_length: int = 160000,  # 10 seconds at 16kHz
        audio_transform: Optional[callable] = None,
        cache_audio: bool = False
    ):
        """
        Args:
            data_path: Path to CSV/JSON with columns ['audio_path', 'text']
            audio_dir: Base directory for audio files
            sample_rate: Target sample rate
            max_audio_length: Max audio length in samples
            audio_transform: Optional augmentation
            cache_audio: Cache loaded audio in memory
        """
        super().__init__()
        
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.audio_transform = audio_transform
        self.cache_audio = cache_audio
        
        # Load metadata
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        elif isinstance(data_path, (str, Path)):
            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {data_path}")
            
            if data_path.suffix == '.csv':
                self.data = pd.read_csv(data_path)
            elif data_path.suffix == '.json':
                self.data = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported format: {data_path.suffix}")
        else:
            raise ValueError("data_path must be string, Path, or DataFrame")
        
        # Validate columns
        if 'audio_path' not in self.data.columns or 'text' not in self.data.columns:
            raise ValueError("Dataset must have 'audio_path' and 'text' columns")
        
        # Audio cache
        self.audio_cache = {} if cache_audio else None
        self._resampler = None
        
        print(f"Loaded {len(self.data)} audio-text pairs from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            audio: [audio_length] tensor
            text: string description
        """
        row = self.data.iloc[idx]
        
        # Load audio
        audio = self._load_audio(row['audio_path'], idx)
        
        # Get text
        text = str(row['text'])
        
        return audio, text
    
    def _load_audio(self, audio_path: str, idx: int) -> torch.Tensor:
        """Load and preprocess audio file."""
        
        # Check cache
        if self.audio_cache is not None and idx in self.audio_cache:
            return self.audio_cache[idx]
        
        # Resolve path
        if self.audio_dir:
            full_path = self.audio_dir / audio_path
        else:
            full_path = Path(audio_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Audio file not found: {full_path}")
        
        # Load audio
        waveform, orig_sr = torchaudio.load(full_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample
        if orig_sr != self.sample_rate:
            if self._resampler is None or self._resampler.orig_freq != orig_sr:
                self._resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr,
                    new_freq=self.sample_rate
                )
            waveform = self._resampler(waveform)
        
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        # Pad or trim
        if waveform.shape[0] > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        elif waveform.shape[0] < self.max_audio_length:
            padding = self.max_audio_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Augmentation
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        
        # Cache
        if self.audio_cache is not None:
            self.audio_cache[idx] = waveform
        
        return waveform


class AudioTextCollator:
    """
    Custom collator for DataLoader.
    Batches audio-text pairs properly.
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        """
        Args:
            batch: List of (audio, text) tuples
        
        Returns:
            audio_batch: [batch_size, audio_length]
            text_batch: List of strings
        """
        audios, texts = zip(*batch)
        
        # Stack audio tensors
        audio_batch = torch.stack(audios)
        
        # Keep texts as list (encoder will tokenize)
        text_batch = list(texts)
        
        return audio_batch, text_batch