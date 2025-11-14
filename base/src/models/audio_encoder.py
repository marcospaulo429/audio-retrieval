import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
import torchaudio
from typing import Optional


class Wav2Vec2AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec 2.0.
    Pre-trained on large-scale audio data, excellent for any audio task.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        embedding_dim: int = 512,
        freeze_feature_extractor: bool = False,
        freeze_encoder: bool = False,
        pooling_strategy: str = "mean",  # 'mean', 'max', 'attention'
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        
        # Load pre-trained Wav2Vec 2.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size  # Usually 768
        
        # Optionally freeze components
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        if freeze_encoder:
            for param in self.wav2vec2.encoder.parameters():
                param.requires_grad = False
        
        # Attention pooling
        if pooling_strategy == 'attention':
            self.attention_weights = nn.Linear(self.hidden_size, 1)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, embedding_dim)
        )
        
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def pool_features(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Apply pooling to sequence of features."""
        if self.pooling_strategy == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return hidden_states.mean(dim=1)
        
        elif self.pooling_strategy == 'max':
            return hidden_states.max(dim=1)[0]
        
        elif self.pooling_strategy == 'attention':
            attention_scores = self.attention_weights(hidden_states).squeeze(-1)
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            return torch.sum(hidden_states * attention_weights, dim=1)
        
        return hidden_states[:, 0, :]  # CLS token
    
    def forward(self, audio_input: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            audio_input: [batch_size, audio_length] or [batch_size, 1, audio_length]
            normalize: Whether to L2 normalize output
        
        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Ensure correct shape
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(1)
        
        # Extract features with Wav2Vec 2.0
        outputs = self.wav2vec2(audio_input)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Pool features
        pooled = self.pool_features(hidden_states)
        
        # Project to embedding space
        embeddings = self.projection(pooled)
        embeddings = self.output_norm(embeddings)
        
        # L2 normalize
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


## 2. **HuBERT** (Best for Speech Understanding)

class HuBERTAudioEncoder(nn.Module):
    """
    Audio encoder using HuBERT (Hidden Unit BERT).
    Better than Wav2Vec 2.0 for speech tasks.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        embedding_dim: int = 512,
        freeze_feature_extractor: bool = False,
        pooling_strategy: str = "mean",
        dropout: float = 0.1
    ):
        super().__init__()
        
        from transformers import HubertModel
        
        self.hubert = HubertModel.from_pretrained(model_name)
        self.hidden_size = self.hubert.config.hidden_size
        
        if freeze_feature_extractor:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
        
        self.pooling_strategy = pooling_strategy
        
        if pooling_strategy == 'attention':
            self.attention_weights = nn.Linear(self.hidden_size, 1)
        
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, embedding_dim)
        )
        
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, audio_input: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(1)
        
        outputs = self.hubert(audio_input)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling
        if self.pooling_strategy == 'mean':
            pooled = hidden_states.mean(dim=1)
        elif self.pooling_strategy == 'attention':
            attention_scores = self.attention_weights(hidden_states).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(hidden_states * attention_weights, dim=1)
        else:
            pooled = hidden_states[:, 0, :]
        
        embeddings = self.projection(pooled)
        embeddings = self.output_norm(embeddings)
        
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


## 3. **AST (Audio Spectrogram Transformer)** - Vision Transformer for Audio

class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer (AST).
    Treats spectrograms as images, uses ViT architecture.
    SOTA for audio classification tasks.
    """
    
    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        embedding_dim: int = 512,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        from transformers import ASTModel
        
        self.ast = ASTModel.from_pretrained(model_name)
        self.hidden_size = self.ast.config.hidden_size  # 768
        
        if freeze_backbone:
            for param in self.ast.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, embedding_dim)
        )
        
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, audio_input: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            audio_input: Raw waveform [batch, audio_length] or [batch, 1, audio_length]
        """
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(1)
        
        # AST expects [batch, audio_length]
        outputs = self.ast(audio_input)
        
        # Use [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]
        
        embeddings = self.projection(pooled)
        embeddings = self.output_norm(embeddings)
        
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


## 4. **Custom CNN + Transformer Hybrid** (Lightweight & Fast)

class CNNTransformerAudioEncoder(nn.Module):
    """
    Custom hybrid architecture: CNN for local features + Transformer for global context.
    Fast and efficient, good for production.
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 512,
        cnn_channels: list = [64, 128, 256, 512],
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Mel spectrogram extraction
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=n_mels
        )
        
        # CNN frontend (process spectrograms)
        layers = []
        in_channels = 1
        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate feature dimension after CNN
        self.cnn_output_dim = cnn_channels[-1]
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_output_dim,
            nhead=transformer_heads,
            dim_feedforward=self.cnn_output_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.cnn_output_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, audio_input: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            audio_input: [batch, audio_length] or [batch, 1, audio_length]
        """
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(1)
        
        # Extract mel spectrogram
        mel_spec = self.mel_spec(audio_input)  # [batch, n_mels, time]
        mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, n_mels, time]
        
        # CNN feature extraction
        cnn_features = self.cnn(mel_spec)  # [batch, channels, h, w]
        
        # Reshape for transformer: [batch, seq_len, features]
        batch, channels, h, w = cnn_features.shape
        cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(batch, h * w, channels)
        
        # Transformer encoding
        transformer_out = self.transformer(cnn_features)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Project to embedding space
        embeddings = self.projection(pooled)
        
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


## 5. **CLAP Audio Encoder** (Best for Audio-Text Retrieval!)

class CLAPAudioEncoder(nn.Module):
    """
    Use CLAP's audio encoder directly.
    Pre-trained on audio-text pairs - perfect for your use case!
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        embedding_dim: int = 512,
        freeze: bool = False
    ):
        super().__init__()
        
        from transformers import ClapModel
        
        self.clap = ClapModel.from_pretrained(model_name)
        self.audio_encoder = self.clap.audio_model
        self.hidden_size = 512  # CLAP default
        
        if freeze:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Optional projection if you want different embedding_dim
        if embedding_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, audio_input: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        embeddings = self.clap.get_audio_features(audio_input)
        embeddings = self.projection(embeddings)
        
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings