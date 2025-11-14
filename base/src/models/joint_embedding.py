import torch
import torch.nn as nn
from typing import Optional, Tuple


class JointEmbeddingModel(nn.Module):
    """
    Joint audio-text embedding model with contrastive learning.
    Maps audio and text to a shared embedding space.
    """
    
    def __init__(
        self,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        embedding_dim: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        super().__init__()
        
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim
        
        # Learnable temperature for contrastive loss (like CLIP)
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones([]) * temperature)
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
    
    def encode_audio(
        self,
        audio_input: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode audio to embedding space.
        
        Args:
            audio_input: Audio tensor [batch_size, channels, time] or [batch_size, time]
            normalize: Whether to L2 normalize
        
        Returns:
            audio_embeddings: [batch_size, embedding_dim]
        """
        return self.audio_encoder(audio_input, normalize=normalize)
    
    def encode_text(
        self,
        text_input: torch.Tensor = None,
        texts: list = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text to embedding space.
        
        Args:
            text_input: Tokenized text [batch_size, seq_len]
            texts: Raw text strings
            normalize: Whether to L2 normalize
        
        Returns:
            text_embeddings: [batch_size, embedding_dim]
        """
        if texts is not None:
            return self.text_encoder(texts=texts, normalize=normalize)
        else:
            return self.text_encoder(input_ids=text_input, normalize=normalize)
    
    def forward(
        self,
        audio_input: torch.Tensor,
        text_input: Optional[torch.Tensor] = None,
        texts: Optional[list] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode both modalities.
        
        Args:
            audio_input: Audio tensor
            text_input: Tokenized text (optional if texts provided)
            texts: Raw text strings (optional if text_input provided)
            return_embeddings: If True, return embeddings; else return similarity matrix
        
        Returns:
            audio_embeddings: [batch_size, embedding_dim]
            text_embeddings: [batch_size, embedding_dim]
            OR
            similarity_matrix: [batch_size, batch_size] if return_embeddings=False
        """
        # Encode both modalities
        audio_embeddings = self.encode_audio(audio_input, normalize=True)
        text_embeddings = self.encode_text(text_input, texts, normalize=True)
        
        if return_embeddings:
            return audio_embeddings, text_embeddings
        
        # Compute similarity matrix (for contrastive loss)
        # similarity[i, j] = cosine_sim(audio[i], text[j])
        similarity = self.compute_similarity(audio_embeddings, text_embeddings)
        
        return similarity
    
    def compute_similarity(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scaled cosine similarity matrix.
        
        Args:
            audio_embeddings: [batch_size, embedding_dim] (L2 normalized)
            text_embeddings: [batch_size, embedding_dim] (L2 normalized)
        
        Returns:
            similarity: [batch_size, batch_size]
        """
        # Cosine similarity (since embeddings are normalized, this is just dot product)
        similarity = audio_embeddings @ text_embeddings.T
        
        # Scale by temperature (inverse)
        similarity = similarity / self.temperature
        
        return similarity
    
    def get_text_audio_similarity(
        self,
        text_query: str,
        audio_inputs: torch.Tensor,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k audio samples for a text query.
        
        Args:
            text_query: Text query string
            audio_inputs: Audio tensors [num_audios, ...]
            k: Number of top results
        
        Returns:
            similarities: Top-k similarity scores
            indices: Top-k audio indices
        """
        self.eval()
        with torch.no_grad():
            # Encode text query
            text_emb = self.encode_text(texts=[text_query])  # [1, embedding_dim]
            
            # Encode all audio
            audio_embs = self.encode_audio(audio_inputs)  # [num_audios, embedding_dim]
            
            # Compute similarities
            similarities = (text_emb @ audio_embs.T).squeeze(0)  # [num_audios]
            
            # Get top-k
            top_k_similarities, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
        
        return top_k_similarities, top_k_indices
    
    def get_audio_text_similarity(
        self,
        audio_input: torch.Tensor,
        text_candidates: list,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k text descriptions for an audio query.
        
        Args:
            audio_input: Audio tensor
            text_candidates: List of text strings
            k: Number of top results
        
        Returns:
            similarities: Top-k similarity scores
            indices: Top-k text indices
        """
        self.eval()
        with torch.no_grad():
            # Encode audio query
            audio_emb = self.encode_audio(audio_input.unsqueeze(0))  # [1, embedding_dim]
            
            # Encode all texts
            text_embs = self.encode_text(texts=text_candidates)  # [num_texts, embedding_dim]
            
            # Compute similarities
            similarities = (audio_emb @ text_embs.T).squeeze(0)  # [num_texts]
            
            # Get top-k
            top_k_similarities, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
        
        return top_k_similarities, top_k_indices
