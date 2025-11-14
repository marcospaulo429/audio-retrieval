import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for audio-text pairs (CLIP-style).
    Maximizes similarity for matching pairs, minimizes for non-matching pairs.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional contrastive loss.

        Args:
            audio_embeddings: [batch_size, embedding_dim] (L2 normalized)
            text_embeddings: [batch_size, embedding_dim] (L2 normalized)

        Returns:
            loss: Scalar loss value
        """
        batch_size = audio_embeddings.shape[0]

        # Compute similarity matrix: [batch_size, batch_size]
        # similarity[i, j] = cosine_sim(audio[i], text[j])
        similarity = audio_embeddings @ text_embeddings.T / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=similarity.device)

        # Audio-to-text loss
        loss_audio_to_text = F.cross_entropy(similarity, labels)

        # Text-to-audio loss (transpose)
        loss_text_to_audio = F.cross_entropy(similarity.T, labels)

        # Average both directions
        loss = (loss_audio_to_text + loss_text_to_audio) / 2

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss: anchor should be closer to positive than to negative.
    Use this if you have triplet data (anchor, positive, negative).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negative: [batch_size, embedding_dim]

        Returns:
            loss: Scalar loss value
        """
        # Euclidean distance
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss (used in SimCLR, MoCo).
    Alternative to contrastive loss with explicit negative sampling.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = audio_embeddings.shape[0]

        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # Compute logits
        logits = audio_embeddings @ text_embeddings.T / self.temperature

        # Labels (diagonal is positive)
        labels = torch.arange(batch_size, device=logits.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss