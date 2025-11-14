import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Union, List


class TextEncoder(nn.Module):
    """
    Flexible text encoder with projection head and multiple pooling strategies.
    Can work with any HuggingFace transformer or custom architectures.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        embedding_dim: int = 512,
        pooling_strategy: str = 'mean',  # 'mean', 'cls', 'max', 'attention'
        freeze_backbone: bool = False,
        use_projection: bool = True,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get backbone hidden size
        self.hidden_size = self.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Attention pooling (if using attention strategy)
        if pooling_strategy == 'attention':
            self.attention_weights = nn.Linear(self.hidden_size, 1)
        
        # Projection head to map to shared embedding space
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, embedding_dim)
            )
        else:
            # Simple linear projection if embedding_dim differs
            if self.hidden_size != embedding_dim:
                self.projection = nn.Linear(self.hidden_size, embedding_dim)
            else:
                self.projection = nn.Identity()
        
        # Layer normalization for final embeddings
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply pooling strategy to sequence of hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            pooled: [batch_size, hidden_size]
        """
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling (ignore padding tokens)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == 'max':
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask_expanded == 0] = -1e9  # Set padding to very negative
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_strategy == 'attention':
            # Attention-based pooling
            attention_scores = self.attention_weights(hidden_states).squeeze(-1)  # [batch, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
            return torch.sum(hidden_states * attention_weights, dim=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = 'pt'
    ) -> dict:
        """
        Tokenize input texts.
        
        Args:
            texts: Single text or list of texts
            return_tensors: 'pt' for PyTorch tensors
        
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            texts: Raw text strings (will be tokenized if provided)
            normalize: Whether to L2 normalize output embeddings
        
        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # If raw texts provided, tokenize them
        if texts is not None:
            encoded = self.tokenize(texts)
            input_ids = encoded['input_ids'].to(self.backbone.device)
            attention_mask = encoded['attention_mask'].to(self.backbone.device)
        
        # Get hidden states from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply pooling
        pooled = self.pool_embeddings(hidden_states, attention_mask)
        
        # Apply projection head
        embeddings = self.projection(pooled)
        
        # Normalize output
        embeddings = self.output_norm(embeddings)
        
        # L2 normalization for cosine similarity
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Convenience method to encode text(s) with batching support.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            device: Device to use (None = use model's device)
        
        Returns:
            embeddings: [num_texts, embedding_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = self.forward(texts=batch_texts, normalize=normalize)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_config(self) -> dict:
        """Return model configuration for saving/loading."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'pooling_strategy': self.pooling_strategy,
            'max_length': self.max_length,
            'hidden_size': self.hidden_size
        }


# Alternative: Custom text encoder without transformers
# class CustomTextEncoder(nn.Module):
#     """
#     Lightweight custom text encoder using LSTM/GRU or CNN.
#     Good for faster inference or when you don't need transformer power.
#     """
    
#     def __init__(
#         self,
#         vocab_size: int = 30000,
#         embedding_dim: int = 512,
#         hidden_dim: int = 256,
#         num_layers: int = 2,
#         encoder_type: str = 'lstm',  # 'lstm', 'gru', 'cnn'
#         dropout: float = 0.3,
#         bidirectional: bool = True
#     ):
#         super().__init__()
        
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.encoder_type = encoder_type
        
#         # Word embeddings
#         self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
#         # Encoder architecture
#         if encoder_type == 'lstm':
#             self.encoder = nn.LSTM(
#                 hidden_dim,
#                 hidden_dim,
#                 num_layers=num_layers,
#                 batch_first=True,
#                 dropout=dropout if num_layers > 1 else 0,
#                 bidirectional=bidirectional
#             )
#             encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        
#         elif encoder_type == 'gru':
#             self.encoder = nn.GRU(
#                 hidden_dim,
#                 hidden_dim,
#                 num_layers=num_layers,
#                 batch_first=True,
#                 dropout=dropout if num_layers > 1 else 0,
#                 bidirectional=bidirectional
#             )
#             encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        
#         elif encoder_type == 'cnn':
#             # 1D CNN for text
#             self.encoder = nn.Sequential(
#                 nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2),
#                 nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.AdaptiveMaxPool1d(1)
#             )
#             encoder_output_dim = hidden_dim * 4
        
#         # Projection to embedding space
#         self.projection = nn.Sequential(
#             nn.Linear(encoder_output_dim, embedding_dim),
#             nn.LayerNorm(embedding_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(embedding_dim, embedding_dim)
#         )
    
#     def forward(self, input_ids: torch.Tensor, normalize: bool = True) -> torch.Tensor:
#         # Embed tokens
#         x = self.embedding(input_ids)  # [batch, seq_len, hidden_dim]
        
#         if self.encoder_type in ['lstm', 'gru']:
#             # RNN encoding
#             output, _ = self.encoder(x)
#             # Take last hidden state (or mean pool)
#             embeddings = output[:, -1, :]
        
#         elif self.encoder_type == 'cnn':
#             # CNN encoding
#             x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
#             x = self.encoder(x)
#             embeddings = x.squeeze(-1)
        
#         # Project to embedding space
#         embeddings = self.projection(embeddings)
        
#         if normalize:
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
#         return embeddings