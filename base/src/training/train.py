import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

from models.audio_encoder import Wav2Vec2AudioEncoder
from models.text_encoder import TextEncoder
from models.joint_embedding import JointEmbeddingModel
from data.dataset import AudioTextDataset, AudioTextCollator
from training.loss import ContrastiveLoss
from training.config import Config


def train_model(config: Config, device: torch.device, checkpoint_path: str = None):
    """
    Main training function.
    
    Args:
        config: Configuration object
        device: Device to train on
        checkpoint_path: Path to resume from checkpoint
    """
    # Create checkpoint directory
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    print("Initializing models...")
    audio_encoder = Wav2Vec2AudioEncoder(
        model_name="facebook/wav2vec2-base",
        embedding_dim=config.model.audio_embedding_dim,
        dropout=config.model.dropout
    ).to(device)
    
    text_encoder = TextEncoder(
        model_name=config.model.text_model_name,
        embedding_dim=config.model.text_embedding_dim,
        dropout=config.model.dropout
    ).to(device)
    
    joint_model = JointEmbeddingModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        embedding_dim=config.model.audio_embedding_dim,
        temperature=config.model.temperature,
        learnable_temperature=config.model.learnable_temperature
    ).to(device)
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Initialize datasets
    print("\nLoading datasets...")
    
    # Check if data files exist
    train_path = Path(config.paths.train_data_path)
    val_path = Path(config.paths.val_data_path)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    # Get audio directory (assuming it's in data/audio or same dir as CSV)
    audio_dir = train_path.parent / "audio"
    if not audio_dir.exists():
        audio_dir = None
        print("Warning: No 'audio' directory found. Using absolute paths from CSV.")
    
    train_dataset = AudioTextDataset(
        data_path=str(train_path),
        audio_dir=str(audio_dir) if audio_dir else None,
        sample_rate=16000,
        max_audio_length=160000,
        cache_audio=False
    )
    
    val_dataset = AudioTextDataset(
        data_path=str(val_path),
        audio_dir=str(audio_dir) if audio_dir else None,
        sample_rate=16000,
        max_audio_length=160000,
        cache_audio=False
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create collator
    collator = AudioTextCollator()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator
    )
    
    # Initialize loss and optimizer
    criterion = ContrastiveLoss(temperature=config.model.temperature)
    
    optimizer = torch.optim.AdamW(
        joint_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs
    )
    
    # Mixed precision training
    scaler = GradScaler() if config.training.mixed_precision else None
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            joint_model, train_loader, criterion, optimizer, 
            device, scaler, config, epoch
        )
        
        # Validate
        val_loss = validate_epoch(joint_model, val_loader, criterion, device, config)
        
        # Step scheduler
        scheduler.step()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Save checkpoint
        if (epoch + 1) % config.logging.save_model_interval == 0:
            checkpoint_path = Path(config.paths.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(joint_model, optimizer, epoch, train_loss, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(config.paths.checkpoint_dir) / "best_model.pth"
            save_checkpoint(joint_model, optimizer, epoch, train_loss, val_loss, best_path)
            print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
    
    print(f"\n{'='*60}")
    print("✓ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    
    for batch_idx, (audio, text) in enumerate(pbar):
        audio = audio.to(device)
        # text is a list of strings
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                # Forward pass
                audio_emb, text_emb = model(audio, texts=text, return_embeddings=True)
                loss = criterion(audio_emb, text_emb)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            audio_emb, text_emb = model(audio, texts=text, return_embeddings=True)
            loss = criterion(audio_emb, text_emb)
            loss.backward()
            
            if config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, config):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for audio, text in pbar:
            audio = audio.to(device)
            
            audio_emb, text_emb = model(audio, texts=text, return_embeddings=True)
            loss = criterion(audio_emb, text_emb)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, path)
    print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    from training.config import Config
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    train_model(config, device)