import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.audio_encoder import Wav2Vec2AudioEncoder
from models.text_encoder import TextEncoder
from models.joint_embedding import JointEmbeddingModel
from data.dataset import AudioTextDataset, AudioTextCollator
from training.loss import ContrastiveLoss
from utils import load_model_and_processor, load_dataset_splits


def train_model(config, device, checkpoint_path=None):
    """Main training function.
    
    Args:
        config: Configuration dictionary
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_path: Optional path to checkpoint to resume from
    """
    print("Initializing models...")
    
    # Load audio encoder and processor
    audio_encoder, audio_processor, target_sample_rate, model_input_key = load_model_and_processor(
        encoder_name=config['audio_encoder']['name'],
        model_name_or_path=config['audio_encoder']['model_name'],
        device=device,
        embedding_dim=config['audio_encoder']['embedding_dim'],
        freeze=config['audio_encoder']['freeze']
    )
    
    # Extract just the audio feature extractor from CLAP processor
    if hasattr(audio_processor, 'feature_extractor'):
        audio_feature_extractor = audio_processor.feature_extractor
    else:
        audio_feature_extractor = audio_processor
    
    # Load text encoder (assuming BERT-based)
    from transformers import AutoTokenizer
    
    text_tokenizer = AutoTokenizer.from_pretrained(config['text_encoder']['model_name'])
    text_encoder = TextEncoder(
        model_name=config['text_encoder']['model_name'],
        embedding_dim=config['text_encoder']['embedding_dim'],
        dropout=config['text_encoder'].get('dropout', 0.1)
    ).to(device)
    
    joint_model = JointEmbeddingModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        embedding_dim=config['audio_encoder']['embedding_dim'],
        temperature=config.get('temperature', 0.07),
        learnable_temperature=config.get('learnable_temperature', False)
    ).to(device)
    
    # Load checkpoint if provided (use parameter, not config)
    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    print("Loading datasets...")
    
    # Load dataset using factory function
    train_df, test_df = load_dataset_splits(
        dataset_name=config['dataset']['name'],
        dataset_path=config['dataset'].get('path', None)
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Columns: {list(train_df.columns)}")
    
    # Determine if this is a captioning dataset
    is_caption_dataset = 'caption' in train_df.columns
    print(f"  Dataset type: {'Captioning' if is_caption_dataset else 'Classification'}")
    
    # Rename columns to match AudioTextDataset expectations
    train_df_renamed = train_df.copy()
    test_df_renamed = test_df.copy()
    
    if 'file_path' in train_df.columns:
        train_df_renamed = train_df_renamed.rename(columns={'file_path': 'audio_path'})
        test_df_renamed = test_df_renamed.rename(columns={'file_path': 'audio_path'})
    
    if 'caption' in train_df.columns:
        train_df_renamed = train_df_renamed.rename(columns={'caption': 'text'})
        test_df_renamed = test_df_renamed.rename(columns={'caption': 'text'})
    elif 'label' in train_df.columns:
        train_df_renamed = train_df_renamed.rename(columns={'label': 'text'})
        test_df_renamed = test_df_renamed.rename(columns={'label': 'text'})
    
    # Create datasets with correct arguments
    train_dataset = AudioTextDataset(
        data_path=train_df_renamed,
        audio_dir=None,  # Paths are absolute
        sample_rate=target_sample_rate,  # Use CLAP's expected sample rate (48000)
        max_audio_length=int(config['dataset']['max_audio_length'] * target_sample_rate),  # 10 sec * 48000 = 480000 samples
        audio_transform=None,
        cache_audio=False
    )
    
    test_dataset = AudioTextDataset(
        data_path=test_df_renamed,
        audio_dir=None,
        sample_rate=target_sample_rate,  # Use CLAP's expected sample rate (48000)
        max_audio_length=int(config['dataset']['max_audio_length'] * target_sample_rate),
        audio_transform=None,
        cache_audio=False
    )
    
    print(f"\nDatasets created:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create collator with audio feature extractor and correct sample rate
    collator = AudioTextCollator(
        audio_processor=audio_feature_extractor,
        sample_rate=target_sample_rate  # Pass the CLAP sample rate (48000)
    )
    
    # Create dataloaders
    num_workers = config['training'].get('num_workers', 4)
    # Set num_workers to 0 if on CPU to avoid multiprocessing issues
    if device == 'cpu':
        num_workers = 0
        print("Running on CPU - setting num_workers=0")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        collate_fn=collator,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        collate_fn=collator
    )
    
    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {config['training']['batch_size']}")
    
    # Initialize loss and optimizer
    criterion = ContrastiveLoss(temperature=config.get('temperature', 0.07))
    
    optimizer = torch.optim.AdamW(
        joint_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Mixed precision training
    use_amp = config['training'].get('mixed_precision', False) and device == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        print("Using mixed precision training")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
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
        save_interval = config.get('save_interval', 5)
        if (epoch + 1) % save_interval == 0:
            checkpoint_path_save = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(joint_model, optimizer, epoch, train_loss, val_loss, checkpoint_path_save)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pth"
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
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch - could be (audio, text) or dict
        if isinstance(batch, dict):
            audio = batch['audio'].to(device)
            text = batch['text']
        else:
            audio, text = batch
            audio = audio.to(device)
        
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
            gradient_clip = config['training'].get('gradient_clip', 1.0)
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            audio_emb, text_emb = model(audio, texts=text, return_embeddings=True)
            loss = criterion(audio_emb, text_emb)
            loss.backward()
            
            gradient_clip = config['training'].get('gradient_clip', 1.0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
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
        
        for batch in pbar:
            # Unpack batch
            if isinstance(batch, dict):
                audio = batch['audio'].to(device)
                text = batch['text']
            else:
                audio, text = batch
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
    parser = argparse.ArgumentParser(description="Train audio-text retrieval model")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name (overrides config): esc50, urbansound8k, gtzan, clotho, etc.')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to dataset directory (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use: cuda or cpu (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.dataset_name:
        config['dataset']['name'] = args.dataset_name
    if args.dataset_path:
        config['dataset']['path'] = args.dataset_path
    if args.device:
        config['device'] = args.device
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    train_model(config, device, checkpoint_path=args.checkpoint)