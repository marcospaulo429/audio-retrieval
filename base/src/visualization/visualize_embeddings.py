"""
Visualize embeddings from a trained model in TensorBoard.
Usage: python src/visualization/visualize_embeddings.py --checkpoint checkpoints/best_model.pth
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import argparse
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_model_and_processor, load_dataset_splits
from models.joint_embedding import JointEmbeddingModel
from models.text_encoder import TextEncoder
from data.dataset import AudioTextDataset, AudioTextCollator


def visualize_embeddings(checkpoint_path, dataset_path, config_path='configs/config.yaml', output_dir='embeddings_viz', n_samples=500):
    """
    Extract and visualize embeddings from a trained model.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        dataset_path: Path to dataset
        config_path: Path to config YAML file
        output_dir: Directory to save TensorBoard logs
        n_samples: Number of samples to visualize (more = slower but better)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check what's in the checkpoint
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Load config from YAML file instead
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')  # Use CPU for visualization
    
    print("Initializing models...")
    
    # Load audio encoder
    audio_encoder, audio_processor, target_sample_rate, model_input_key = load_model_and_processor(
        encoder_name=config['audio_encoder']['name'],
        model_name_or_path=config['audio_encoder']['model_name'],
        device=device,
        embedding_dim=config['audio_encoder']['embedding_dim'],
        freeze=True
    )
    
    # Load text encoder
    text_encoder = TextEncoder(
        model_name=config['text_encoder']['model_name'],
        embedding_dim=config['text_encoder']['embedding_dim'],
        dropout=config['text_encoder']['dropout']
    ).to(device)
    
    # Create joint model
    model = JointEmbeddingModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        temperature=config['model'].get('temperature', 0.07),
        learnable_temperature=config['model'].get('learnable_temperature', False)
    ).to(device)
    
    # Load trained weights
    print("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best loss: {checkpoint.get('best_loss', 'unknown')}")
    
    print("Loading dataset...")
    
    # Load dataset
    train_df, test_df = load_dataset_splits(
        dataset_name=config['dataset']['name'],
        dataset_path=dataset_path
    )
    
    # Rename columns
    test_df_renamed = test_df.copy()
    if 'file_path' in test_df.columns:
        test_df_renamed = test_df_renamed.rename(columns={'file_path': 'audio_path'})
    if 'caption' in test_df.columns:
        test_df_renamed = test_df_renamed.rename(columns={'caption': 'text'})
    
    # Take subset for visualization
    test_df_renamed = test_df_renamed.head(n_samples)
    
    # Create dataset
    test_dataset = AudioTextDataset(
        data_path=test_df_renamed,
        audio_dir=None,
        sample_rate=target_sample_rate,
        max_audio_length=int(config['dataset']['max_audio_length'] * target_sample_rate),
        audio_transform=None,
        cache_audio=False
    )
    
    # Extract audio feature extractor
    if hasattr(audio_processor, 'feature_extractor'):
        audio_feature_extractor = audio_processor.feature_extractor
    else:
        audio_feature_extractor = audio_processor
    
    # Create collator
    collator = AudioTextCollator(
        audio_processor=audio_feature_extractor,
        sample_rate=target_sample_rate
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    print(f"Extracting embeddings from {len(test_dataset)} samples...")
    
    # Extract embeddings
    audio_embeddings = []
    text_embeddings = []
    text_labels = []
    
    with torch.no_grad():
        for batch_idx, (audio, text) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...", end='\r')
            
            audio = audio.to(device)
            
            # Get embeddings
            audio_emb, text_emb = model(audio, texts=text, return_embeddings=True)
            
            audio_embeddings.append(audio_emb.cpu())
            text_embeddings.append(text_emb.cpu())
            
            # Use first 30 chars of each caption as label
            text_labels.extend([t[:30] + '...' if len(t) > 30 else t for t in text])
    
    print("\nConcatenating embeddings...")
    
    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    print(f"Audio embeddings shape: {audio_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Create TensorBoard writer
    print(f"Writing to TensorBoard in {output_dir}...")
    writer = SummaryWriter(log_dir=output_dir)
    
    # Add embeddings to TensorBoard
    writer.add_embedding(
        audio_embeddings,
        metadata=text_labels,
        tag='audio_embeddings'
    )
    
    writer.add_embedding(
        text_embeddings,
        metadata=text_labels,
        tag='text_embeddings'
    )
    
    writer.close()
    
    print(f"\n✅ Done! Embeddings saved to {output_dir}")
    print(f"\nTo visualize, run:")
    print(f"  docker exec -d audio-retrieval-app tensorboard --logdir {output_dir} --host 0.0.0.0 --port 6006")
    print(f"  Then open: http://localhost:6006")
    print(f"\nIn TensorBoard, go to the 'PROJECTOR' tab to see the 3D visualization!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize embeddings from trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset_path', type=str, default='/app/data/clotho', help='Path to dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='embeddings_viz', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    visualize_embeddings(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        config_path=args.config,
        output_dir=args.output_dir,
        n_samples=args.n_samples
    )