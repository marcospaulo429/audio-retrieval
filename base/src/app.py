"""
Main application entry point for audio-text retrieval system.
Can be used for training, inference, or API serving.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch

from training.config import Config
from training.train import train_model
# from retrieval.audio_to_text import retrieve_text_from_audio
# from retrieval.text_to_audio import retrieve_audio_from_text


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Audio-Text Retrieval System')
    
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'inference', 'serve'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    # Inference arguments
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Text query for audio retrieval'
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Audio file path for text retrieval'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top results to return'
    )
    
    # API serving arguments
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API host address'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API port'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = setup_args()
    
    # Load configuration
    config = Config(args.config)
    config.display()
    
    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}\n")
    
    if args.mode == 'train':
        print("Starting training...")
        config_dict = {
            'audio_encoder': {
                'name': config.audio_encoder.name,
                'model_name': config.audio_encoder.model_name,
                'embedding_dim': config.audio_encoder.embedding_dim,
                'freeze': config.audio_encoder.freeze
            },
            'text_encoder': {
                'model_name': config.text_encoder.model_name,
                'embedding_dim': config.text_encoder.embedding_dim,
                'dropout': config.text_encoder.dropout
            },
            'dataset': {
                'name': config.dataset.name,
                'path': config.dataset.path,
                'max_audio_length': config.dataset.max_audio_length,
                'max_text_length': config.dataset.max_text_length
            },
            'training': {
                'batch_size': config.training.batch_size,
                'num_epochs': config.training.num_epochs,
                'learning_rate': config.training.learning_rate,
                'weight_decay': config.training.weight_decay,
                'gradient_clip': config.training.gradient_clip,
                'mixed_precision': config.training.mixed_precision,
                'num_workers': config.training.num_workers
            },
            'temperature': config.model.temperature,
            'learnable_temperature': config.model.learnable_temperature,
            'checkpoint_dir': config.paths.checkpoints
        }
        train_model(config_dict, device, checkpoint_path=args.checkpoint)
    
    elif args.mode == 'inference':
        print("Running inference...")
        
        if args.query and args.audio:
            print("Error: Provide either --query or --audio, not both")
            return
        
        if args.query:
            # Text to audio retrieval
            print(f"Query: {args.query}")
            print("Not implemented yet - create retrieval/text_to_audio.py")
        
        elif args.audio:
            # Audio to text retrieval
            print(f"Audio file: {args.audio}")
            print("Not implemented yet - create retrieval/audio_to_text.py")
        
        else:
            print("Error: Provide either --query or --audio for inference")
    
    elif args.mode == 'serve':
        print(f"Starting API server on {args.host}:{args.port}...")
        print("Not implemented yet - create api/server.py with FastAPI")


if __name__ == '__main__':
    main()