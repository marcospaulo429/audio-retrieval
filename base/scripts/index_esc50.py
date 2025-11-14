"""
ESC-50 Indexing Pipeline
Indexes all ESC-50 audio files into ChromaDB using CLAPAudioEncoder.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import librosa
import chromadb
from tqdm import tqdm
from transformers import ClapProcessor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.audio_encoder import CLAPAudioEncoder


def setup_environment():
    """Setup device, processor, and model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "laion/clap-htsat-unfused"
    
    # Load processor
    print(f"Loading processor: {model_name}")
    processor = ClapProcessor.from_pretrained(model_name)
    
    # Instantiate model
    print(f"Loading model: {model_name}")
    model = CLAPAudioEncoder(model_name=model_name, freeze=True).to(device)
    model.eval()
    
    return device, processor, model


def load_esc50_dataset(esc50_path):
    """
    Load ESC-50 dataset metadata.
    
    Args:
        esc50_path: Path to ESC-50 dataset directory
        
    Returns:
        DataFrame with filename and category mapping
    """
    meta_path = os.path.join(esc50_path, "meta", "esc50.csv")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {meta_path}\n"
            "Please download ESC-50 from: https://github.com/karolpiczak/ESC-50\n"
            "Expected structure:\n"
            "  ESC-50/\n"
            "    meta/\n"
            "      esc50.csv\n"
            "    audio/\n"
            "      *.wav"
        )
    
    df = pd.read_csv(meta_path)
    
    # Create mapping: filename -> category
    filename_to_category = dict(zip(df['filename'], df['category']))
    
    print(f"Loaded {len(filename_to_category)} audio file mappings")
    return filename_to_category, df


def get_indexing_files(df):
    """
    Get list of filenames for indexing (folds 1-4).
    
    Args:
        df: DataFrame with ESC-50 metadata
        
    Returns:
        List of filenames to index
    """
    # Indexing Set: Files where fold is 1, 2, 3, or 4
    indexing_df = df[df['fold'].isin([1, 2, 3, 4])]
    indexing_files = indexing_df['filename'].tolist()
    
    print(f"Indexing set: {len(indexing_files)} files (folds 1-4)")
    return indexing_files


def index_audio_files(
    esc50_path,
    device,
    processor,
    model,
    batch_size=100,
    collection_name="esc50",
    chromadb_path="./chromadb",
    distance_metric="cosine"
):
    """
    Index all ESC-50 audio files into ChromaDB.
    
    Args:
        esc50_path: Path to ESC-50 dataset directory
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        batch_size: Number of items to add to ChromaDB per batch
        collection_name: Name of ChromaDB collection
        chromadb_path: Path to persist ChromaDB data
        distance_metric: Distance metric ('cosine', 'l2', 'ip')
    """
    # Load dataset
    filename_to_category, df = load_esc50_dataset(esc50_path)
    
    # Get indexing files (folds 1-4)
    indexing_files = get_indexing_files(df)
    
    # ChromaDB setup with persistent storage
    print(f"Setting up ChromaDB (persistent storage at: {chromadb_path})...")
    print(f"Distance metric: {distance_metric}")
    client = chromadb.PersistentClient(path=chromadb_path)
    
    # Create or get collection with specified distance metric
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric}
    )
    
    # Check if collection already has data
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection '{collection_name}' already has {existing_count} items.")
        response = input("Do you want to clear it and re-index? (y/n): ")
        if response.lower() == 'y':
            client.delete_collection(name=collection_name)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
        else:
            print("Skipping indexing.")
            return
    
    audio_dir = os.path.join(esc50_path, "audio")
    
    print(f"Indexing {len(indexing_files)} audio files from folds 1-4")
    
    # Batch storage
    embeddings_batch = []
    ids_batch = []
    metadatas_batch = []
    
    # Indexing loop - only process files from indexing set (folds 1-4)
    with torch.no_grad():
        for idx, filename in enumerate(tqdm(indexing_files, desc="Indexing audio files")):
            audio_path = os.path.join(audio_dir, filename)
            
            try:
                # Load audio with librosa (CLAP expects 48kHz)
                audio_array, sr = librosa.load(audio_path, sr=48000)
                
                # Use processor to format the audio
                inputs = processor(
                    text=None,
                    audio=[audio_array],  # Use 'audio' instead of deprecated 'audios'
                    return_tensors="pt",
                    sampling_rate=48000
                ).to(device)
                
                # Get embedding
                embedding = model(inputs["input_features"])
                embedding_np = embedding.cpu().numpy()[0]  # Remove batch dimension
                
                # Get label from CSV
                category = filename_to_category.get(filename, "unknown")
                
                # Prepare metadata
                metadata = {
                    "label": category,
                    "filename": filename
                }
                
                # Add to batch
                embeddings_batch.append(embedding_np.tolist())
                ids_batch.append(filename)
                metadatas_batch.append(metadata)
                
                # Add to ChromaDB in batches
                if len(embeddings_batch) >= batch_size:
                    collection.add(
                        embeddings=embeddings_batch,
                        ids=ids_batch,
                        metadatas=metadatas_batch
                    )
                    embeddings_batch = []
                    ids_batch = []
                    metadatas_batch = []
            
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                continue
        
        # Add remaining items
        if len(embeddings_batch) > 0:
            collection.add(
                embeddings=embeddings_batch,
                ids=ids_batch,
                metadatas=metadatas_batch
            )
    
    final_count = collection.count()
    print(f"\nIndexing complete! Added {final_count} audio files to ChromaDB collection '{collection_name}'")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index ESC-50 dataset into ChromaDB")
    parser.add_argument(
        "--esc50_path",
        type=str,
        required=True,
        help="Path to ESC-50 dataset directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for ChromaDB insertion (default: 100)"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="esc50",
        help="ChromaDB collection name (default: esc50)"
    )
    parser.add_argument(
        "--chromadb_path",
        type=str,
        default="./chromadb",
        help="Path to persist ChromaDB data (default: ./chromadb)"
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=['cosine', 'ip', 'l2'],
        default='cosine',
        help="Distance metric for ChromaDB (default: cosine)"
    )
    
    args = parser.parse_args()
    
    # Generate collection name with distance metric
    collection_name = f"{args.collection_name}_{args.distance_metric}"
    print(f"Collection name: {collection_name}")
    
    # Setup environment
    device, processor, model = setup_environment()
    
    # Index audio files
    index_audio_files(
        esc50_path=args.esc50_path,
        device=device,
        processor=processor,
        model=model,
        batch_size=args.batch_size,
        collection_name=collection_name,
        chromadb_path=args.chromadb_path,
        distance_metric=args.distance_metric
    )


if __name__ == "__main__":
    main()

