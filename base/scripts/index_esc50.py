"""
Generic Audio Dataset Indexing Pipeline
Indexes audio files into ChromaDB using various audio encoders.
"""

import os
import sys
from pathlib import Path
import torch
import librosa
import chromadb
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model_and_processor, load_dataset_splits


def index_audio_files(
    train_df,
    device,
    processor,
    model,
    target_sample_rate,
    model_input_key,
    batch_size=100,
    collection_name="dataset",
    chromadb_path="./chromadb",
    distance_metric="cosine"
):
    """
    Index audio files into ChromaDB.
    
    Args:
        train_df: DataFrame with 'file_path' and 'label' columns
        device: torch device
        processor: Processor instance (or None for custom_cnn)
        model: Audio encoder model instance
        target_sample_rate: Target sample rate for audio loading
        model_input_key: Key to use when passing input to model
        batch_size: Number of items to add to ChromaDB per batch
        collection_name: Name of ChromaDB collection
        chromadb_path: Path to persist ChromaDB data
        distance_metric: Distance metric ('cosine', 'l2', 'ip')
    """
    
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
    
    print(f"Indexing {len(train_df)} audio files")
    
    # Batch storage
    embeddings_batch = []
    ids_batch = []
    metadatas_batch = []
    
    # Indexing loop
    with torch.no_grad():
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Indexing audio files"):
            audio_path = row['file_path']
            label = row['label']
            filename = os.path.basename(audio_path)
            
            try:
                # Load audio with librosa at target sample rate
                audio_array, sr = librosa.load(audio_path, sr=target_sample_rate)
                
                # Handle chunking for GTZAN (or other datasets with chunk_id)
                if 'chunk_id' in row:
                    chunk_id = row['chunk_id']
                    CHUNK_DURATION_SEC = 3
                    start_sample = int(chunk_id * CHUNK_DURATION_SEC * target_sample_rate)
                    end_sample = int((chunk_id + 1) * CHUNK_DURATION_SEC * target_sample_rate)
                    # Slice the audio array to get the specific chunk
                    audio_array = audio_array[start_sample:end_sample]
                    # Update filename to include chunk_id for unique identification
                    filename = f"{os.path.splitext(filename)[0]}_chunk{chunk_id}{os.path.splitext(filename)[1]}"
                
                # Prepare model input based on encoder type
                if processor is not None:
                    # Use processor for encoders that require it
                    if model_input_key == 'input_features':
                        # CLAP processor
                        inputs = processor(
                            text=None,
                            audio=[audio_array],
                            return_tensors="pt",
                            sampling_rate=target_sample_rate
                        ).to(device)
                        model_input = inputs[model_input_key]
                    else:
                        # Wav2Vec2, HuBERT, AST processors
                        inputs = processor(
                            audio_array,
                            sampling_rate=target_sample_rate,
                            return_tensors="pt"
                        ).to(device)
                        model_input = inputs[model_input_key]
                else:
                    # Custom CNN: process raw audio tensor directly
                    audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(device)
                    model_input = audio_tensor
                
                # Get embedding
                embedding = model(model_input)
                embedding_np = embedding.cpu().numpy()[0]  # Remove batch dimension
                
                # Prepare metadata
                metadata = {
                    "label": label,
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
    
    parser = argparse.ArgumentParser(description="Index audio dataset into ChromaDB")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset directory (optional for Hugging Face datasets, use 'hf:dataset_name')"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="esc50",
        help="Dataset name. Options: 'esc50', 'urbansound8k', 'gtzan', 'nsynth', 'audioset', or 'hf:dataset_name' for Hugging Face datasets (default: esc50)"
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="clap",
        choices=['clap', 'wav2vec2', 'hubert', 'ast', 'custom_cnn'],
        help="Encoder name (default: clap)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="laion/clap-htsat-unfused",
        help="Hugging Face model name or path (default: laion/clap-htsat-unfused)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for ChromaDB insertion (default: 100)"
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
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU if available"
    )
    
    args = parser.parse_args()
    
    # Validate dataset_path for non-HF datasets
    # Allow nsynth without dataset_path (it tries to load from Hugging Face first)
    if not args.dataset_name.startswith('hf:') and args.dataset_name != 'nsynth' and args.dataset_path is None:
        parser.error("--dataset_path is required for non-Hugging Face datasets")
    
    # Setup device
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        if args.use_gpu:
            print("Warning: GPU requested but not available, using CPU")
        else:
            print(f"Using device: {device}")
    
    # Load model and processor using factory function
    print(f"\nLoading encoder: {args.encoder_name}")
    print(f"Model: {args.model_name}")
    model, processor, target_sample_rate, model_input_key = load_model_and_processor(
        encoder_name=args.encoder_name,
        model_name_or_path=args.model_name,
        device=device
    )
    print(f"Target sample rate: {target_sample_rate} Hz")
    print(f"Model input key: {model_input_key}")
    
    # Load dataset splits using factory function
    print(f"\nLoading dataset: {args.dataset_name}")
    # For Hugging Face datasets or nsynth, dataset_path can be None
    train_df, test_df = load_dataset_splits(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path
    )
    
    # Generate collection name: dataset_encoder_distance
    # For Hugging Face datasets, sanitize the name (replace ':' and '/' with '_')
    dataset_name_clean = args.dataset_name.replace(':', '_').replace('/', '_')
    collection_name = f"{dataset_name_clean}_{args.encoder_name}_{args.distance_metric}"
    print(f"\nCollection name: {collection_name}")
    
    # Index audio files
    index_audio_files(
        train_df=train_df,
        device=device,
        processor=processor,
        model=model,
        target_sample_rate=target_sample_rate,
        model_input_key=model_input_key,
        batch_size=args.batch_size,
        collection_name=collection_name,
        chromadb_path=args.chromadb_path,
        distance_metric=args.distance_metric
    )


if __name__ == "__main__":
    main()

