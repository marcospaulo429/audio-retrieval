"""
Audio Classification Pipeline
Classifies audio clips using retrieval-based weighted k-NN.
"""

import os
import sys
from pathlib import Path
import torch
import librosa
import chromadb
import numpy as np
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
    processor = ClapProcessor.from_pretrained(model_name)
    
    # Instantiate model
    model = CLAPAudioEncoder(model_name=model_name, freeze=True).to(device)
    model.eval()
    
    return device, processor, model


def classify_audio(
    audio_path,
    device,
    processor,
    model,
    collection_name="esc50",
    k=5,
    chromadb_path="./chromadb"
):
    """
    Classify an audio clip using weighted k-NN retrieval.
    
    Args:
        audio_path: Path to audio file to classify
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        collection_name: Name of ChromaDB collection
        k: Number of nearest neighbors to retrieve
        chromadb_path: Path to ChromaDB persistent storage
        
    Returns:
        predicted_class: Predicted class label
        class_scores: Dictionary of class scores
        neighbors: List of k nearest neighbors with their distances
    """
    # Load ChromaDB collection with persistent storage
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_collection(name=collection_name)
    
    # Load audio with librosa
    audio_array, sr = librosa.load(audio_path, sr=48000)
    
    # Use processor to prepare it
    inputs = processor(
        text=None,
        audio=[audio_array],  # Use 'audio' instead of deprecated 'audios'
        return_tensors="pt",
        sampling_rate=48000
    ).to(device)
    
    # Generate query embedding
    with torch.no_grad():
        query_embedding = model(inputs["input_features"])
        query_embedding_np = query_embedding.cpu().numpy()[0]  # Remove batch dimension
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding_np.tolist()],
        n_results=k
    )
    
    # Implement Weighted k-NN
    class_scores = {}
    neighbors = []
    
    # Iterate through k results
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    ids = results['ids'][0]
    
    for i, (distance, metadata, file_id) in enumerate(zip(distances, metadatas, ids)):
        label = metadata['label']
        filename = metadata.get('filename', file_id)
        
        # Calculate similarity: Cosine Distance = 1 - Cosine Similarity
        similarity = 1.0 - distance
        
        # Add to class score
        class_scores[label] = class_scores.get(label, 0.0) + similarity
        
        # Store neighbor info
        neighbors.append({
            'filename': filename,
            'label': label,
            'distance': distance,
            'similarity': similarity
        })
    
    # Find the class with the highest score
    predicted_class = max(class_scores, key=class_scores.get)
    
    return predicted_class, class_scores, neighbors


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify audio using retrieval-based k-NN")
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to audio file to classify"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors (default: 5)"
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
        help="Path to ChromaDB persistent storage (default: ./chromadb)"
    )
    parser.add_argument(
        "--show_neighbors",
        action="store_true",
        help="Show k nearest neighbors"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return
    
    # Setup environment
    device, processor, model = setup_environment()
    
    # Classify audio
    print(f"\nClassifying audio: {args.audio_path}")
    predicted_class, class_scores, neighbors = classify_audio(
        audio_path=args.audio_path,
        device=device,
        processor=processor,
        model=model,
        collection_name=args.collection_name,
        k=args.k,
        chromadb_path=args.chromadb_path
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Predicted Class: {predicted_class}")
    print(f"{'='*60}")
    
    print(f"\nClass Scores (Weighted k-NN):")
    # Sort by score descending
    sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    for label, score in sorted_scores:
        print(f"  {label}: {score:.4f}")
    
    if args.show_neighbors:
        print(f"\nTop {args.k} Nearest Neighbors:")
        for i, neighbor in enumerate(neighbors, 1):
            print(f"  {i}. {neighbor['filename']}")
            print(f"     Label: {neighbor['label']}")
            print(f"     Similarity: {neighbor['similarity']:.4f} (Distance: {neighbor['distance']:.4f})")
    
    print()


if __name__ == "__main__":
    main()

