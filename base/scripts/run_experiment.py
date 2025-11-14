"""
Production/Research-Ready ESC-50 Experiment Script

This script provides a comprehensive experiment framework for evaluating
retrieval-based audio classification with different models, distance metrics,
and retrieval strategies.

Docker Usage (with GPU):
    docker run --gpus all \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/chromadb:/app/chromadb \
        -v $(pwd)/ESC-50:/app/data/ESC-50 \
        audio-retrieval-app \
        python scripts/run_experiment.py \
            --esc50_path /app/data/ESC-50 \
            --model_name laion/clap-htsat-unfused \
            --distance_metric cosine \
            --retrieval_strategy basic \
            --use_gpu \
            --log_dir /app/logs
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import chromadb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import ClapProcessor
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.audio_encoder import CLAPAudioEncoder


def setup_environment(model_name, use_gpu):
    """
    Setup device, processor, and model.
    
    Args:
        model_name: CLAP model name
        use_gpu: Whether to use GPU
        
    Returns:
        device, processor, model
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        if use_gpu:
            print("Warning: GPU requested but not available, using CPU")
        else:
            print(f"Using device: {device}")
    
    # Load processor
    print(f"Loading processor: {model_name}")
    processor = ClapProcessor.from_pretrained(model_name)
    
    # Instantiate model
    print(f"Loading model: {model_name}")
    model = CLAPAudioEncoder(model_name=model_name, freeze=True).to(device)
    model.eval()
    
    return device, processor, model


def get_collection_name(base_name, distance_metric):
    """
    Generate dynamic collection name based on distance metric.
    
    Args:
        base_name: Base collection name
        distance_metric: Distance metric ('cosine', 'l2', 'ip')
        
    Returns:
        Collection name string
    """
    return f"{base_name}_{distance_metric}"


def distance_to_similarity(distance, distance_metric):
    """
    Convert distance to similarity based on metric.
    
    Args:
        distance: Distance value
        distance_metric: Distance metric ('cosine', 'l2', 'ip')
        
    Returns:
        Similarity value
    """
    if distance_metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        return 1.0 - distance
    elif distance_metric == 'l2':
        # L2 distance, convert to similarity: 1 / (1 + distance)
        return 1.0 / (1.0 + distance)
    elif distance_metric == 'ip':
        # Inner product (already a similarity measure, but may need normalization)
        # For inner product, higher is better, so we can use it directly
        # But ChromaDB returns it as distance, so we might need to negate
        # Actually, ChromaDB's 'ip' space uses negative inner product as distance
        # So similarity = -distance
        return -distance
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")


def classify_audio_basic(
    audio_path,
    device,
    processor,
    model,
    collection,
    k,
    distance_metric
):
    """
    Basic retrieval strategy: Query ChromaDB and perform weighted voting.
    
    Args:
        audio_path: Path to audio file
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        collection: ChromaDB collection
        k: Number of nearest neighbors
        distance_metric: Distance metric used
        
    Returns:
        predicted_class, class_scores, neighbors
    """
    # Load audio
    audio_array, sr = librosa.load(audio_path, sr=48000)
    
    # Process audio
    inputs = processor(
        text=None,
        audio=[audio_array],
        return_tensors="pt",
        sampling_rate=48000
    ).to(device)
    
    # Generate query embedding
    with torch.no_grad():
        query_embedding = model(inputs["input_features"])
        query_embedding_np = query_embedding.cpu().numpy()[0]
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding_np.tolist()],
        n_results=k
    )
    
    # Weighted k-NN
    class_scores = {}
    neighbors = []
    
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    ids = results['ids'][0]
    
    for distance, metadata, file_id in zip(distances, metadatas, ids):
        label = metadata['label']
        filename = metadata.get('filename', file_id)
        
        # Convert distance to similarity
        similarity = distance_to_similarity(distance, distance_metric)
        
        # Add to class score
        class_scores[label] = class_scores.get(label, 0.0) + similarity
        
        neighbors.append({
            'filename': filename,
            'label': label,
            'distance': distance,
            'similarity': similarity
        })
    
    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class, class_scores, neighbors


def classify_audio_rerank(
    audio_path,
    device,
    processor,
    model,
    collection,
    k,
    distance_metric
):
    """
    Rerank retrieval strategy: Get 3*k candidates from ChromaDB, 
    then re-calculate exact similarity in PyTorch and pick top k.
    
    Args:
        audio_path: Path to audio file
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        collection: ChromaDB collection
        k: Number of nearest neighbors (final)
        distance_metric: Distance metric used
        
    Returns:
        predicted_class, class_scores, neighbors
    """
    # Load audio
    audio_array, sr = librosa.load(audio_path, sr=48000)
    
    # Process audio
    inputs = processor(
        text=None,
        audio=[audio_array],
        return_tensors="pt",
        sampling_rate=48000
    ).to(device)
    
    # Generate query embedding
    with torch.no_grad():
        query_embedding = model(inputs["input_features"])  # [1, embedding_dim]
    
    # Query ChromaDB for 3*k candidates (fast HNSW search)
    n_candidates = k * 3
    try:
        results = collection.query(
            query_embeddings=[query_embedding.cpu().numpy()[0].tolist()],
            n_results=n_candidates,
            include=['embeddings', 'metadatas', 'distances']
        )
        
        # Check if embeddings are returned
        if 'embeddings' not in results or results['embeddings'] is None or len(results['embeddings']) == 0:
            raise ValueError("ChromaDB did not return embeddings. Rerank strategy requires embeddings.")
        
        # Extract candidate embeddings and metadata
        candidate_embeddings = torch.tensor(
            results['embeddings'][0], 
            dtype=torch.float32,
            device=device
        )  # [n_candidates, embedding_dim]
    except Exception as e:
        raise ValueError(
            f"Failed to retrieve embeddings from ChromaDB for rerank strategy: {e}\n"
            "Make sure your ChromaDB version supports returning embeddings in queries."
        )
    
    metadatas = results['metadatas'][0]
    ids = results['ids'][0]
    chroma_distances = results['distances'][0]
    
    # Calculate exact similarity in PyTorch
    query_embedding_expanded = query_embedding.expand(candidate_embeddings.shape[0], -1)
    
    if distance_metric == 'cosine':
        # Cosine similarity
        similarities = F.cosine_similarity(
            query_embedding_expanded, 
            candidate_embeddings, 
            dim=1
        )  # [n_candidates]
    elif distance_metric == 'l2':
        # L2 distance, convert to similarity
        distances = torch.norm(query_embedding_expanded - candidate_embeddings, p=2, dim=1)
        similarities = 1.0 / (1.0 + distances)
    elif distance_metric == 'ip':
        # Inner product
        similarities = torch.sum(query_embedding_expanded * candidate_embeddings, dim=1)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Get top k indices
    top_k_indices = torch.topk(similarities, k=k).indices.cpu().numpy()
    
    # Weighted k-NN on top k
    class_scores = {}
    neighbors = []
    
    for idx in top_k_indices:
        label = metadatas[idx]['label']
        filename = metadatas[idx].get('filename', ids[idx])
        similarity = similarities[idx].item()
        distance = chroma_distances[idx]
        
        # Add to class score
        class_scores[label] = class_scores.get(label, 0.0) + similarity
        
        neighbors.append({
            'filename': filename,
            'label': label,
            'distance': distance,
            'similarity': similarity
        })
    
    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class, class_scores, neighbors


def classify_audio(
    audio_path,
    device,
    processor,
    model,
    collection,
    k,
    distance_metric,
    retrieval_strategy
):
    """
    Classify audio using specified retrieval strategy.
    
    Args:
        audio_path: Path to audio file
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        collection: ChromaDB collection
        k: Number of nearest neighbors
        distance_metric: Distance metric
        retrieval_strategy: 'basic' or 'rerank'
        
    Returns:
        predicted_class, class_scores, neighbors
    """
    if retrieval_strategy == 'basic':
        return classify_audio_basic(
            audio_path, device, processor, model, collection, k, distance_metric
        )
    elif retrieval_strategy == 'rerank':
        return classify_audio_rerank(
            audio_path, device, processor, model, collection, k, distance_metric
        )
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")


def load_test_set(esc50_path):
    """Load test set files (fold 5)."""
    meta_path = os.path.join(esc50_path, "meta", "esc50.csv")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {meta_path}\n"
            "Please download ESC-50 from: https://github.com/karolpiczak/ESC-50"
        )
    
    df = pd.read_csv(meta_path)
    test_df = df[df['fold'] == 5].copy()
    
    print(f"Test set: {len(test_df)} files (fold 5)")
    return test_df, df


def plot_confusion_matrix_to_image(y_true, y_pred, labels):
    """
    Plot confusion matrix and convert to numpy array for TensorBoard.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of all class labels
        
    Returns:
        numpy array of the image
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=labels,
        yticklabels=labels,
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix - ESC-50 Classification', fontsize=14, pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    
    # Convert to numpy array
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    
    return img_array


def evaluate_test_set(
    esc50_path,
    device,
    processor,
    model,
    collection,
    k,
    distance_metric,
    retrieval_strategy,
    writer=None,
    step=0
):
    """
    Evaluate the classification pipeline on the test set.
    
    Args:
        esc50_path: Path to ESC-50 dataset directory
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        collection: ChromaDB collection
        k: Number of nearest neighbors
        distance_metric: Distance metric
        retrieval_strategy: Retrieval strategy
        writer: TensorBoard SummaryWriter (optional)
        step: Step number for TensorBoard logging
        
    Returns:
        y_true, y_pred, all_labels, metrics_dict
    """
    # Load test set
    test_df, full_df = load_test_set(esc50_path)
    all_labels = sorted(full_df['category'].unique())
    
    # Initialize lists
    y_true = []
    y_pred = []
    
    audio_dir = os.path.join(esc50_path, "audio")
    
    # Evaluation loop
    print(f"\nRunning inference on {len(test_df)} test files...")
    print(f"Strategy: {retrieval_strategy}, Distance: {distance_metric}, k: {k}")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        filename = row['filename']
        real_label = row['category']
        
        audio_path = os.path.join(audio_dir, filename)
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        try:
            predicted_label, _, _ = classify_audio(
                audio_path=audio_path,
                device=device,
                processor=processor,
                model=model,
                collection=collection,
                k=k,
                distance_metric=distance_metric,
                retrieval_strategy=retrieval_strategy
            )
            
            y_true.append(real_label)
            y_pred.append(predicted_label)
        
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    print(f"\nSuccessfully evaluated {len(y_true)} files")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    metrics_dict = {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }
    
    # Log to TensorBoard if writer provided
    if writer is not None:
        writer.add_scalar('Metrics/Accuracy', accuracy, step)
        writer.add_scalar('Metrics/Macro_F1', macro_f1, step)
        
        # Log confusion matrix as image
        cm_image = plot_confusion_matrix_to_image(y_true, y_pred, all_labels)
        writer.add_image('Confusion_Matrix', cm_image, step, dataformats='HWC')
    
    return y_true, y_pred, all_labels, metrics_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Production/Research-Ready ESC-50 Experiment Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--esc50_path",
        type=str,
        required=True,
        help="Path to ESC-50 dataset directory"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="laion/clap-htsat-unfused",
        help="CLAP model name for CLAPAudioEncoder"
    )
    
    # ChromaDB arguments
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=['cosine', 'ip', 'l2'],
        default='cosine',
        help="Distance metric for ChromaDB and similarity calculation"
    )
    parser.add_argument(
        "--chromadb_path",
        type=str,
        default="./chromadb",
        help="Path to ChromaDB persistent storage"
    )
    parser.add_argument(
        "--collection_base_name",
        type=str,
        default="esc50",
        help="Base name for ChromaDB collection (will be appended with distance metric)"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        choices=['basic', 'rerank'],
        default='basic',
        help="Retrieval strategy: 'basic' (standard k-NN) or 'rerank' (3*k candidates, exact similarity)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors for classification"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Path for TensorBoard logs"
    )
    
    # GPU argument
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU if available"
    )
    
    # Output arguments
    parser.add_argument(
        "--save_confusion_matrix",
        type=str,
        default=None,
        help="Path to save confusion matrix figure (optional)"
    )
    
    args = parser.parse_args()
    
    # Generate collection name
    collection_name = get_collection_name(args.collection_base_name, args.distance_metric)
    
    print("="*70)
    print("ESC-50 Experiment Configuration")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Distance Metric: {args.distance_metric}")
    print(f"Retrieval Strategy: {args.retrieval_strategy}")
    print(f"k: {args.k}")
    print(f"Collection: {collection_name}")
    print(f"ChromaDB Path: {args.chromadb_path}")
    print(f"Log Dir: {args.log_dir}")
    print(f"GPU: {args.use_gpu}")
    print("="*70)
    
    # Setup environment
    print("\nSetting up environment...")
    device, processor, model = setup_environment(args.model_name, args.use_gpu)
    
    # Setup ChromaDB
    print(f"\nConnecting to ChromaDB collection: {collection_name}")
    client = chromadb.PersistentClient(path=args.chromadb_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection found with {collection.count()} items")
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found!")
        print(f"Please run index_esc50.py first with distance_metric={args.distance_metric}")
        print(f"Error details: {e}")
        return
    
    # Setup TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create experiment tag for TensorBoard
    experiment_tag = f"{args.model_name.split('/')[-1]}_{args.distance_metric}_{args.retrieval_strategy}_k{args.k}"
    
    print(f"\nTensorBoard logs will be saved to: {args.log_dir}")
    print(f"View with: tensorboard --logdir {args.log_dir}")
    
    # Evaluate test set
    y_true, y_pred, all_labels, metrics = evaluate_test_set(
        esc50_path=args.esc50_path,
        device=device,
        processor=processor,
        model=model,
        collection=collection,
        k=args.k,
        distance_metric=args.distance_metric,
        retrieval_strategy=args.retrieval_strategy,
        writer=writer,
        step=0
    )
    
    # Print results
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("="*70)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save confusion matrix if requested
    if args.save_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=all_labels,
            yticklabels=all_labels,
            cmap='Blues',
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix - ESC-50 Classification', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(args.save_confusion_matrix, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {args.save_confusion_matrix}")
        plt.close()
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTensorBoard logs saved. Run 'tensorboard --logdir {args.log_dir}' to view.")


if __name__ == "__main__":
    main()

