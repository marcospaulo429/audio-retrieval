"""
Production/Research-Ready Audio Classification Experiment Script

This script provides a comprehensive experiment framework for evaluating
retrieval-based audio classification with different encoders, datasets,
distance metrics, and retrieval strategies.

Docker Usage (with GPU):
    docker run --gpus all \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/chromadb:/app/chromadb \
        -v $(pwd)/ESC-50:/app/data/ESC-50 \
        audio-retrieval-app \
        python scripts/run_experiment.py \
            --dataset_path /app/data/ESC-50 \
            --dataset_name esc50 \
            --encoder_name clap \
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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model_and_processor, load_dataset_splits


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
    target_sample_rate,
    model_input_key,
    collection,
    k,
    distance_metric,
    chunk_id=None
):
    """
    Basic retrieval strategy: Query ChromaDB and perform weighted voting.
    
    Args:
        audio_path: Path to audio file
        device: torch device
        processor: Processor instance (or None for custom_cnn)
        model: Audio encoder model instance
        target_sample_rate: Target sample rate for audio loading
        model_input_key: Key to use when passing input to model
        collection: ChromaDB collection
        k: Number of nearest neighbors
        distance_metric: Distance metric used
        chunk_id: Optional chunk ID for datasets with chunking (e.g., GTZAN)
        
    Returns:
        predicted_class, class_scores, neighbors
    """
    # Load audio at target sample rate
    audio_array, sr = librosa.load(audio_path, sr=target_sample_rate)
    
    # Handle chunking for GTZAN (or other datasets with chunk_id)
    if chunk_id is not None:
        CHUNK_DURATION_SEC = 3
        start_sample = int(chunk_id * CHUNK_DURATION_SEC * target_sample_rate)
        end_sample = int((chunk_id + 1) * CHUNK_DURATION_SEC * target_sample_rate)
        # Slice the audio array to get the specific chunk
        audio_array = audio_array[start_sample:end_sample]
    
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
    
    # Generate query embedding
    with torch.no_grad():
        query_embedding = model(model_input)
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
    target_sample_rate,
    model_input_key,
    collection,
    k,
    distance_metric,
    chunk_id=None
):
    """
    Rerank retrieval strategy: Get 3*k candidates from ChromaDB, 
    then re-calculate exact similarity in PyTorch and pick top k.
    
    Args:
        audio_path: Path to audio file
        device: torch device
        processor: Processor instance (or None for custom_cnn)
        model: Audio encoder model instance
        target_sample_rate: Target sample rate for audio loading
        model_input_key: Key to use when passing input to model
        collection: ChromaDB collection
        k: Number of nearest neighbors (final)
        distance_metric: Distance metric used
        chunk_id: Optional chunk ID for datasets with chunking (e.g., GTZAN)
        
    Returns:
        predicted_class, class_scores, neighbors
    """
    # Load audio at target sample rate
    audio_array, sr = librosa.load(audio_path, sr=target_sample_rate)
    
    # Handle chunking for GTZAN (or other datasets with chunk_id)
    if chunk_id is not None:
        CHUNK_DURATION_SEC = 3
        start_sample = int(chunk_id * CHUNK_DURATION_SEC * target_sample_rate)
        end_sample = int((chunk_id + 1) * CHUNK_DURATION_SEC * target_sample_rate)
        # Slice the audio array to get the specific chunk
        audio_array = audio_array[start_sample:end_sample]
    
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
    
    # Generate query embedding
    with torch.no_grad():
        query_embedding = model(model_input)  # [1, embedding_dim]
    
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
    target_sample_rate,
    model_input_key,
    collection,
    k,
    distance_metric,
    retrieval_strategy,
    chunk_id=None
):
    """
    Classify audio using specified retrieval strategy.
    
    Args:
        audio_path: Path to audio file
        device: torch device
        processor: Processor instance (or None for custom_cnn)
        model: Audio encoder model instance
        target_sample_rate: Target sample rate for audio loading
        model_input_key: Key to use when passing input to model
        collection: ChromaDB collection
        k: Number of nearest neighbors
        distance_metric: Distance metric
        retrieval_strategy: 'basic' or 'rerank'
        chunk_id: Optional chunk ID for datasets with chunking (e.g., GTZAN)
        
    Returns:
        predicted_class, class_scores, neighbors
    """
    if retrieval_strategy == 'basic':
        return classify_audio_basic(
            audio_path, device, processor, model, target_sample_rate, model_input_key,
            collection, k, distance_metric, chunk_id=chunk_id
        )
    elif retrieval_strategy == 'rerank':
        return classify_audio_rerank(
            audio_path, device, processor, model, target_sample_rate, model_input_key,
            collection, k, distance_metric, chunk_id=chunk_id
        )
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")




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
    ax.set_title('Confusion Matrix - Audio Classification', fontsize=14, pad=20)
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
    test_df,
    device,
    processor,
    model,
    target_sample_rate,
    model_input_key,
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
        test_df: DataFrame with 'file_path' and 'label' columns
        device: torch device
        processor: Processor instance (or None for custom_cnn)
        model: Audio encoder model instance
        target_sample_rate: Target sample rate for audio loading
        model_input_key: Key to use when passing input to model
        collection: ChromaDB collection
        k: Number of nearest neighbors
        distance_metric: Distance metric
        retrieval_strategy: Retrieval strategy
        writer: TensorBoard SummaryWriter (optional)
        step: Step number for TensorBoard logging
        
    Returns:
        y_true, y_pred, all_labels, metrics_dict
    """
    # Get all unique labels
    all_labels = sorted(test_df['label'].unique())
    
    # Initialize lists
    y_true = []
    y_pred = []
    
    # Evaluation loop
    print(f"\nRunning inference on {len(test_df)} test files...")
    print(f"Strategy: {retrieval_strategy}, Distance: {distance_metric}, k: {k}")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        audio_path = row['file_path']
        real_label = row['label']
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        # Get chunk_id if present (for GTZAN)
        chunk_id = row.get('chunk_id', None)
        
        try:
            predicted_label, _, _ = classify_audio(
                audio_path=audio_path,
                device=device,
                processor=processor,
                model=model,
                target_sample_rate=target_sample_rate,
                model_input_key=model_input_key,
                collection=collection,
                k=k,
                distance_metric=distance_metric,
                retrieval_strategy=retrieval_strategy,
                chunk_id=chunk_id
            )
            
            y_true.append(real_label)
            y_pred.append(predicted_label)
        
        except Exception as e:
            filename = os.path.basename(audio_path)
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
        description="Production/Research-Ready Audio Classification Experiment Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset directory (optional for Hugging Face datasets, use 'hf:dataset_name')"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="esc50",
        help="Dataset name. Options: 'esc50', 'urbansound8k', 'gtzan', 'nsynth', 'audioset', or 'hf:dataset_name' for Hugging Face datasets (default: esc50)"
    )
    
    # Model arguments
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
    
    print("="*70)
    print("Experiment Configuration")
    print("="*70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Model: {args.model_name}")
    print(f"Distance Metric: {args.distance_metric}")
    print(f"Retrieval Strategy: {args.retrieval_strategy}")
    print(f"k: {args.k}")
    print(f"Collection: {collection_name}")
    print(f"ChromaDB Path: {args.chromadb_path}")
    print(f"Log Dir: {args.log_dir}")
    print("="*70)
    
    # Setup ChromaDB
    print(f"\nConnecting to ChromaDB collection: {collection_name}")
    client = chromadb.PersistentClient(path=args.chromadb_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection found with {collection.count()} items")
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found!")
        print(f"Please run index_esc50.py first with:")
        print(f"  --dataset_name {args.dataset_name}")
        print(f"  --encoder_name {args.encoder_name}")
        print(f"  --distance_metric {args.distance_metric}")
        print(f"Error details: {e}")
        return
    
    # Setup TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    print(f"\nTensorBoard logs will be saved to: {args.log_dir}")
    print(f"View with: tensorboard --logdir {args.log_dir}")
    
    # Evaluate test set
    y_true, y_pred, all_labels, metrics = evaluate_test_set(
        test_df=test_df,
        device=device,
        processor=processor,
        model=model,
        target_sample_rate=target_sample_rate,
        model_input_key=model_input_key,
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
        plt.title('Confusion Matrix - Audio Classification', fontsize=14, pad=20)
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

