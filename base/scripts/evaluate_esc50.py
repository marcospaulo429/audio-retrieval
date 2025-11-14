"""
ESC-50 Evaluation Pipeline
Evaluates the retrieval-based audio classification on the test set (fold 5).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import chromadb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import from classify_audio script
from classify_audio import setup_environment, classify_audio


def load_test_set(esc50_path):
    """
    Load test set files (fold 5).
    
    Args:
        esc50_path: Path to ESC-50 dataset directory
        
    Returns:
        DataFrame with test set files and their labels
    """
    meta_path = os.path.join(esc50_path, "meta", "esc50.csv")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {meta_path}\n"
            "Please download ESC-50 from: https://github.com/karolpiczak/ESC-50"
        )
    
    df = pd.read_csv(meta_path)
    
    # Test Set: Files where fold is 5
    test_df = df[df['fold'] == 5].copy()
    
    print(f"Test set: {len(test_df)} files (fold 5)")
    return test_df, df


def evaluate_test_set(
    esc50_path,
    device,
    processor,
    model,
    collection_name="esc50",
    k=5,
    chromadb_path="./chromadb"
):
    """
    Evaluate the classification pipeline on the test set.
    
    Args:
        esc50_path: Path to ESC-50 dataset directory
        device: torch device
        processor: ClapProcessor instance
        model: CLAPAudioEncoder instance
        collection_name: Name of ChromaDB collection
        k: Number of nearest neighbors for classification
        chromadb_path: Path to ChromaDB persistent storage
        
    Returns:
        y_true: List of true labels
        y_pred: List of predicted labels
        all_labels: Sorted list of all class labels
    """
    # Load test set
    test_df, full_df = load_test_set(esc50_path)
    
    # Get all unique class labels (sorted)
    all_labels = sorted(full_df['category'].unique())
    
    # Initialize lists
    y_true = []
    y_pred = []
    
    audio_dir = os.path.join(esc50_path, "audio")
    
    # Evaluation loop
    print(f"\nRunning inference on {len(test_df)} test files...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        filename = row['filename']
        real_label = row['category']
        
        audio_path = os.path.join(audio_dir, filename)
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        try:
            # Call classify_audio function
            predicted_label, _, _ = classify_audio(
                audio_path=audio_path,
                device=device,
                processor=processor,
                model=model,
                collection_name=collection_name,
                k=k,
                chromadb_path=chromadb_path
            )
            
            # Append to lists
            y_true.append(real_label)
            y_pred.append(predicted_label)
        
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    print(f"\nSuccessfully evaluated {len(y_true)} files")
    return y_true, y_pred, all_labels


def calculate_and_print_metrics(y_true, y_pred):
    """
    Calculate and print evaluation metrics.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate classification report
    report = classification_report(y_true, y_pred)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print(report)


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        labels: List of all class labels
        save_path: Optional path to save the figure
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create figure
    plt.figure(figsize=(20, 20))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=labels,
        yticklabels=labels,
        cmap='Blues',
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - ESC-50 Classification', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
    
    plt.show()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ESC-50 classification pipeline")
    parser.add_argument(
        "--esc50_path",
        type=str,
        required=True,
        help="Path to ESC-50 dataset directory"
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
        "--save_confusion_matrix",
        type=str,
        default=None,
        help="Path to save confusion matrix figure (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    print("Setting up environment...")
    device, processor, model = setup_environment()
    
    # Evaluate test set
    y_true, y_pred, all_labels = evaluate_test_set(
        esc50_path=args.esc50_path,
        device=device,
        processor=processor,
        model=model,
        collection_name=args.collection_name,
        k=args.k,
        chromadb_path=args.chromadb_path
    )
    
    # Calculate and print metrics
    calculate_and_print_metrics(y_true, y_pred)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, all_labels, save_path=args.save_confusion_matrix)


if __name__ == "__main__":
    main()

