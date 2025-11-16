"""
Utility functions for loading models, processors, and datasets.

This module provides factory functions to abstract the complexity of
loading different audio encoders and datasets.
"""

import os
import json
import pandas as pd
import torch
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

# Import all encoder classes
from models.audio_encoder import (
    CLAPAudioEncoder,
    Wav2Vec2AudioEncoder,
    HuBERTAudioEncoder,
    AudioSpectrogramTransformer,
    CNNTransformerAudioEncoder
)


def load_model_and_processor(
    encoder_name: str,
    model_name_or_path: str,
    device: str,
    embedding_dim: int = 512,
    freeze: bool = True
) -> Tuple[torch.nn.Module, Optional[object], int, str]:
    """
    Factory function to load model and processor based on encoder name.
    
    Args:
        encoder_name: Name of the encoder ('clap', 'wav2vec2', 'hubert', 'ast', 'custom_cnn')
        model_name_or_path: Hugging Face model name or local path
        device: Device to load model on ('cuda' or 'cpu')
        embedding_dim: Embedding dimension (default: 512)
        freeze: Whether to freeze model parameters (default: True)
        
    Returns:
        Tuple of (model, processor, target_sample_rate, model_input_key)
        - model: The loaded encoder model
        - processor: The processor/feature extractor (None for custom_cnn)
        - target_sample_rate: Target sample rate for audio loading
        - model_input_key: Key to use when passing input to model
    """
    encoder_name = encoder_name.lower()
    
    if encoder_name == 'clap':
        from transformers import ClapProcessor
        
        processor = ClapProcessor.from_pretrained(model_name_or_path)
        model = CLAPAudioEncoder(
            model_name=model_name_or_path,
            embedding_dim=embedding_dim,
            freeze=freeze
        )
        target_sample_rate = 48000
        model_input_key = 'input_features'
        
    elif encoder_name == 'wav2vec2':
        from transformers import Wav2Vec2Processor
        
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        model = Wav2Vec2AudioEncoder(
            model_name=model_name_or_path,
            embedding_dim=embedding_dim,
            freeze_encoder=freeze,
            freeze_feature_extractor=freeze
        )
        target_sample_rate = 16000
        model_input_key = 'input_values'
        
    elif encoder_name == 'hubert':
        from transformers import HubertProcessor
        
        processor = HubertProcessor.from_pretrained(model_name_or_path)
        model = HuBERTAudioEncoder(
            model_name=model_name_or_path,
            embedding_dim=embedding_dim,
            freeze_feature_extractor=freeze
        )
        target_sample_rate = 16000
        model_input_key = 'input_values'
        
    elif encoder_name == 'ast':
        from transformers import AutoFeatureExtractor
        
        processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = AudioSpectrogramTransformer(
            model_name=model_name_or_path,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze
        )
        target_sample_rate = 16000
        model_input_key = 'input_values'
        
    elif encoder_name == 'custom_cnn':
        processor = None
        model = CNNTransformerAudioEncoder(
            embedding_dim=embedding_dim
        )
        target_sample_rate = 16000
        model_input_key = 'raw_tensor'
        
    else:
        raise ValueError(
            f"Unknown encoder_name: {encoder_name}. "
            f"Supported: 'clap', 'wav2vec2', 'hubert', 'ast', 'custom_cnn'"
        )
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, processor, target_sample_rate, model_input_key


def load_dataset_splits(
    dataset_name: str,
    dataset_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Factory function to load and split dataset into train and test sets.
    
    Args:
        dataset_name: Name of the dataset ('esc50', 'urbansound8k', 'gtzan', 'nsynth', 'audioset', or 'hf:dataset_name' for Hugging Face)
        dataset_path: Path to the dataset directory (optional for Hugging Face datasets)
        
    Returns:
        Tuple of (train_df, test_df)
        Both DataFrames have columns: 'file_path' (full path to audio) and 'label' (class)
        For GTZAN, also includes 'chunk_id' column (0-9)
    """
    dataset_name = dataset_name.lower()
    
    # Validate dataset_path for non-HF datasets
    # Allow nsynth without dataset_path (it tries to load from Hugging Face first)
    if not dataset_name.startswith('hf:') and dataset_name != 'nsynth' and dataset_path is None:
        raise ValueError(f"dataset_path is required for dataset '{dataset_name}'")
    
    if dataset_name == 'esc50':
        meta_path = os.path.join(dataset_path, "meta", "esc50.csv")
        
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
        audio_dir = os.path.join(dataset_path, "audio")
        
        # Create full file paths
        df['file_path'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))
        df['label'] = df['category']
        
        # Split: folds 1-4 for training, fold 5 for test
        train_df = df[df['fold'].isin([1, 2, 3, 4])].copy()
        test_df = df[df['fold'] == 5].copy()
        
        # Select only required columns
        train_df = train_df[['file_path', 'label']].copy()
        test_df = test_df[['file_path', 'label']].copy()
        
        print(f"ESC-50 dataset loaded:")
        print(f"  Training set: {len(train_df)} files (folds 1-4)")
        print(f"  Test set: {len(test_df)} files (fold 5)")
        
        return train_df, test_df
        
    elif dataset_name == 'urbansound8k':
        meta_path = os.path.join(dataset_path, "metadata", "UrbanSound8K.csv")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"UrbanSound8K metadata not found at {meta_path}\n"
                "Please download UrbanSound8K from: https://urbansounddataset.weebly.com/\n"
                "Expected structure:\n"
                "  UrbanSound8K/\n"
                "    metadata/\n"
                "      UrbanSound8K.csv\n"
                "    audio/\n"
                "      fold1/\n"
                "      fold2/\n"
                "      ...\n"
                "      fold10/"
            )
        
        df = pd.read_csv(meta_path)
        
        # Create full file paths
        # UrbanSound8K has files in subdirectories: fold1, fold2, etc.
        def get_file_path(row):
            fold = f"fold{row['fold']}"
            return os.path.join(dataset_path, "audio", fold, row['slice_file_name'])
        
        df['file_path'] = df.apply(get_file_path, axis=1)
        df['label'] = df['class']
        
        # Split: folds 1-9 for training, fold 10 for test
        train_df = df[df['fold'].isin(range(1, 10))].copy()
        test_df = df[df['fold'] == 10].copy()
        
        # Select only required columns
        train_df = train_df[['file_path', 'label']].copy()
        test_df = test_df[['file_path', 'label']].copy()
        
        print(f"UrbanSound8K dataset loaded:")
        print(f"  Training set: {len(train_df)} files (folds 1-9)")
        print(f"  Test set: {len(test_df)} files (fold 10)")
        
        return train_df, test_df
    
    elif dataset_name == 'gtzan':
        # GTZAN: Music genre classification dataset
        # Files are 30 seconds, need to chunk into 10 chunks of 3 seconds each
        audio_dir = os.path.join(dataset_path, "genres_original")
        
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(
                f"GTZAN audio directory not found at {audio_dir}\n"
                "Please download GTZAN from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification\n"
                "Expected structure:\n"
                "  GTZAN/\n"
                "    genres_original/\n"
                "      blues/\n"
                "      classical/\n"
                "      ...\n"
                "      *.wav"
            )
        
        # Collect all audio files with their labels (genre = folder name)
        data = []
        for genre_folder in os.listdir(audio_dir):
            genre_path = os.path.join(audio_dir, genre_folder)
            if os.path.isdir(genre_path):
                for filename in os.listdir(genre_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(genre_path, filename)
                        data.append({
                            'file_path': file_path,
                            'label': genre_folder
                        })
        
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        # Create chunks: each 30s file becomes 10 chunks of 3s
        chunked_data = []
        for _, row in df.iterrows():
            for chunk_id in range(10):  # 0 to 9
                chunked_data.append({
                    'file_path': row['file_path'],
                    'label': row['label'],
                    'chunk_id': chunk_id
                })
        
        df_chunked = pd.DataFrame(chunked_data)
        
        # Stratified split: 80% train, 20% test
        train_df, test_df = train_test_split(
            df_chunked,
            test_size=0.2,
            stratify=df_chunked['label'],
            random_state=42
        )
        
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        print(f"GTZAN dataset loaded:")
        print(f"  Training set: {len(train_df)} chunks ({len(train_df) // 10} files)")
        print(f"  Test set: {len(test_df)} chunks ({len(test_df) // 10} files)")
        print(f"  Chunk duration: 3 seconds per chunk")
        
        return train_df, test_df
    
    elif dataset_name == 'nsynth':
        # NSynth: Instrument notes dataset from Hugging Face
        # Try multiple approaches to load the dataset
        hf_error = None
        
        # Try 1: Load from Hugging Face (may fail due to deprecated script format)
        try:
            from datasets import load_dataset
            print("Attempting to load NSynth from Hugging Face...")
            
            # Try different loading methods
            try:
                # Method 1: Direct load (may fail with newer datasets library)
                dataset = load_dataset("jg583/NSynth")
            except RuntimeError as e:
                if "Dataset scripts are no longer supported" in str(e):
                    # Try with trust_remote_code=True (if supported)
                    try:
                        dataset = load_dataset("jg583/NSynth", trust_remote_code=True)
                    except Exception:
                        # Try with an older datasets version workaround
                        raise RuntimeError(
                            "NSynth dataset uses an outdated script format that is no longer supported.\n"
                            "Solutions:\n"
                            "  1. Download NSynth locally and provide --dataset_path\n"
                            "  2. Use a compatible datasets version: pip install 'datasets<2.14.0'\n"
                            "  3. Download from: https://magenta.tensorflow.org/datasets/nsynth"
                        ) from e
                else:
                    raise
            
            # Convert to DataFrames
            train_data = []
            for item in dataset['train']:
                # Use instrument_family_str as label (or instrument_source_str)
                label = item.get('instrument_family_str', item.get('instrument_source_str', 'unknown'))
                train_data.append({
                    'file_path': item['audio']['path'],
                    'label': label
                })
            
            test_data = []
            for item in dataset['test']:
                label = item.get('instrument_family_str', item.get('instrument_source_str', 'unknown'))
                test_data.append({
                    'file_path': item['audio']['path'],
                    'label': label
                })
            
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            
            print(f"NSynth dataset loaded from Hugging Face:")
            print(f"  Training set: {len(train_df)} files")
            print(f"  Test set: {len(test_df)} files")
            
            return train_df, test_df
            
        except Exception as e:
            hf_error = str(e)
            print(f"Could not load from Hugging Face: {hf_error}")
            print("Falling back to local JSON files...")
            
            # If dataset_path is None, provide helpful error message
            if dataset_path is None:
                raise ValueError(
                    f"Could not load NSynth from Hugging Face.\n"
                    f"Error: {hf_error}\n\n"
                    "Solutions:\n"
                    "  1. Download NSynth dataset locally and provide --dataset_path:\n"
                    "     Download from: https://magenta.tensorflow.org/datasets/nsynth\n"
                    "     Then use: --dataset_path /path/to/NSynth --dataset_name nsynth\n\n"
                    "  2. Try installing an older datasets version:\n"
                    "     pip install 'datasets<2.14.0'\n\n"
                    "  3. Use the dataset in Hugging Face format 'hf:jg583/NSynth' (may have same issue)"
                )
            
            # Fall back to local JSON files
            train_json = os.path.join(dataset_path, "nsynth-train", "examples.json")
            valid_json = os.path.join(dataset_path, "nsynth-valid", "examples.json")
            test_json = os.path.join(dataset_path, "nsynth-test", "examples.json")
            
            # Try alternative paths
            if not os.path.exists(train_json):
                train_json = os.path.join(dataset_path, "train.json")
            if not os.path.exists(valid_json):
                valid_json = os.path.join(dataset_path, "valid.json")
            if not os.path.exists(test_json):
                test_json = os.path.join(dataset_path, "test.json")
            
            if not os.path.exists(train_json):
                raise FileNotFoundError(
                    f"NSynth train JSON not found. Tried:\n"
                    f"  - {os.path.join(dataset_path, 'nsynth-train', 'examples.json')}\n"
                    f"  - {os.path.join(dataset_path, 'train.json')}\n"
                    "Please download NSynth from: https://huggingface.co/datasets/jg583/NSynth\n"
                    "Or provide local JSON files with structure:\n"
                    "  NSynth/\n"
                    "    train.json (or nsynth-train/examples.json)\n"
                    "    test.json (or nsynth-test/examples.json)"
                )
            
            # Load JSON files
            def load_nsynth_json(json_path, base_dir):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                items = []
                for key, value in data.items():
                    # Map to label (prefer instrument_family_str, fallback to instrument_source_str)
                    label = value.get('instrument_family_str', 
                                    value.get('instrument_source_str', 'unknown'))
                    # Construct audio file path
                    audio_filename = f"{key}.wav"
                    audio_path = os.path.join(base_dir, audio_filename)
                    items.append({
                        'file_path': audio_path,
                        'label': label
                    })
                return items
            
            train_base = os.path.dirname(train_json)
            train_items = load_nsynth_json(train_json, train_base)
            train_df = pd.DataFrame(train_items)
            
            # Use validation as test if test not available
            if os.path.exists(test_json):
                test_base = os.path.dirname(test_json)
                test_items = load_nsynth_json(test_json, test_base)
                test_df = pd.DataFrame(test_items)
            elif os.path.exists(valid_json):
                valid_base = os.path.dirname(valid_json)
                test_items = load_nsynth_json(valid_json, valid_base)
                test_df = pd.DataFrame(test_items)
            else:
                # Split train into train/test if no test/valid available
                train_df, test_df = train_test_split(
                    train_df,
                    test_size=0.2,
                    stratify=train_df['label'],
                    random_state=42
                )
                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)
            
            print(f"NSynth dataset loaded from local JSON:")
            print(f"  Training set: {len(train_df)} files")
            print(f"  Test set: {len(test_df)} files")
            
            return train_df, test_df
    
    elif dataset_name == 'audioset':
        # AudioSet: Multi-label audio ontology dataset
        # We simplify by taking the first label only
        eval_csv = os.path.join(dataset_path, "eval_segments.csv")
        unbalanced_train_csv = os.path.join(dataset_path, "unbalanced_train_segments.csv")
        balanced_train_csv = os.path.join(dataset_path, "balanced_train_segments.csv")
        
        # Try to find CSV files
        train_csv = None
        if os.path.exists(unbalanced_train_csv):
            train_csv = unbalanced_train_csv
        elif os.path.exists(balanced_train_csv):
            train_csv = balanced_train_csv
        
        if not os.path.exists(eval_csv) or train_csv is None:
            raise FileNotFoundError(
                f"AudioSet CSV files not found.\n"
                f"Tried:\n"
                f"  - {eval_csv}\n"
                f"  - {unbalanced_train_csv}\n"
                f"  - {balanced_train_csv}\n"
                "Please download AudioSet from: https://research.google.com/audioset/download.html\n"
                "Expected CSV files with columns: YTID, start_seconds, end_seconds, positive_labels"
            )
        
        # Load CSV files (AudioSet format: YTID, start_seconds, end_seconds, positive_labels)
        # Note: AudioSet uses YouTube IDs, so file_path will need special handling
        # For now, we assume the user has downloaded and converted the audio files
        # and stored them in a directory structure
        
        def load_audioset_csv(csv_path, audio_base_dir):
            df = pd.read_csv(csv_path, skiprows=2, names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
            
            data = []
            for _, row in df.iterrows():
                # Take first label only (simplification for multi-label to single-label)
                labels = row['positive_labels'].split(',')
                first_label = labels[0].strip() if labels else 'unknown'
                
                # Construct file path (assuming audio files are named as YTID.wav or YTID_<start>_<end>.wav)
                # Try multiple naming conventions
                ytid = row['YTID']
                start = int(row['start_seconds'])
                end = int(row['end_seconds'])
                
                possible_paths = [
                    os.path.join(audio_base_dir, f"{ytid}.wav"),
                    os.path.join(audio_base_dir, f"{ytid}_{start}_{end}.wav"),
                    os.path.join(audio_base_dir, f"{ytid}_{start}.wav")
                ]
                
                file_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        file_path = path
                        break
                
                # If no file found, still add entry (will fail later if file doesn't exist)
                if file_path is None:
                    file_path = possible_paths[0]  # Use first convention as default
                
                data.append({
                    'file_path': file_path,
                    'label': first_label
                })
            
            return pd.DataFrame(data)
        
        # Determine audio base directory (try common locations)
        audio_base_dirs = [
            os.path.join(dataset_path, "audio"),
            os.path.join(dataset_path, "wav"),
            dataset_path
        ]
        
        audio_base_dir = None
        for base_dir in audio_base_dirs:
            if os.path.exists(base_dir):
                audio_base_dir = base_dir
                break
        
        if audio_base_dir is None:
            audio_base_dir = os.path.join(dataset_path, "audio")  # Default
        
        train_df = load_audioset_csv(train_csv, audio_base_dir)
        test_df = load_audioset_csv(eval_csv, audio_base_dir)
        
        print(f"AudioSet dataset loaded:")
        print(f"  Training set: {len(train_df)} files")
        print(f"  Test set: {len(test_df)} files")
        print(f"  Note: Multi-label simplified to first label only")
        print(f"  Audio base directory: {audio_base_dir}")
        
        return train_df, test_df
        
    elif dataset_name.startswith('hf:'):
        # Hugging Face dataset: format is 'hf:dataset_name' or 'hf:dataset_name/config_name'
        # Example: 'hf:common_voice' or 'hf:common_voice/en'
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for Hugging Face datasets. "
                "Install with: pip install datasets"
            )
        
        hf_dataset_name = dataset_name[3:]  # Remove 'hf:' prefix
        
        # Parse config if present (format: 'dataset_name/config')
        if '/' in hf_dataset_name:
            parts = hf_dataset_name.split('/', 1)
            hf_dataset_name = parts[0]
            config_name = parts[1]
        else:
            config_name = None
        
        print(f"Loading Hugging Face dataset: {hf_dataset_name}" + 
              (f" (config: {config_name})" if config_name else ""))
        
        # Load dataset from Hugging Face
        if config_name:
            dataset = load_dataset(hf_dataset_name, config_name)
        else:
            dataset = load_dataset(hf_dataset_name)
        
        # Expected structure:
        # - dataset['train'] and dataset['test'] (or 'validation')
        # - Each item has 'audio' key with 'path' or 'array' and 'sampling_rate'
        # - Each item has a label field (can be 'label', 'label_str', 'class', etc.)
        
        def extract_hf_data(split_name):
            """Extract file_path and label from HF dataset split."""
            if split_name not in dataset:
                # Try alternative names
                if split_name == 'test' and 'validation' in dataset:
                    split_name = 'validation'
                elif split_name == 'validation' and 'test' in dataset:
                    split_name = 'test'
                else:
                    return None
            
            split_data = dataset[split_name]
            data = []
            
            for item in split_data:
                # Extract audio path
                if 'audio' in item:
                    audio_info = item['audio']
                    if isinstance(audio_info, dict):
                        # Audio can be 'path' (string) or 'array' (numpy array)
                        if 'path' in audio_info:
                            file_path = audio_info['path']
                        elif 'array' in audio_info:
                            # For in-memory audio, we need to save it temporarily
                            # For now, skip or use a placeholder
                            print("Warning: In-memory audio arrays not supported. Skipping...")
                            continue
                        else:
                            print("Warning: Unknown audio format. Skipping...")
                            continue
                    else:
                        # Audio might be directly a path string
                        file_path = audio_info
                elif 'path' in item:
                    file_path = item['path']
                elif 'file' in item:
                    file_path = item['file']
                else:
                    print("Warning: Could not find audio path. Skipping...")
                    continue
                
                # Extract label
                label = None
                for label_key in ['label', 'label_str', 'class', 'category', 'genre', 'instrument_family_str']:
                    if label_key in item:
                        label = str(item[label_key])
                        break
                
                if label is None:
                    print(f"Warning: Could not find label for item. Available keys: {list(item.keys())}")
                    continue
                
                data.append({
                    'file_path': file_path,
                    'label': label
                })
            
            return pd.DataFrame(data)
        
        train_df = extract_hf_data('train')
        test_df = extract_hf_data('test') or extract_hf_data('validation')
        
        if train_df is None or len(train_df) == 0:
            raise ValueError(
                f"Could not load training split from Hugging Face dataset '{hf_dataset_name}'. "
                f"Available splits: {list(dataset.keys())}"
            )
        
        if test_df is None or len(test_df) == 0:
            # Split train into train/test if no test/validation available
            print("No test/validation split found. Splitting train set...")
            train_df, test_df = train_test_split(
                train_df,
                test_size=0.2,
                stratify=train_df['label'] if 'label' in train_df.columns else None,
                random_state=42
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
        
        print(f"Hugging Face dataset '{hf_dataset_name}' loaded:")
        print(f"  Training set: {len(train_df)} files")
        print(f"  Test set: {len(test_df)} files")
        
        return train_df, test_df
        
    else:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. "
            f"Supported: 'esc50', 'urbansound8k', 'gtzan', 'nsynth', 'audioset', or 'hf:dataset_name' for Hugging Face datasets"
        )


