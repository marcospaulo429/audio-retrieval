# ESC-50 Retrieval-Based Audio Classification

This pipeline implements a retrieval-based audio classification system using:
- **ESC-50 Dataset**: Environmental Sound Classification dataset with 50 classes
- **CLAPAudioEncoder**: Pre-trained CLAP model for audio embeddings
- **ChromaDB**: Vector database for similarity search
- **Weighted k-NN**: Classification using weighted nearest neighbors

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download ESC-50 Dataset

Clone or download the ESC-50 dataset from:
```
https://github.com/karolpiczak/ESC-50
```

Expected directory structure:
```
ESC-50/
  meta/
    esc50.csv
  audio/
    *.wav (2000 audio files)
```

## Usage

### Step 1: Index the Dataset (Folds 1-4)

Index ESC-50 audio files from folds 1-4 into ChromaDB:

```bash
python scripts/index_esc50.py --esc50_path ESC-50
```

Optional arguments:
- `--batch_size`: Batch size for ChromaDB insertion (default: 100)
- `--collection_name`: ChromaDB collection name (default: "esc50")
- `--chromadb_path`: Path to persist ChromaDB data (default: "./chromadb")

Example:
```bash
python scripts/index_esc50.py \
    --esc50_path /data/ESC-50 \
    --batch_size 100 \
    --collection_name esc50 \
    --chromadb_path ./chromadb
```

**Note:** ChromaDB data is persisted to disk, so the collection will be available for subsequent runs. Make sure to use the same `--chromadb_path` in all scripts.

This will:
1. Load the ESC-50 metadata from `meta/esc50.csv`
2. Filter files from **folds 1-4** (indexing set, ~1600 files)
3. Process audio files with CLAPAudioEncoder
4. Store embeddings in ChromaDB with metadata (label, filename)

**Note:** Files from fold 5 are held out for testing and are NOT indexed.

### Step 2: Evaluate on Test Set (Fold 5)

Run evaluation on the test set and compute metrics:

```bash
python scripts/evaluate_esc50.py --esc50_path /path/to/ESC-50
```

Optional arguments:
- `--k`: Number of nearest neighbors (default: 5)
- `--collection_name`: ChromaDB collection name (default: "esc50")
- `--chromadb_path`: Path to ChromaDB persistent storage (default: "./chromadb")
- `--save_confusion_matrix`: Path to save confusion matrix figure (optional)

Example:
```bash
python scripts/evaluate_esc50.py \
    --esc50_path /data/ESC-50 \
    --k 5 \
    --chromadb_path ./chromadb \
    --save_confusion_matrix confusion_matrix.png
```

**Important:** Use the same `--chromadb_path` as used during indexing!

This will:
1. Load test set files from **fold 5** (~400 files)
2. Run inference on all test files using the indexed database
3. Calculate and display:
   - Overall accuracy
   - Classification report (Precision/Recall/F1 per class)
   - Confusion matrix (visualized with seaborn)

### Step 3: Classify Individual Audio (Optional)

Classify a single 5-second audio clip:

```bash
python scripts/classify_audio.py --audio_path /path/to/audio.wav
```

Optional arguments:
- `--k`: Number of nearest neighbors (default: 5)
- `--collection_name`: ChromaDB collection name (default: "esc50")
- `--chromadb_path`: Path to ChromaDB persistent storage (default: "./chromadb")
- `--show_neighbors`: Show k nearest neighbors details

Example:
```bash
python scripts/classify_audio.py \
    --audio_path test_audio.wav \
    --k 10 \
    --chromadb_path ./chromadb \
    --show_neighbors
```

**Important:** Use the same `--chromadb_path` as used during indexing!

## How It Works

### Dataset Split

The ESC-50 dataset is split using the `fold` column:
- **Indexing Set (Folds 1-4)**: ~1600 files used to build the ChromaDB vector database
- **Test Set (Fold 5)**: ~400 files held out for evaluation

This ensures the test set is completely unseen during indexing, providing a fair evaluation.

### Indexing Pipeline

1. **Load Dataset**: Reads `meta/esc50.csv` to get filename-to-category mappings
2. **Filter by Fold**: Selects only files from folds 1-4 for indexing
3. **Process Audio**: For each audio file:
   - Load with librosa at 48kHz (CLAP requirement)
   - Process with ClapProcessor
   - Generate embedding using CLAPAudioEncoder
   - Store in ChromaDB with metadata (label, filename)

### Classification Pipeline

1. **Load Audio**: Load the query audio file
2. **Generate Embedding**: Create embedding using CLAPAudioEncoder
3. **Query ChromaDB**: Find k nearest neighbors using cosine similarity
4. **Weighted k-NN**: 
   - For each neighbor, calculate similarity = 1.0 - distance
   - Sum similarities per class
   - Predict class with highest score

### Evaluation Pipeline

1. **Load Test Set**: Load files from fold 5 with their true labels
2. **Run Inference**: Classify each test file using the classification pipeline
3. **Calculate Metrics**:
   - **Accuracy**: Overall classification accuracy
   - **Classification Report**: Precision, Recall, F1-score per class
   - **Confusion Matrix**: Visual representation of classification performance

## Model Details

### CLAPAudioEncoder

- **Model**: `laion/clap-htsat-unfused` (default)
- **Embedding Dimension**: 512
- **Frozen**: True (for inference)
- **Normalization**: L2 normalized embeddings

The encoder uses CLAP's `get_audio_features()` API which handles internal audio processing.

## ESC-50 Classes

The dataset contains 50 environmental sound classes:
- Animals (dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow)
- Natural soundscapes (rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm)
- Human sounds (crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking sipping)
- Interior/domestic sounds (door knock, mouse click, keyboard typing, door wood creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking)
- Exterior/urban sounds (helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw)

## Notes

- Audio files should be 5 seconds long (ESC-50 standard)
- The model expects 48kHz sample rate
- ChromaDB uses cosine similarity for nearest neighbor search
- The weighted k-NN uses similarity scores (1 - distance) as weights

