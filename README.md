# Audio Encoder Application

This project is an encoder-only audio processing application designed to extract mel spectrograms from audio files stored in `.tar` archives located in the `/data` directory. The extracted mel spectrograms can then be used for various audio processing tasks, including training and inference with a transformer model.

## Project Structure

```
audio-encoder-app
├── src
│   ├── config               # Configuration files for the application
│   ├── data                 # Data processing scripts
│   ├── datasets             # Dataset classes for handling mel spectrograms
│   ├── models               # Model definitions, including layers and transformer
│   ├── train.py             # Main training script
│   └── infer.py             # Inference script for generating embeddings
├── scripts                  # Scripts for data preparation, training, and evaluation
├── data                     # Directory for storing data, mel spectrograms, and logs
├── tests                    # Unit tests for the application
├── pyproject.toml          # Project configuration file
└── requirements.txt         # Required Python packages
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd audio-encoder-app
pip install -r requirements.txt
```

## Usage

### Extracting Mel Spectrograms

To extract mel spectrograms from the `.tar` files in the `/data` directory, run the following command:

```bash
python scripts/prepare_data.py
```

### Training the Encoder Model

To train the encoder model, use the following command:

```bash
python src/train.py
```

### Inference

To generate embeddings from new audio data, run:

```bash
python src/infer.py
```

## Testing

To run the unit tests for the dataset and model functionality, execute:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.