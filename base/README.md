# Audio-Text Retrieval Project

This project implements an audio-text retrieval system that utilizes audio-text embedding pairs in a shared latent space to generate similarities. The system allows for retrieving audio files through text queries and retrieving text descriptions through audio queries.

## Project Structure

The project is organized into the following directories and files:

- **src/**: Contains the main source code for the audio-text retrieval system.
  - **models/**: Implements the models for encoding audio and text inputs.
    - `audio_encoder.py`: Contains the `AudioEncoder` class for encoding audio inputs.
    - `text_encoder.py`: Contains the `TextEncoder` class for encoding text inputs.
    - `joint_embedding.py`: Contains the `JointEmbedding` class for combining audio and text embeddings.
  - **data/**: Handles data loading, preprocessing, and augmentation.
    - `dataset.py`: Defines the `Dataset` class for managing audio-text pairs.
    - `preprocessing.py`: Contains functions for preprocessing audio and text data.
    - `augmentation.py`: Contains functions for augmenting audio data.
  - **training/**: Contains the training loop and loss functions.
    - `train.py`: Implements the main training loop for the model.
    - `loss.py`: Defines the loss functions used during training.
    - `config.py`: Contains configuration settings for training.
  - **retrieval/**: Implements retrieval functions for audio and text.
    - `audio_to_text.py`: Functions for retrieving text from audio queries.
    - `text_to_audio.py`: Functions for retrieving audio from text queries.
    - `similarity.py`: Functions for calculating similarities between embeddings.
  - **utils/**: Contains utility functions for handling embeddings and metrics.
    - `embedding_utils.py`: Utility functions for saving and loading embeddings.
    - `metrics.py`: Functions for evaluating model performance.
  - `app.py`: Main entry point for the application.

- **tests/**: Contains unit tests for the various components of the project.
  - `test_models.py`: Unit tests for model components.
  - `test_retrieval.py`: Unit tests for retrieval functions.
  - `test_utils.py`: Unit tests for utility functions.

- **configs/**: Contains configuration settings in YAML format.
  - `config.yaml`: Configuration settings for the model and training process.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd audio-text-retrieval
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the training settings in `configs/config.yaml` as needed.

## Usage

To train the model, run the following command:
```
python src/training/train.py
```

To retrieve audio from a text query, use:
```
python src/retrieval/text_to_audio.py --query "<your text query>"
```

To retrieve text from an audio query, use:
```
python src/retrieval/audio_to_text.py --audio "<path to audio file>"
```

## Overview of the Architecture

The architecture consists of separate encoders for audio and text, which generate embeddings that are then combined in a shared latent space. The retrieval functions utilize these embeddings to compute similarities and facilitate the retrieval process.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.