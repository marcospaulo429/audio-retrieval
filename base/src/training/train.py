import os
import torch
from torch.utils.data import DataLoader
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from data.dataset import Dataset
from training.loss import contrastive_loss
from training.config import Config

def train_model():
    config = Config()
    audio_encoder = AudioEncoder(config.audio_model_path)
    text_encoder = TextEncoder(config.text_model_path)

    dataset = Dataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(list(audio_encoder.parameters()) + list(text_encoder.parameters()), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        audio_encoder.train()
        text_encoder.train()
        total_loss = 0

        for audio, text in dataloader:
            optimizer.zero_grad()
            audio_embedding = audio_encoder.encode_audio(audio)
            text_embedding = text_encoder.encode_text(text)

            loss = contrastive_loss(audio_embedding, text_embedding)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

def validate_model():
    # Validation logic can be implemented here
    pass

if __name__ == "__main__":
    train_model()