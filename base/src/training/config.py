# Configuration settings for training the audio-text retrieval model

class Config:
    def __init__(self):
        # Model hyperparameters
        self.audio_embedding_dim = 256
        self.text_embedding_dim = 256
        self.joint_embedding_dim = 512
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        
        # Paths
        self.audio_model_path = 'models/audio_encoder.pth'
        self.text_model_path = 'models/text_encoder.pth'
        self.joint_model_path = 'models/joint_embedding.pth'
        
        # Data paths
        self.train_data_path = 'data/train_data.csv'
        self.val_data_path = 'data/val_data.csv'
        
        # Logging
        self.log_interval = 10
        self.save_model_interval = 5
        
    def display(self):
        print("Configuration Settings:")
        print(f"Audio Embedding Dimension: {self.audio_embedding_dim}")
        print(f"Text Embedding Dimension: {self.text_embedding_dim}")
        print(f"Joint Embedding Dimension: {self.joint_embedding_dim}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Audio Model Path: {self.audio_model_path}")
        print(f"Text Model Path: {self.text_model_path}")
        print(f"Joint Model Path: {self.joint_model_path}")
        print(f"Training Data Path: {self.train_data_path}")
        print(f"Validation Data Path: {self.val_data_path}")
        print(f"Log Interval: {self.log_interval}")
        print(f"Save Model Interval: {self.save_model_interval}")