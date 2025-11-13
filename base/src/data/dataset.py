class Dataset:
    def __init__(self, audio_text_pairs):
        self.audio_text_pairs = audio_text_pairs

    def load_data(self):
        # Load audio-text pairs from the provided source        
        return self.audio_text_pairs

    def get_item(self, index):
        # Retrieve a specific audio-text pair by index
        return self.audio_text_pairs[index] if index < len(self.audio_text_pairs) else None