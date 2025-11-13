def save_embedding(embedding, file_path):
    import numpy as np
    np.save(file_path, embedding)

def load_embedding(file_path):
    import numpy as np
    return np.load(file_path)