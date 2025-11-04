import torch
from msclap import CLAP
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np

# --- 1. Configuração ---
K_VIZINHOS = 5
BATCH_SIZE = 16 # Processamento sequencial em lotes de 16 para economizar RAM

# Nossas 5 classes escolhidas para o teste
CLASSES_ESCOLHIDAS = [
    'dog', 
    'rain', 
    'helicopter', 
    'keyboard_typing', 
    'chirping_birds'
]

# Definir o dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# --- 2. Carregar o Modelo CLAP ---
print("Carregando modelo CLAP (versão 2023)...")
clap_model = CLAP(version='2023', use_cuda=(device == "cuda"))
print("Modelo CLAP carregado.")

# --- 3. Carregar e Filtrar o Dataset ---
print("Carregando dataset ESC-50 do Hugging Face...")
dataset = load_dataset("ashraq/esc50", trust_remote_code=True, split="train")

# Pegar o mapa de todas as 50 classes
labels_map = dataset.features['category'].names

# Converter nossos nomes de classes em IDs numéricos
try:
    chosen_class_ids = [labels_map.index(name) for name in CLASSES_ESCOLHIDAS]
except ValueError as e:
    print(f"ERRO: Uma das classes escolhidas não existe no dataset. {e}")
    exit()

print(f"Filtrando o dataset para manter apenas as classes: {CLASSES_ESCOLHIDAS}")

# Filtrar o dataset para conter APENAS as amostras das 5 classes
filtered_dataset = dataset.filter(
    lambda x: x['category'] in chosen_class_ids
)

print(f"Dataset filtrado! Total de amostras restantes: {len(filtered_dataset)}")

# Dividir o dataset filtrado em treino e teste
train_dataset = filtered_dataset.filter(lambda x: x['fold'] != 5)
test_dataset = filtered_dataset.filter(lambda x: x['fold'] == 5)

print(f"Total de amostras de treino (5 classes): {len(train_dataset)}")
print(f"Total de amostras de teste (5 classes): {len(test_dataset)}")

# --- 4. Geração Sequencial de Embeddings ---
# Esta função processa o dataset em pequenos lotes para não estourar a RAM

def generate_embeddings_sequentially(dataset, model, batch_size, desc):
    """
    Processa um dataset em mini-lotes para gerar embeddings
    e retorna os embeddings e labels como arrays numpy.
    """
    all_embeddings = []
    all_labels = []
    
    # tqdm mostra uma barra de progresso
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        # Pega um pequeno lote (batch) do dataset
        # O .select() é a forma eficiente de fatiar um dataset do Hugging Face
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # Extrai os áudios brutos e os rótulos deste lote
        audio_arrays = [x['audio']['array'] for x in batch]
        labels = [x['category'] for x in batch]
        
        # Gera embeddings para o lote
        # (batch_size na função get_audio_embeddings é interno do msclap,
        #  nosso batch_size aqui controla o uso de RAM)
        embeddings = model.get_audio_embeddings(
            audio_arrays, 
            resample=True,
            batch_size=batch_size 
        )
        
        all_embeddings.extend(embeddings)
        all_labels.extend(labels)
        
    return np.array(all_embeddings), np.array(all_labels)

# --- 5. Fase de Indexação (Treino) ---
print("Gerando embeddings para a base de dados de treino (Sequencial)...")
train_embeddings, train_labels = generate_embeddings_sequentially(
    train_dataset, 
    clap_model, 
    BATCH_SIZE,
    desc="Gerando embeddings de TREINO"
)
print(f"Embeddings de treino gerados. Shape: {train_embeddings.shape}")

# --- 6. Treinar o Classificador k-NN ---
print(f"Treinando o classificador k-NN (k={K_VIZINHOS}) com métrica 'cosine'...")
knn_classifier = KNeighborsClassifier(
    n_neighbors=K_VIZINHOS, 
    metric='cosine', 
    n_jobs=-1
)
knn_classifier.fit(train_embeddings, train_labels)
print("Classificador k-NN treinado.")

# --- 7. Fase de Avaliação (Teste) ---
print("Gerando embeddings para o conjunto de teste (Sequencial)...")
test_embeddings, test_labels_true = generate_embeddings_sequentially(
    test_dataset, 
    clap_model, 
    BATCH_SIZE,
    desc="Gerando embeddings de TESTE"
)
print(f"Embeddings de teste gerados. Shape: {test_embeddings.shape}")

print("Realizando predições (retrieval + votação k-NN)...")
test_labels_pred = knn_classifier.predict(test_embeddings)

# --- 8. Exibir os Resultados (Métricas) ---
print(f"\n--- Resultados da Avaliação no ESC-50 (Apenas 5 Classes) ---")

accuracy = accuracy_score(test_labels_true, test_labels_pred)
print(f"\nAcurácia Geral: {accuracy * 100:.2f}%")

print("\nRelatório de Classificação Detalhado:")
# Usamos os 'labels' e 'target_names' para mostrar um relatório
# limpo, apenas com as 5 classes que nos interessam.
report = classification_report(
    test_labels_true, 
    test_labels_pred, 
    labels=chosen_class_ids,
    target_names=CLASSES_ESCOLHIDAS,
    zero_division=0
)
print(report)

print("\nExemplo de 10 predições (Real vs. Previsto):")
# Pegamos as 10 primeiras amostras do dataset de teste filtrado
for i in range(min(10, len(test_labels_true))):
    # Usamos o 'labels_map' original para converter o ID numérico em nome
    real = labels_map[test_labels_true[i]]
    pred = labels_map[test_labels_pred[i]]
    print(f"  Amostra {i}: Real='{real}', Previsto='{pred}' {'(Correto)' if real == pred else '(Errado)'}")