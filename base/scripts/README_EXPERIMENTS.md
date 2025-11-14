Aqui está uma fusão dos dois READMEs, otimizada para a sua solicitação de um fluxo "Docker-first" que abre um terminal `bash`.

Este README unificado prioriza o framework de experimentação avançado (`run_experiment.py`) e trata o Docker como o método de setup principal.

-----

# ESC-50 Retrieval-Based Audio Classification

Este projeto implementa um sistema de classificação de áudio por recuperação usando:

  * **ESC-50 Dataset**: Dataset de 50 classes de sons ambientes.
  * **CLAPAudioEncoder**: Modelo CLAP pré-treinado para gerar embeddings de áudio.
  * **ChromaDB**: Banco de dados vetorial para busca por similaridade.
  * **Weighted k-NN**: Lógica de classificação usando os vizinhos mais próximos.

## 🚀 Features

  * **Framework de Experimentação**: Script `run_experiment.py` para testar diferentes configurações.
  * **Métricas de Distância**: Suporte para `cosine`, `l2` (Euclidiana) e `ip` (Produto Interno).
  * **Estratégias de Recuperação**: `basic` (k-NN simples) e `rerank` (k-NN com re-rankeamento de similaridade exata).
  * **Logging**: Salva métricas (Acurácia, F1-Score) e a Matriz de Confusão no **TensorBoard**.
  * **Suporte a Docker & GPU**: Ambiente Docker configurado para uso de GPU (recomendado).

## 🐋 Docker Quick Start (Recomendado)

Este guia foca em usar o Docker para garantir um ambiente consistente e com suporte a GPU. Nós vamos abrir um terminal `bash` interativo dentro do container.

### Pré-requisitos

  * [Docker](https://docs.docker.com/get-docker/) instalado.
  * [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) para suporte a GPU.
  * O dataset ESC-50.

### Passo 1: Clone o Repositório e Baixe o ESC-50

1.  Clone este repositório o repositorio do ESC-50:

    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DO_REPOSITORIO]

    git clone https://github.com/karolpiczak/ESC-50
    ```

Expected directory structure:
```
ESC-50/
  meta/
    esc50.csv
  audio/
    *.wav (2000 audio files)
```

2.  Baixe o dataset ESC-50 e coloque-o na raiz do projeto:

    ```bash
    # Você pode precisar descompactar
    wget https://github.com/karolpiczak/ESC-50/archive/master.zip
    unzip master.zip
    mv ESC-50-master ESC-50
    ```

    A estrutura de pastas esperada é:

    ```
    .
    ├── ESC-50/
    │   ├── meta/
    │   │   └── esc50.csv
    │   └── audio/
    │       └── *.wav
    ├── Dockerfile
    ├── scripts/
    └── ...
    ```

### Passo 2: Build a Imagem Docker

```bash
docker build -t audio-retrieval .
```

### Passo 3: Inicie um Container `bash` Interativo

Este comando inicia o container, mapeia suas pastas locais (dados, logs, e o banco de dados) para dentro dele, e abre um terminal `bash`:

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/chromadb:/app/chromadb \
    -v $(pwd)/logs:/app/logs \
    audio-retrieval \
    bash
```

  * `--gpus all`: Ativa a GPU NVIDIA.
  * `-it`: Modo interativo (para o terminal `bash`).
  * `--rm`: Remove o container ao sair.
  * `-v ...`: Mapeia volumes:
      * `ESC-50` -\> `/app/data/ESC-50` (Seu dataset)
      * `chromadb` -\> `/app/chromadb` (Onde o banco vetorial será salvo)
      * `logs` -\> `/app/logs` (Onde os logs do TensorBoard serão salvos)

Você agora deve estar em um prompt `bash` dentro do container (ex: `root@<id_container>:/app#`).

### Passo 4: (Dentro do Docker) Indexe o Dataset

Vamos indexar os *folds 1-4* usando a métrica `cosine`.

```bash
python scripts/index_esc50.py \
    --esc50_path ESC-50 \
    --distance_metric cosine \
    --chromadb_path /app/chromadb
```

*Note que estamos usando os caminhos de *dentro* do container.*

### Passo 5: (Dentro do Docker) Rode um Experimento

Agora, rode a avaliação no *fold 5* e salve os logs no TensorBoard.

```bash
python scripts/run_experiment.py \
    --esc50_path ESC-50 \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --chromadb_path /app/chromadb \
    --log_dir /app/logs \
    --use_gpu
```

### Passo 6: (Fora do Docker) Veja os Resultados

Abra um **novo terminal na sua máquina local (host)** e rode o TensorBoard. Como a pasta `./logs` foi mapeada, os logs aparecerão instantaneamente.

```bash
tensorboard --logdir ./logs
```

Abra `http://localhost:6006/` no seu navegador para ver a acurácia, F1-Score e a Matriz de Confusão.

-----

## 💻 Setup Local (Alternativo)

Se preferir não usar Docker:

1.  **Instale Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Baixe o ESC-50:** Siga o "Passo 1" da seção Docker para baixar o dataset.
3.  **Rode os Comandos:** Siga os "Passos 4 e 5" da seção Docker, mas usando os caminhos locais (ex: `--esc50_path ./ESC-50`, `--log_dir ./logs`).

-----

## 🛠️ Detalhes dos Scripts e Comandos

### 1\. Indexação (`scripts/index_esc50.py`)

Este script processa os áudios dos **folds 1-4** e os armazena no ChromaDB.

**Argumentos:**

  * `--esc50_path`: (Obrigatório) Caminho para a pasta `ESC-50`.
  * `--distance_metric`: Métrica a ser usada pelo ChromaDB. Cada métrica cria uma coleção separada (ex: `esc50_cosine`, `esc50_l2`).
      * `cosine` (default): Similaridade de Cosseno.
      * `l2`: Distância Euclidiana.
      * `ip`: Produto Interno (Dot Product).
  * `--chromadb_path`: Onde salvar o banco de dados persistente (default: `./chromadb`).
  * `--batch_size`: Tamanho do lote para inserção no ChromaDB (default: 100).

### 2\. Execução de Experimentos (`scripts/run_experiment.py`)

Este é o script principal para avaliar o desempenho no **fold 5**.

**Argumentos Principais:**

  * `--esc50_path`: (Obrigatório) Caminho para a pasta `ESC-50`.
  * `--model_name`: Modelo CLAP do Hugging Face (default: `laion/clap-htsat-unfused`).
  * `--distance_metric`: Métrica usada. **Deve ser a mesma usada na indexação.**
  * `--retrieval_strategy`:
      * `basic`: k-NN padrão (mais rápido, usa distâncias aproximadas do HNSW).
      * `rerank`: Pega 3\*k candidatos, extrai seus embeddings e recalcula a similaridade exata em PyTorch (mais lento, mais preciso).
  * `--k`: Número de vizinhos (default: 5).
  * `--log_dir`: Onde salvar logs do TensorBoard (default: `./logs`).
  * `--use_gpu`: Flag para tentar usar a GPU (se disponível).
  * `--save_confusion_matrix`: Caminho opcional para salvar a Matriz de Confusão como `.png`.

### 3\. Classificação Simples (`scripts/classify_audio.py`)

Útil para testar rapidamente um único arquivo de áudio.

```bash
python scripts/classify_audio.py \
    --audio_path /caminho/para/audio.wav \
    --chromadb_path ./chromadb \
    --k 5 \
    --show_neighbors
```

-----

## 🔬 Conceitos Técnicos

### Divisão do Dataset (Folds)

  * **Indexação (Treino):** Folds 1, 2, 3, 4 (\~1600 arquivos). Usados para construir o banco de dados vetorial.
  * **Teste (Avaliação):** Fold 5 (\~400 arquivos). Usado para testar o sistema.

### Estratégias de Recuperação

  * **Basic (`basic`):**
    1.  Consulta o ChromaDB pelos `k` vizinhos mais próximos.
    2.  Converte as distâncias (aproximadas pelo HNSW) em similaridade.
    3.  Realiza a votação ponderada.
  * **Re-rank (`rerank`):**
    1.  Consulta o ChromaDB por mais candidatos (ex: `k * 3`).
    2.  Extrai os embeddings *reais* desses `k*3` candidatos.
    3.  Calcula a similaridade *exata* em PyTorch entre a query e os candidatos.
    4.  Seleciona os `k` melhores com base na similaridade exata.
    5.  Realiza a votação ponderada.

### Métricas de Distância

  * `cosine` (Distância Cosseno): `dist = 1 - sim`. Foca na direção (ângulo) dos vetores. Ideal para embeddings normalizados como os do CLAP.
  * `l2` (Euclidiana): Distância em linha reta.
  * `ip` (Produto Interno): `dist = -sim`. Similar ao cosseno para vetores normalizados, mas geralmente mais rápido.