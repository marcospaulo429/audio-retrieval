# Framework de Experimentação - Classificação de Áudio por Recuperação

Este projeto implementa um sistema **genérico e flexível** de classificação de áudio por recuperação usando:

* **Múltiplos Encoders**: CLAP, Wav2Vec2, HuBERT, AST, CNN Transformer
* **Múltiplos Datasets**: ESC-50, UrbanSound8K, GTZAN, NSynth, AudioSet, e qualquer dataset do Hugging Face
* **ChromaDB**: Banco de dados vetorial para busca por similaridade
* **Weighted k-NN**: Lógica de classificação usando os vizinhos mais próximos
* **TensorBoard**: Logging de métricas e visualizações

## 🚀 Features

* **Framework Genérico**: Suporte para diferentes encoders e datasets via argumentos de linha de comando
* **Factory Functions**: Abstração em `src/utils.py` para carregar modelos e datasets facilmente
* **Hugging Face Integration**: Suporte nativo para datasets do Hugging Face (formato `hf:dataset_name`)
* **Métricas de Distância**: Suporte para `cosine`, `l2` (Euclidiana) e `ip` (Produto Interno)
* **Estratégias de Recuperação**: `basic` (k-NN simples) e `rerank` (k-NN com re-rankeamento de similaridade exata)
* **Logging**: Salva métricas (Acurácia, F1-Score) e a Matriz de Confusão no **TensorBoard**
* **Suporte a Docker & GPU**: Ambiente Docker configurado para uso de GPU (recomendado)

## 📋 Arquitetura Refatorada

### Estrutura de Arquivos

```
base/
├── src/
│   ├── utils.py                    # ✨ Funções factory para modelos e datasets
│   └── models/
│       └── audio_encoder.py        # 5 classes de encoders
├── scripts/
│   ├── index_esc50.py              # Indexação genérica
│   ├── run_experiment.py           # Experimentos genéricos
│   ├── README_EXPERIMENTS.md       # 📝 Este arquivo
│   └── README_HUGGINGFACE.md       # 📝 Guia para datasets do Hugging Face
```

### Funções Factory (`src/utils.py`)

#### 1. `load_model_and_processor(encoder_name, model_name_or_path, device)`

Carrega modelo e processor baseado no nome do encoder.

**Encoders Suportados:**
- `clap`: CLAPAudioEncoder (sample_rate: 48000, input_key: 'input_features')
- `wav2vec2`: Wav2Vec2AudioEncoder (sample_rate: 16000, input_key: 'input_values')
- `hubert`: HuBERTAudioEncoder (sample_rate: 16000, input_key: 'input_values')
- `ast`: AudioSpectrogramTransformer (sample_rate: 16000, input_key: 'input_values')
- `custom_cnn`: CNNTransformerAudioEncoder (sample_rate: 16000, input_key: 'raw_tensor', processor: None)

**Retorna:** `(model, processor, target_sample_rate, model_input_key)`

#### 2. `load_dataset_splits(dataset_name, dataset_path)`

Carrega e divide dataset em treino/teste.

**Datasets Suportados:**
- `esc50`: Folds 1-4 (treino), Fold 5 (teste)
- `urbansound8k`: Folds 1-9 (treino), Fold 10 (teste)
- `gtzan`: 80% treino, 20% teste (estratificado)
- `nsynth`: Carrega do Hugging Face ou arquivos locais JSON
- `audioset`: Eval (teste), Unbalanced/Balanced train (treino)
- `hf:dataset_name`: Qualquer dataset do Hugging Face (veja README_HUGGINGFACE.md)

**Retorna:** `(train_df, test_df)` com colunas: `file_path`, `label`

## 🐋 Docker Quick Start (Recomendado)

### Pré-requisitos

* [Docker](https://docs.docker.com/get-docker/) instalado
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) para suporte a GPU
* Dataset escolhido

### Passo 1: Build da Imagem Docker

```bash
docker build -t audio-retrieval .
```

### Passo 2: Inicie Container Interativo

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/ESC-50:/app/data/ESC-50 \
    -v $(pwd)/chromadb:/app/chromadb \
    -v $(pwd)/logs:/app/logs \
    audio-retrieval \
    bash
```

## 📊 Guia de Uso por Dataset

### 1. ESC-50 (Environmental Sound Classification)

**Download:** https://github.com/karolpiczak/ESC-50

**Estrutura:**
```
ESC-50/
  meta/
    esc50.csv
  audio/
    *.wav (2000 arquivos)
```

**Indexar:**
```bash
python scripts/index_esc50.py \
    --dataset_path /app/data/ESC-50 \
    --dataset_name esc50 \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --chromadb_path /app/chromadb \
    --use_gpu
```

**Experimentar:**
```bash
python scripts/run_experiment.py \
    --dataset_path /app/data/ESC-50 \
    --dataset_name esc50 \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --k 5 \
    --chromadb_path /app/chromadb \
    --log_dir /app/logs \
    --use_gpu
```

**Características:**
- 50 classes de sons ambientais
- 2000 arquivos de 5 segundos
- Split: Folds 1-4 (treino), Fold 5 (teste)

---

### 2. UrbanSound8K (Urban Sound Classification)

**Download:** https://urbansounddataset.weebly.com/

**Estrutura:**
```
UrbanSound8K/
  metadata/
    UrbanSound8K.csv
  audio/
    fold1/
    fold2/
    ...
    fold10/
```

**Indexar:**
```bash
python scripts/index_esc50.py \
    --dataset_path /app/data/UrbanSound8K \
    --dataset_name urbansound8k \
    --encoder_name wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --distance_metric cosine \
    --chromadb_path /app/chromadb \
    --use_gpu
```

**Experimentar:**
```bash
python scripts/run_experiment.py \
    --dataset_path /app/data/UrbanSound8K \
    --dataset_name urbansound8k \
    --encoder_name wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --chromadb_path /app/chromadb \
    --log_dir /app/logs \
    --use_gpu
```

**Características:**
- 10 classes de sons urbanos
- ~9000 arquivos
- Split: Folds 1-9 (treino), Fold 10 (teste)

---

### 3. GTZAN (Music Genre Classification)

**Download:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

**Estrutura:**
```
GTZAN/
  genres_original/
    blues/
    classical/
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/
```

**Indexar:**
```bash
python scripts/index_esc50.py \
    --dataset_path /app/data/GTZAN \
    --dataset_name gtzan \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --chromadb_path /app/chromadb \
    --use_gpu
```

**Experimentar:**
```bash
python scripts/run_experiment.py \
    --dataset_path /app/data/GTZAN \
    --dataset_name gtzan \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --chromadb_path /app/chromadb \
    --log_dir /app/logs \
    --use_gpu
```

**Características:**
- 10 gêneros musicais
- 1000 arquivos de 30 segundos cada
- Cada arquivo é dividido em 10 chunks de 3 segundos
- Split: 80% treino, 20% teste (estratificado)

**Nota:** O DataFrame inclui coluna `chunk_id` (0-9) para identificar qual chunk foi usado.

---

### 4. NSynth (Musical Notes Dataset)

**Opção A: Via Hugging Face (Recomendado)**

```bash
# Indexar (sem dataset_path necessário)
python scripts/index_esc50.py \
    --dataset_path hf:jg583/NSynth \
    --dataset_name nsynth \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --chromadb_path /app/chromadb \
    --use_gpu

# Experimentar
python scripts/run_experiment.py \
    --dataset_name hf:jg583/NSynth \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --chromadb_path /app/chromadb \
    --log_dir /app/logs \
    --use_gpu
```

**Opção B: Arquivos Locais**

**Estrutura:**
```
NSynth/
  nsynth-train/
    examples.json
    audio/
      *.wav
  nsynth-test/
    examples.json
    audio/
      *.wav
```

```bash
python scripts/index_esc50.py \
    --dataset_path /app/data/NSynth \
    --dataset_name nsynth \
    --encoder_name clap \
    --distance_metric cosine \
    --use_gpu
```

**Características:**
- ~300k notas musicais
- 11 famílias de instrumentos
- Label: `instrument_family_str` (bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal)

---

### 5. AudioSet (Google Research)

**Download:** https://research.google.com/audioset/download.html

**Estrutura:**
```
AudioSet/
  eval_segments.csv
  unbalanced_train_segments.csv  # ou balanced_train_segments.csv
  audio/
    YTID.wav  # ou YTID_start_end.wav
    ...
```

**Indexar:**
```bash
python scripts/index_esc50.py \
    --dataset_path /app/data/AudioSet \
    --dataset_name audioset \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --chromadb_path /app/chromadb \
    --use_gpu
```

**Experimentar:**
```bash
python scripts/run_experiment.py \
    --dataset_path /app/data/AudioSet \
    --dataset_name audioset \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --chromadb_path /app/chromadb \
    --log_dir /app/logs \
    --use_gpu
```

**Características:**
- 527 classes de sons
- Multi-label (simplificado para primeira label)
- Segmentos de 10 segundos de vídeos do YouTube
- Split: Eval (teste), Unbalanced/Balanced train (treino)

**⚠️ Importante:** Você precisa baixar e converter os vídeos do YouTube para arquivos de áudio. O código procura arquivos com padrões:
- `{YTID}.wav`
- `{YTID}_{start}_{end}.wav`
- `{YTID}_{start}.wav`

---

### 6. Datasets do Hugging Face (Genérico)

**Formato:** `hf:dataset_name` ou `hf:dataset_name/config_name`

**Exemplo 1: NSynth**
```bash
python scripts/index_esc50.py \
    --dataset_name hf:jg583/NSynth \
    --encoder_name clap \
    --distance_metric cosine \
    --use_gpu
```

**Exemplo 2: Common Voice (com config)**
```bash
python scripts/index_esc50.py \
    --dataset_name hf:mozilla-foundation/common_voice_13_0/en \
    --encoder_name wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --distance_metric cosine \
    --use_gpu
```

**Exemplo 3: FLEURS**
```bash
python scripts/index_esc50.py \
    --dataset_name hf:google/fleurs/en_us \
    --encoder_name wav2vec2 \
    --distance_metric cosine \
    --use_gpu
```

**Características:**
- **Sem necessidade de `--dataset_path`**: O dataset é baixado automaticamente
- **Download automático**: Primeira execução baixa o dataset
- **Suporte a configs**: Use `hf:dataset/config` para datasets com múltiplas configurações

**📖 Para mais detalhes:** Veja [README_HUGGINGFACE.md](README_HUGGINGFACE.md)

---

## 🛠️ Detalhes dos Scripts

### 1. Indexação (`scripts/index_esc50.py`)

**Argumentos:**

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--dataset_path` | str | `None` | Caminho para dataset (opcional para HF) |
| `--dataset_name` | str | `esc50` | Nome do dataset ou `hf:dataset_name` |
| `--encoder_name` | str | `clap` | Encoder (`clap`, `wav2vec2`, `hubert`, `ast`, `custom_cnn`) |
| `--model_name` | str | `laion/clap-htsat-unfused` | Modelo Hugging Face |
| `--distance_metric` | str | `cosine` | Métrica (`cosine`, `l2`, `ip`) |
| `--chromadb_path` | str | `./chromadb` | Caminho para ChromaDB |
| `--batch_size` | int | `100` | Tamanho do lote |
| `--use_gpu` | flag | - | Usar GPU se disponível |

**Exemplos de Collection Names:**
- `esc50_clap_cosine`
- `gtzan_wav2vec2_l2`
- `hf:jg583/NSynth_clap_cosine`

### 2. Experimentos (`scripts/run_experiment.py`)

**Argumentos:**

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--dataset_path` | str | `None` | Caminho para dataset (opcional para HF) |
| `--dataset_name` | str | `esc50` | Nome do dataset (deve ser o mesmo da indexação) |
| `--encoder_name` | str | `clap` | Encoder (deve ser o mesmo da indexação) |
| `--model_name` | str | `laion/clap-htsat-unfused` | Modelo Hugging Face |
| `--distance_metric` | str | `cosine` | Métrica (deve ser a mesma da indexação) |
| `--retrieval_strategy` | str | `basic` | Estratégia (`basic`, `rerank`) |
| `--k` | int | `5` | Número de vizinhos |
| `--chromadb_path` | str | `./chromadb` | Caminho para ChromaDB |
| `--log_dir` | str | `./logs` | Diretório para logs TensorBoard |
| `--use_gpu` | flag | - | Usar GPU se disponível |
| `--save_confusion_matrix` | str | `None` | Caminho opcional para salvar matriz |

**⚠️ Importante:** Os argumentos `--dataset_name`, `--encoder_name` e `--distance_metric` devem ser **idênticos** aos usados na indexação!

## 🔬 Conceitos Técnicos

### Encoders Disponíveis

| Encoder | Sample Rate | Input Key | Processor | Melhor Para |
|---------|-------------|-----------|-----------|-------------|
| CLAP | 48000 | `input_features` | ClapProcessor | Audio-text retrieval |
| Wav2Vec2 | 16000 | `input_values` | Wav2Vec2Processor | General audio tasks |
| HuBERT | 16000 | `input_values` | HubertProcessor | Speech understanding |
| AST | 16000 | `input_values` | AutoFeatureExtractor | Audio classification |
| Custom CNN | 16000 | `raw_tensor` | None | Lightweight, fast |

### Divisão de Datasets

| Dataset | Treino | Teste | Método |
|---------|--------|-------|--------|
| ESC-50 | Folds 1-4 (~1600) | Fold 5 (~400) | Pre-definido |
| UrbanSound8K | Folds 1-9 (~8000) | Fold 10 (~1000) | Pre-definido |
| GTZAN | 80% | 20% | Stratified split |
| NSynth | Train split | Test/Valid split | Pre-definido ou split |
| AudioSet | Unbalanced/Balanced train | Eval | Pre-definido |
| Hugging Face | Train split | Test/Valid split ou 80/20 | Automático |

### Estratégias de Recuperação

**Basic (`basic`):**
1. Consulta ChromaDB pelos `k` vizinhos mais próximos (HNSW aproximado)
2. Converte distâncias em similaridade
3. Realiza votação ponderada

**Rerank (`rerank`):**
1. Consulta ChromaDB por `3*k` candidatos (HNSW rápido)
2. Extrai embeddings reais dos candidatos
3. Calcula similaridade **exata** em PyTorch (GPU)
4. Seleciona top `k` baseado na similaridade exata
5. Realiza votação ponderada

**Comparação:**
- `basic`: Mais rápido, usa distâncias aproximadas
- `rerank`: Mais preciso, usa similaridade exata (mais lento)

### Métricas de Distância

| Métrica | Conversão | Melhor Para |
|---------|-----------|-------------|
| `cosine` | `similarity = 1 - distance` | Embeddings normalizados (CLAP) |
| `l2` | `similarity = 1 / (1 + distance)` | Comparações sensíveis à magnitude |
| `ip` | `similarity = -distance` | Embeddings não normalizados |

## 📊 Exemplos de Experimentos

### Comparar Encoders no ESC-50

```bash
# Indexar com diferentes encoders
for encoder in clap wav2vec2 hubert; do
    python scripts/index_esc50.py \
        --dataset_path ESC-50 \
        --dataset_name esc50 \
        --encoder_name $encoder \
        --model_name laion/clap-htsat-unfused \
        --distance_metric cosine \
        --use_gpu
done

# Avaliar cada um
for encoder in clap wav2vec2 hubert; do
    python scripts/run_experiment.py \
        --dataset_path ESC-50 \
        --dataset_name esc50 \
        --encoder_name $encoder \
        --model_name laion/clap-htsat-unfused \
        --distance_metric cosine \
        --retrieval_strategy basic \
        --log_dir ./logs/${encoder} \
        --use_gpu
done
```

### Comparar Datasets

```bash
# Indexar diferentes datasets
python scripts/index_esc50.py --dataset_path ESC-50 --dataset_name esc50 --encoder_name clap --use_gpu
python scripts/index_esc50.py --dataset_path GTZAN --dataset_name gtzan --encoder_name clap --use_gpu
python scripts/index_esc50.py --dataset_name hf:jg583/NSynth --encoder_name clap --use_gpu

# Avaliar cada um
python scripts/run_experiment.py --dataset_path ESC-50 --dataset_name esc50 --encoder_name clap --log_dir ./logs/esc50 --use_gpu
python scripts/run_experiment.py --dataset_path GTZAN --dataset_name gtzan --encoder_name clap --log_dir ./logs/gtzan --use_gpu
python scripts/run_experiment.py --dataset_name hf:jg583/NSynth --encoder_name clap --log_dir ./logs/nsynth --use_gpu
```

### Comparar Estratégias de Recuperação

```bash
# Indexar uma vez
python scripts/index_esc50.py \
    --dataset_path ESC-50 \
    --dataset_name esc50 \
    --encoder_name clap \
    --distance_metric cosine \
    --use_gpu

# Testar ambas estratégias
for strategy in basic rerank; do
    python scripts/run_experiment.py \
        --dataset_path ESC-50 \
        --dataset_name esc50 \
        --encoder_name clap \
        --distance_metric cosine \
        --retrieval_strategy $strategy \
        --log_dir ./logs/${strategy} \
        --use_gpu
done
```

## 🔧 Extensibilidade

### Adicionar Novo Encoder

1. Adicione a classe do encoder em `src/models/audio_encoder.py` (se ainda não existir)
2. Adicione caso em `load_model_and_processor()` em `src/utils.py`:

```python
elif encoder_name == 'novo_encoder':
    from transformers import NovoProcessor
    
    processor = NovoProcessor.from_pretrained(model_name_or_path)
    model = NovoAudioEncoder(model_name=model_name_or_path, ...)
    target_sample_rate = 16000  # ou outro valor
    model_input_key = 'input_values'  # ou outra chave
```

### Adicionar Novo Dataset Local

1. Adicione caso em `load_dataset_splits()` em `src/utils.py`:

```python
elif dataset_name == 'novo_dataset':
    # Carregar metadata
    df = pd.read_csv(...)
    
    # Criar file_path e label
    df['file_path'] = ...
    df['label'] = ...
    
    # Dividir treino/teste
    train_df = df[...]
    test_df = df[...]
    
    return train_df[['file_path', 'label']], test_df[['file_path', 'label']]
```

### Usar Dataset do Hugging Face

Simplesmente use o formato `hf:dataset_name`:

```bash
python scripts/index_esc50.py \
    --dataset_name hf:seu-usuario/nome-do-dataset \
    --encoder_name clap \
    --use_gpu
```

**📖 Para mais detalhes sobre formato requerido:** Veja [README_HUGGINGFACE.md](README_HUGGINGFACE.md)

## 📝 Notas Importantes

1. **Consistência de Argumentos**: Use os mesmos `--dataset_name`, `--encoder_name` e `--distance_metric` na indexação e no experimento
2. **Collection Names**: Gerados automaticamente como `{dataset}_{encoder}_{metric}`
3. **GPU**: Use `--use_gpu` para acelerar processamento (especialmente útil para rerank)
4. **Sample Rates**: Cada encoder tem seu sample rate específico (gerenciado automaticamente)
5. **Custom CNN**: Não requer processor externo, processa áudio bruto internamente
6. **Hugging Face**: Não requer `--dataset_path`, o dataset é baixado automaticamente

## 🐛 Troubleshooting

**Erro: "Collection not found"**
- Certifique-se de ter indexado o dataset primeiro
- Verifique se os argumentos `--dataset_name`, `--encoder_name` e `--distance_metric` são idênticos

**Erro: "CUDA out of memory"**
- Reduza `--batch_size` na indexação
- Use CPU: remova `--use_gpu`

**Erro: "Audio file not found"**
- Verifique se `--dataset_path` está correto
- Verifique estrutura do dataset (meta/, audio/, etc.)

**Erro: "Could not find label for item" (Hugging Face)**
- Verifique se o dataset tem um dos campos de label suportados
- Veja README_HUGGINGFACE.md para formato requerido

**Erro: "datasets library is required"**
- Instale: `pip install datasets`

## 🤖 Script de Experimentos Automatizado

### `run_experiments.sh` - Execução em Lote

Para executar experimentos em múltiplos datasets de forma automatizada, use o script `run_experiments.sh`:

**Características:**
- ✅ Configuração modular no topo do arquivo
- ✅ Suporte a múltiplos datasets (locais e Hugging Face)
- ✅ Indexação e avaliação automáticas
- ✅ Logging organizado por dataset
- ✅ Resumo de resultados ao final

**Uso Básico:**

```bash
# 1. Edite as configurações no topo do script
nano scripts/run_experiments.sh

# 2. Execute o script
./scripts/run_experiments.sh
```

**Configuração Modular:**

Todas as configurações estão no topo do arquivo `run_experiments.sh`:

```bash
# Model Configuration
ENCODER_NAME="clap"
MODEL_NAME="laion/clap-htsat-unfused"

# Retrieval Configuration
DISTANCE_METRIC="cosine"
RETRIEVAL_STRATEGY="basic"
K_NEIGHBORS=5

# Dataset Paths
DATASET_BASE_PATH="/app/data"
DATASETS=(
    "esc50:${DATASET_BASE_PATH}/ESC-50"
    "urbansound8k:${DATASET_BASE_PATH}/UrbanSound8K"
    "gtzan:${DATASET_BASE_PATH}/GTZAN"
    "nsynth:hf:jg583/NSynth"
)

# Experiment Settings
USE_GPU=true
SKIP_INDEXING=false
SKIP_EVALUATION=false
```

**Exemplos de Uso:**

1. **Executar todos os experimentos:**
   ```bash
   ./scripts/run_experiments.sh
   ```

2. **Apenas indexar (sem avaliar):**
   ```bash
   # Edite o script e defina: SKIP_EVALUATION=true
   ./scripts/run_experiments.sh
   ```

3. **Apenas avaliar (usar ChromaDB existente):**
   ```bash
   # Edite o script e defina: SKIP_INDEXING=true
   ./scripts/run_experiments.sh
   ```

4. **Comparar diferentes encoders:**
   ```bash
   # Crie cópias do script com diferentes configurações
   cp run_experiments.sh run_experiments_clap.sh
   cp run_experiments.sh run_experiments_wav2vec2.sh
   
   # Edite cada um com ENCODER_NAME diferente
   ./scripts/run_experiments_clap.sh
   ./scripts/run_experiments_wav2vec2.sh
   ```

5. **Dentro do Docker:**
   ```bash
   docker run --gpus all -it --rm \
       -v $(pwd)/ESC-50:/app/data/ESC-50 \
       -v $(pwd)/GTZAN:/app/data/GTZAN \
       -v $(pwd)/chromadb:/app/chromadb \
       -v $(pwd)/logs:/app/logs \
       audio-retrieval \
       bash
   
   # Dentro do container:
   ./scripts/run_experiments.sh
   ```

**Estrutura de Logs:**

Os logs são organizados automaticamente:
```
logs/
├── esc50_clap_cosine_basic/
│   ├── events.out.tfevents.*
│   └── confusion_matrix.png
├── urbansound8k_clap_cosine_basic/
│   ├── events.out.tfevents.*
│   └── confusion_matrix.png
└── ...
```

**Visualizar Resultados:**

```bash
tensorboard --logdir logs
```

**Arquivo de Exemplo:**

Veja `run_experiments.example.sh` para exemplos de configurações alternativas.

## 📚 Recursos Adicionais

- [README_HUGGINGFACE.md](README_HUGGINGFACE.md) - Guia completo para datasets do Hugging Face
- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50)
- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/)
- [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- [AudioSet Dataset](https://research.google.com/audioset/download.html)
- [NSynth Dataset](https://huggingface.co/datasets/jg583/NSynth)
