# Guia de Uso de Datasets do Hugging Face

Este documento explica como usar datasets do Hugging Face com o framework de experimentação de classificação de áudio.

## 📋 Formato Requerido do Dataset

Para que um dataset do Hugging Face funcione automaticamente com nosso framework, ele deve seguir esta estrutura:

### Estrutura Esperada

O dataset deve ter os seguintes splits:
- **`train`**: Split de treinamento (obrigatório)
- **`test`** ou **`validation`**: Split de teste/validação (opcional, será criado automaticamente se não existir)

### Formato dos Itens

Cada item no dataset deve ter:

1. **Campo de Áudio** (um dos seguintes):
   - `audio`: Dicionário com `path` (caminho para arquivo) ou `array` (numpy array)
   - `path`: String com caminho para arquivo de áudio
   - `file`: String com caminho para arquivo de áudio

2. **Campo de Label** (um dos seguintes será procurado nesta ordem):
   - `label`: ID numérico ou string da classe
   - `label_str`: String da classe
   - `class`: Nome da classe
   - `category`: Categoria
   - `genre`: Gênero musical
   - `instrument_family_str`: Família de instrumento

### Exemplo de Estrutura

```python
# Dataset deve ter esta estrutura:
dataset = {
    'train': [
        {
            'audio': {'path': '/path/to/audio1.wav', 'sampling_rate': 16000},
            'label': 'blues'  # ou 'label_str', 'class', etc.
        },
        {
            'audio': {'path': '/path/to/audio2.wav', 'sampling_rate': 16000},
            'label': 'rock'
        },
        ...
    ],
    'test': [
        {
            'audio': {'path': '/path/to/test1.wav', 'sampling_rate': 16000},
            'label': 'jazz'
        },
        ...
    ]
}
```

## 🚀 Como Usar

### Formato Básico

```bash
# Indexar dataset do Hugging Face
python scripts/index_esc50.py \
    --dataset_name hf:dataset_name \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --use_gpu

# Executar experimento
python scripts/run_experiment.py \
    --dataset_name hf:dataset_name \
    --encoder_name clap \
    --model_name laion/clap-htsat-unfused \
    --distance_metric cosine \
    --retrieval_strategy basic \
    --use_gpu
```

### Com Configuração

Se o dataset tiver múltiplas configurações:

```bash
# Exemplo: Common Voice com idioma específico
python scripts/index_esc50.py \
    --dataset_name hf:common_voice/en \
    --encoder_name clap \
    --distance_metric cosine \
    --use_gpu
```

**Nota:** `--dataset_path` não é necessário para datasets do Hugging Face (pode ser omitido).

## ✅ Datasets Testados e Compatíveis

### NSynth (Instrument Classification)

```bash
# NSynth já está implementado como 'nsynth', mas também funciona via HF
python scripts/index_esc50.py \
    --dataset_name hf:jg583/NSynth \
    --encoder_name clap \
    --distance_metric cosine \
    --use_gpu
```

### Common Voice (Speech Recognition)

```bash
# Common Voice (exemplo com inglês)
python scripts/index_esc50.py \
    --dataset_name hf:mozilla-foundation/common_voice_13_0/en \
    --encoder_name wav2vec2 \
    --model_name facebook/wav2vec2-base \
    --distance_metric cosine \
    --use_gpu
```

**Nota:** Common Voice pode precisar de processamento adicional para classificação, pois é um dataset de reconhecimento de fala.

### FLEURS (Multilingual Speech)

```bash
python scripts/index_esc50.py \
    --dataset_name hf:google/fleurs/en_us \
    --encoder_name wav2vec2 \
    --distance_metric cosine \
    --use_gpu
```

## 🔍 Verificando Compatibilidade

Para verificar se um dataset do Hugging Face é compatível:

```python
from datasets import load_dataset

# Carregar dataset
dataset = load_dataset("nome_do_dataset")

# Verificar estrutura
print("Splits disponíveis:", list(dataset.keys()))

# Verificar formato de um item
example = dataset['train'][0]
print("Chaves disponíveis:", list(example.keys()))

# Verificar se tem áudio
if 'audio' in example:
    print("Formato de áudio:", type(example['audio']))
    if isinstance(example['audio'], dict):
        print("Chaves de áudio:", list(example['audio'].keys()))

# Verificar se tem label
label_keys = ['label', 'label_str', 'class', 'category', 'genre']
for key in label_keys:
    if key in example:
        print(f"Label encontrado em '{key}':", example[key])
        break
```

## ⚠️ Limitações e Considerações

### 1. Áudio em Memória (Arrays)

Se o dataset fornecer áudio como arrays numpy em memória (não como caminhos de arquivo), esses itens serão **pulados** com um aviso. O framework atualmente suporta apenas arquivos de áudio em disco.

**Solução:** Use datasets que forneçam caminhos de arquivo, ou converta o dataset para salvar os arquivos localmente.

### 2. Multi-label

Se o dataset tiver múltiplas labels por item, apenas a primeira será usada. Para suporte completo a multi-label, seria necessário modificar o código.

### 3. Sem Split de Teste

Se o dataset não tiver split de teste ou validação, o código automaticamente divide o treino em 80% treino / 20% teste (estratificado por label).

### 4. Download Automático

O Hugging Face `datasets` baixa automaticamente os dados na primeira execução. Certifique-se de ter espaço em disco suficiente.

## 📝 Criando um Dataset Compatível no Hugging Face

Se você quiser criar seu próprio dataset no Hugging Face que seja compatível:

### Exemplo de Script de Criação

```python
from datasets import Dataset, Audio
import pandas as pd

# Seus dados
data = {
    'audio': ['path/to/audio1.wav', 'path/to/audio2.wav', ...],
    'label': ['class1', 'class2', ...]
}

# Criar dataset
dataset = Dataset.from_dict(data)

# Adicionar feature de áudio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Dividir em train/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Upload para Hugging Face (opcional)
# dataset.push_to_hub("seu-usuario/nome-do-dataset")
```

### Checklist de Compatibilidade

- [ ] Dataset tem split `train`
- [ ] Dataset tem split `test` ou `validation` (ou será criado automaticamente)
- [ ] Cada item tem campo de áudio (`audio.path`, `path`, ou `file`)
- [ ] Cada item tem campo de label (`label`, `label_str`, `class`, `category`, `genre`, ou `instrument_family_str`)
- [ ] Áudio é fornecido como caminhos de arquivo (não arrays em memória)

## 🐛 Troubleshooting

**Erro: "Could not find label for item"**
- Verifique se o dataset tem um dos campos de label suportados
- Adicione um campo de label ao dataset ou modifique `utils.py` para procurar outro nome

**Erro: "Could not find audio path"**
- Verifique se o dataset tem campo `audio.path`, `path`, ou `file`
- Se o áudio está em memória (array), você precisa salvar os arquivos primeiro

**Erro: "No test/validation split found"**
- Isso é normal! O código criará automaticamente um split de teste
- Ou adicione um split `test` ou `validation` ao dataset

**Erro: "datasets library is required"**
- Instale: `pip install datasets`

## 📚 Recursos

- [Documentação do Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Guia de Criação de Datasets](https://huggingface.co/docs/datasets/create_dataset)
- [Lista de Datasets de Áudio](https://huggingface.co/datasets?task_categories=task_categories:audio-classification)

