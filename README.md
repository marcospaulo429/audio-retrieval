# 🎧 Classificador de Áudio k-NN com CLAP

Este projeto implementa um classificador de áudio simples usando a técnica de **k-Nearest Neighbors (k-NN)** aplicada sobre *embeddings* de áudio gerados pelo modelo **CLAP (Contrastive Language-Audio Pretraining)** da Microsoft.

O script `testar_clap.py` é configurado para:
1.  Baixar o modelo CLAP (`msclap`, versão 2023).
2.  Baixar o dataset **ESC-50**.
3.  Dividir o dataset em treino (Folds 1-4) e teste (Fold 5).
4.  Gerar *embeddings* para todos os áudios de treino.
5.  Treinar um classificador k-NN (`k=5`) com os *embeddings* de treino.
6.  Avaliar o classificador no conjunto de teste e exibir a acurácia e o relatório de classificação.

---

## 🚀 Como Instalar e Rodar

Estas instruções usam [**`uv`**](https://astral.sh/uv), um gerenciador de ambientes e pacotes Python moderno e rápido.

### 1. Pré-requisitos

* [Python 3.10+](https://www.python.org/downloads/)
* [**`uv`**](https://astral.sh/uv/install.sh) (Recomendado)
* `git` (para clonar, opcional)

### 2. Passos de Instalação

1.  **Clone ou baixe os arquivos do projeto:**
    ```bash
    # Se você estiver usando git
    git clone [https://seu-repositorio-aqui.git](https://seu-repositorio-aqui.git)
    cd nome-do-projeto
    
    # Ou apenas navegue até a pasta se você já baixou os arquivos
    cd /caminho/para/meu_projeto_clap
    ```

2.  **Crie um Ambiente Virtual:**
    (Isso cria uma pasta `.venv` para isolar as dependências)
    ```bash
    uv venv
    ```

3.  **Ative o Ambiente Virtual:**
    * No macOS / Linux:
        ```bash
        source .venv/bin/activate
        ```
    * No Windows (PowerShell):
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    * No Windows (CMD):
        ```cmd
        .venv\Scripts\activate.bat
        ```
    Você saberá que funcionou pois o nome `(.venv)` aparecerá no seu prompt.

4.  **Instale as Dependências:**
    O `uv` lerá o arquivo `pyproject.toml` e instalará tudo o que está listado em `[project.dependencies]`.
    ```bash
    uv pip install .
    ```
    *Nota: O `.` significa "instalar o projeto no diretório atual".*

### 3. Como Executar

Com o ambiente ativado e as dependências instaladas, basta rodar o script Python:

```bash
python testar_clap.py
```

### O que Esperar

* Na primeira execução, o script levará alguns minutos para:
    1.  Baixar os pesos do modelo CLAP (aprox. 600MB).
    2.  Baixar o dataset ESC-50 (aprox. 650MB).
* Em seguida, ele começará a gerar os *embeddings* (o que pode demorar um pouco, dependendo do seu CPU/GPU).
* Ao final, ele exibirá um relatório com a **Acurácia Geral** e as métricas de Precisão, Recall e F1-Score para cada classe.