#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys

# tqdm
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False
    def tqdm(x, **k): return x

# yaml
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


# -------------------------------
# Load YAML
# -------------------------------
def load_yaml_config(path: str):
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML não instalado. Instale com: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {path}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise RuntimeError("YAML inválido ou vazio.")

    return cfg


# -------------------------------
# Dataset
# -------------------------------
class TokenDataset(Dataset):
    def __init__(self, tokens_dir: str):
        self.root = Path(tokens_dir)
        self.files = sorted([p for p in self.root.rglob("*.pt") if p.is_file()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        tokens = torch.load(p)
        return tokens, str(p.relative_to(self.root))


def collate_tokens(batch):
    lengths = [t.shape[0] for t, _ in batch]
    maxL = max(lengths)
    d = batch[0][0].shape[1]
    B = len(batch)
    out = torch.zeros(B, maxL, d, dtype=torch.float)
    mask = torch.zeros(B, maxL, dtype=torch.bool)
    rels = []

    for i, (t, rel) in enumerate(batch):
        L = t.shape[0]
        out[i, :L] = t
        if L < maxL:
            mask[i, L:] = True
        rels.append(rel)

    return out, mask, rels


# -------------------------------
# Modelo Transformer
# -------------------------------
class AudioTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_ff, dropout, pool):
        super().__init__()
        self.pool = pool

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model)) if pool == "cls" else None
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        B = x.size(0)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

            pad_extra = torch.zeros(B,1, dtype=torch.bool, device=pad_mask.device)
            pad_mask = torch.cat([pad_extra, pad_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        if self.pool == "cls" and self.cls_token is not None:
            out = x[:,0]
        else:
            inv = (~pad_mask).float().unsqueeze(-1)
            out = (x * inv).sum(dim=1) / inv.sum(dim=1).clamp(min=1)

        out = self.norm(self.proj(out))
        out = out / out.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return out


# -------------------------------
# Execução
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Arquivo YAML obrigatório")
    args = parser.parse_args()

    # Impedir rodar sem config
    if args.config is None:
        print("Erro: é obrigatório fornecer --config <arquivo.yaml>")
        sys.exit(1)

    cfg = load_yaml_config(args.config)

    # Conferência obrigatória dos campos:
    required = [
        "tokens_dir", "output_dir", "device", "batch_size",
        "d_model", "n_layers", "n_heads", "dim_ff",
        "dropout", "pool", "skip_existing"
    ]

    for r in required:
        if r not in cfg:
            print(f"Erro: campo '{r}' faltando no YAML {args.config}")
            sys.exit(1)

    # Leitura das configs
    tokens_dir = cfg["tokens_dir"]
    output_dir = cfg["output_dir"]
    device = torch.device(cfg["device"])
    batch_size = int(cfg["batch_size"])
    d_model = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    n_heads = int(cfg["n_heads"])
    dim_ff = int(cfg["dim_ff"])
    dropout = float(cfg["dropout"])
    pool = cfg["pool"]
    skip_existing = bool(cfg["skip_existing"])

    ds = TokenDataset(tokens_dir)
    if len(ds) == 0:
        print("Nenhum token encontrado em:", tokens_dir)
        sys.exit(1)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_tokens)

    model = AudioTransformer(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_ff=dim_ff,
        dropout=dropout,
        pool=pool
    ).to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    model.eval()
    with torch.no_grad():
        iterator = tqdm(dl, desc="Transformando", unit="batch") if TQDM else dl

        for tokens, mask, rels in iterator:
            out_paths = [output_dir / (r + ".pt") for r in rels]

            if skip_existing and all(p.exists() for p in out_paths):
                continue

            tokens = tokens.to(device)
            mask = mask.to(device)

            emb = model(tokens, mask).cpu()

            for i, r in enumerate(rels):
                path = output_dir / (r + ".pt")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emb[i], path)
                saved += 1

    print("Embeddings salvos:", saved)


if __name__ == "__main__":
    if not YAML_AVAILABLE:
        print("Erro: PyYAML não instalado. Instale com: pip install pyyaml")
        sys.exit(1)

    main()
