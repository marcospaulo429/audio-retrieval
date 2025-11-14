#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import math

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False
    def tqdm(x, **k): return x


def sinusoidal_pos_encoding(length: int, d_model: int, device=None):
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def extract_and_save_embeddings(
    data_dir: str,
    output_dir: str,
    patch_size: int,
    d_model: int,
    pos_type: str,
    device_str: str
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    proj = None
    bad_files = []

    files = sorted(p for p in data_dir.rglob("*.npy"))
    iterator = tqdm(files, desc="Arquivos", unit="file") if TQDM else files

    saved = 0

    for p in iterator:
        outp = output_dir / (p.relative_to(data_dir).with_suffix(".pt").as_posix())

        # -------------------------
        # Skip automático
        # -------------------------
        if outp.exists():
            continue

        # -------------------------
        # Ler .npy (com detecção de corrupção)
        # -------------------------
        try:
            arr = np.load(p)
        except Exception:
            bad_files.append(str(p))
            continue

        mel = torch.from_numpy(arr).float()
        if mel.ndim == 1:
            mel = mel.unsqueeze(1)
        if mel.ndim == 3:
            mel = mel.squeeze(0)

        F, T = mel.shape

        # -------------------------
        # Patch embedding
        # -------------------------
        pad = (patch_size - (T % patch_size)) % patch_size
        if pad > 0:
            mel = nn.functional.pad(mel, (0, pad))
            T += pad

        P = T // patch_size

        patches = (
            mel.view(F, P, patch_size)
               .permute(1, 0, 2)
               .contiguous()
               .view(P, F * patch_size)
        ).to(device)

        if proj is None:
            proj = nn.Linear(F * patch_size, d_model).to(device)

        with torch.no_grad():
            tokens = proj(patches)

            # -------------------------
            # Positional encoding
            # -------------------------
            if pos_type == "sinusoidal":
                pe = sinusoidal_pos_encoding(P, d_model, device=device)
            else:
                pe = torch.randn(P, d_model, device=device) * 0.02

            tokens = tokens + pe

            # -------------------------
            # Pooling
            # -------------------------
            embedding = tokens.mean(dim=0).cpu()

        # -------------------------
        # Salvar embedding
        # -------------------------
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, outp)
        saved += 1

    if bad_files:
        with open("bad_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(bad_files))

    return saved, bad_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/melspecs_extracted")
    parser.add_argument("--output-dir", type=str, default="data/embeddings")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--pos-type", type=str, choices=["sinusoidal", "learnable"], default="sinusoidal")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    saved, corrupted = extract_and_save_embeddings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        d_model=args.d_model,
        pos_type=args.pos_type,
        device_str=args.device,
    )

    print(f"Embeddings gerados: {saved}")
    print(f"Arquivos corrompidos: {len(corrupted)}")
    if corrupted:
        print("Lista salva em: bad_files.txt")


if __name__ == "__main__":
    if not TQDM:
        print("Instale tqdm para ver o progresso.")
    main()
