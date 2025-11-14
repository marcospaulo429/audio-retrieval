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
    pos = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

def tokenize_and_save(data_dir: str, output_dir: str, patch_size: int, d_model: int, pos_type: str, device_str: str, skip_existing: bool):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)
    proj = {}
    bad_files = []
    files = sorted(p for p in data_dir.rglob("*.npy"))
    it = tqdm(files, desc="Tokenizando", unit="file") if TQDM else files
    saved = 0
    for p in it:
        outp = output_dir / (p.relative_to(data_dir).with_suffix(".pt").as_posix())
        if skip_existing and outp.exists():
            continue
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
        pad = (patch_size - (T % patch_size)) % patch_size
        if pad > 0:
            mel = nn.functional.pad(mel, (0, pad))
            T += pad
        P = T // patch_size
        patches = mel.view(F, P, patch_size).permute(1,0,2).contiguous().view(P, F * patch_size).to(device)  # (P, F*patch)
        key = f"{F}x{patch_size}"
        if key not in proj:
            proj[key] = nn.Linear(F * patch_size, d_model).to(device)
        with torch.no_grad():
            tokens = proj[key](patches)  # (P, d_model)
            if pos_type == "sinusoidal":
                pe = sinusoidal_pos_encoding(tokens.size(0), d_model, device=device)
            else:
                pe = torch.randn(tokens.size(0), d_model, device=device) * 0.02
            tokens = tokens + pe
            tokens = tokens.cpu()
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, outp)
        saved += 1
    if bad_files:
        with open("bad_files_tokens.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(bad_files))
    return saved, bad_files

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/melspecs_extracted")
    p.add_argument("--output-dir", type=str, default="data/tokens")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--pos-type", type=str, choices=["sinusoidal","learnable"], default="sinusoidal")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    saved, bad = tokenize_and_save(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        d_model=args.d_model,
        pos_type=args.pos_type,
        device_str=args.device,
        skip_existing=args.skip_existing
    )
    print(f"Tokens salvos: {saved}")
    print(f"Arquivos com erro: {len(bad)}")
    if bad:
        print("Lista em bad_files_tokens.txt")

if __name__ == "__main__":
    if not TQDM:
        print("Aviso: tqdm não instalado; instale para barras de progresso.")
    main()
