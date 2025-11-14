#!/usr/bin/env python3
"""
extract_mels.py

Extrai recursivamente arquivos .tar* encontrados em --data-dir
para --output-dir/<nome_do_tar_sem_ext>/ e grava um log em --log.

Mostra barras de progresso (tqdm) durante a execução.
"""
import tarfile
import os
from pathlib import Path
import traceback
import sys
import time
import argparse

# tenta importar tqdm; se não instalado, define passthrough
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False
    def tqdm(x, **kw):
        return x

# Extensões reconhecidas
TAR_EXTS = [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"]

def is_tar_file(path: Path):
    name = path.name.lower()
    return any(name.endswith(ext) for ext in TAR_EXTS)

def safe_member_path(dest: Path, member_name: str) -> Path:
    """
    Resolve o caminho de destino de um membro do tar e evita path traversal.
    """
    member_path = Path(member_name)
    resolved = (dest / member_path).resolve()
    if not str(resolved).startswith(str(dest.resolve())):
        raise Exception(f"Arquivo {member_name} tenta sair do diretório de destino ({resolved})")
    return resolved

def find_tar_files(root: Path):
    """
    Retorna generator de Path para arquivos .tar* sob root (recursivo).
    """
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_tar_file(p):
                yield p

def sanitize_name_no_ext(filename: str):
    """
    Remove a extensão conhecida para formar nome da pasta de destino.
    """
    name = filename
    for ext in TAR_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def extract_tar_with_progress(tarpath: Path, dest: Path, log_lines: list):
    """
    Extrai o tar iterando membro a membro para permitir barra de progresso.
    """
    extracted_count = 0
    errors = []
    try:
        with tarfile.open(tarpath, mode='r:*') as tf:
            members = tf.getmembers()
            # tqdm sobre os membros
            for m in (tqdm(members, desc=f"  extraindo {tarpath.name}", leave=False) if TQDM_AVAILABLE else members):
                try:
                    # evita path traversal
                    _ = safe_member_path(dest, m.name)
                    tf.extract(m, path=dest)
                    extracted_count += 1
                except Exception as me:
                    errors.append((m.name, str(me)))
    except Exception as e:
        tb = traceback.format_exc()
        log_lines.append(f"  ERRO ao abrir/extrair {tarpath}: {e}")
        log_lines.append(tb)
        return 0, [(str(tarpath), str(e))]
    return extracted_count, errors

def main():
    parser = argparse.ArgumentParser(description="Extrai melspec .tar* para uma pasta organizada (com progresso)")
    parser.add_argument("--data-dir", type=str, default="data/raw_melspecs",
                        help="Diretório onde estão os .tar (default: data/raw_melspecs)")
    parser.add_argument("--output-dir", type=str, default="data/melspecs_extracted",
                        help="Diretório base para extração (default: data/melspecs_extracted)")
    parser.add_argument("--log", type=str, default="data/extract_mels.log",
                        help="Arquivo de log (default: data/extract_mels.log)")
    parser.add_argument("--list-only", action="store_true",
                        help="Apenas lista .tar encontrados sem extrair")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_base = Path(args.output_dir).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()

    start = time.time()
    log_lines = []
    # tenta criar output_base (ou fallback para cwd/melspecs_extracted)
    try:
        ensure_dir(output_base)
    except Exception as e:
        fallback = Path.cwd() / "melspecs_extracted"
        try:
            ensure_dir(fallback)
            log_lines.append(f"Aviso: não foi possível criar {output_base}, usando fallback {fallback}")
            output_base = fallback
        except Exception as e2:
            log_lines.append(f"Erro: não foi possível criar diretório de saída: {e2}")
            print("\n".join(log_lines))
            sys.exit(1)

    # coleta arquivos tar
    tar_files = []
    if data_dir.exists():
        tar_files = list(find_tar_files(data_dir))
    else:
        log_lines.append(f"Atenção: data-dir {data_dir} não existe.")

    log_lines.append(f"Execução iniciada: {time.ctime()}")
    log_lines.append(f"Procurando .tar em: {data_dir} (existente: {data_dir.exists()})")
    log_lines.append(f"{len(tar_files)} arquivos tar encontrados.")

    total_extracted = 0
    details = []
    all_member_errors = []

    if args.list_only:
        for t in tar_files:
            log_lines.append(f"  - {t}")
    else:
        # tqdm sobre tar_files
        tar_iter = tqdm(tar_files, desc="Arquivos .tar", unit="tar") if (TQDM_AVAILABLE and tar_files) else tar_files
        for tarpath in tar_iter:
            try:
                name_no_ext = sanitize_name_no_ext(tarpath.name)
                dest = output_base / name_no_ext
                ensure_dir(dest)
                log_lines.append(f"\nExtraindo: {tarpath} -> {dest}")
                # Extrai membro a membro com progresso
                extracted_count, member_errors = extract_tar_with_progress(tarpath, dest, log_lines)
                total_extracted += extracted_count
                if member_errors:
                    for me in member_errors:
                        all_member_errors.append((str(tarpath), me))
                # pegar alguns exemplos
                examples = []
                for i, p in enumerate(sorted(dest.rglob("*"))):
                    if p.is_file():
                        examples.append(str(p.relative_to(dest)))
                    if i >= 9:
                        break
                details.append((str(tarpath), extracted_count, examples))
                log_lines.append(f"  -> {extracted_count} itens extraídos.")
                if examples:
                    log_lines.append("  Exemplos: " + ", ".join(examples[:5]))
            except Exception as e:
                tb = traceback.format_exc()
                log_lines.append(f"  ERRO ao extrair {tarpath}: {e}")
                log_lines.append(tb)

    end = time.time()
    log_lines.append(f"\nResumo final: {len(tar_files)} tar(s) processado(s).")
    if not args.list_only:
        log_lines.append(f"Total aproximado de arquivos extraídos: {total_extracted}")
    log_lines.append(f"Tempo decorrido: {end - start:.1f} segundos.")
    log_lines.append(f"Output base: {output_base}")
    log_lines.append(f"Log path: {log_path}")
    if all_member_errors:
        log_lines.append(f"Erros em membros: {len(all_member_errors)} (ver detalhes abaixo)")

    # grava log (tenta gravar no caminho escolhido, se falhar imprime no stdout)
    try:
        ensure_dir(log_path.parent)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
            if all_member_errors:
                f.write("\n\nDetalhes erros membros:\n")
                for tpath, err in all_member_errors:
                    f.write(f"{tpath} :: {err}\n")
    except Exception as e:
        print(f"Não foi possível gravar o log em {log_path} : {e}", file=sys.stderr)
        print("\n".join(log_lines))

    # resumo para o usuário (stdout)
    print("\n".join(log_lines[:60]))
    if details:
        print("\nDetalhes por arquivo tar (até 10 exemplos por tar):")
        for tarpath, cnt, examples in details:
            print(f"- {tarpath}: {cnt} itens. Exemplos: {', '.join(examples[:5])}")
    else:
        if tar_files:
            print("Arquivos .tar listados (nenhuma extração foi feita?):")
            for t in tar_files[:50]:
                print(" -", t)
        else:
            print("Nenhum arquivo tar processado.")

    print(f"\nLog gravado em: {log_path}")
    print(f"Extrações gravadas em: {output_base}")

    return {
        "n_tar_files": len(tar_files),
        "total_extracted": total_extracted,
        "output_dir": str(output_base),
        "log_path": str(log_path)
    }

if __name__ == "__main__":
    if not TQDM_AVAILABLE:
        print("Aviso: pacote 'tqdm' não encontrado — rodando sem barras de progresso. "
              "Instale com: pip install tqdm")
    main()
