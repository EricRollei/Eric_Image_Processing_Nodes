"""Utilities for downloading third-party transformer restoration models."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

TRANSFORMER_LAB_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_DIR = TRANSFORMER_LAB_ROOT / "third_party"
MODELS_DIR = TRANSFORMER_LAB_ROOT / "models"
PREFERRED_WEIGHT_DIR = MODELS_DIR / "pretrained_weights"
GLOBAL_MODELS_DIR = TRANSFORMER_LAB_ROOT.parent / "models"
GLOBAL_PRETRAINED_DIR = GLOBAL_MODELS_DIR / "pretrained_weights"

_SEARCH_PATH_CANDIDATES = (
    PREFERRED_WEIGHT_DIR,
    MODELS_DIR,
    GLOBAL_PRETRAINED_DIR,
    GLOBAL_MODELS_DIR,
)

seen_paths = []
for _path in _SEARCH_PATH_CANDIDATES:
    if _path not in seen_paths:
        seen_paths.append(_path)

WEIGHT_SEARCH_DIRS = tuple(seen_paths)
WEIGHT_WRITE_DIRS = (PREFERRED_WEIGHT_DIR,)


@dataclass
class WeightFile:
    filename: str
    url: str
    sha256: Optional[str] = None


@dataclass
class RepoSpec:
    name: str
    local_dirname: str
    git_url: Optional[str]
    zip_urls: tuple[str, ...]
    deprecation_reason: Optional[str] = None
    weight_files: Iterable[WeightFile] = field(default_factory=list)


REPO_SPECS: Dict[str, RepoSpec] = {
    "diffbir": RepoSpec(
        name="diffbir",
        local_dirname="diffbir",
        git_url="https://github.com/XPixelGroup/DiffBIR.git",
        zip_urls=(
            "https://codeload.github.com/XPixelGroup/DiffBIR/zip/refs/heads/main",
            "https://github.com/XPixelGroup/DiffBIR/archive/refs/heads/main.zip",
        ),
        deprecation_reason=None,
        weight_files=[
            # Stage-2 IRControlNet checkpoints
            WeightFile(
                filename="DiffBIR_v2.1.pt",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt",
            ),
            WeightFile(
                filename="v2.pth",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
            ),
            WeightFile(
                filename="v1_general.pth",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
            ),
            WeightFile(
                filename="v1_face.pth",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
            ),
            # Stage-1 cleaner models
            WeightFile(
                filename="realesrgan_s4_swinir_100k.pth",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
            ),
            WeightFile(
                filename="general_swinir_v1.ckpt",
                url="https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
            ),
            WeightFile(
                filename="face_swinir_v1.ckpt",
                url="https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
            ),
            WeightFile(
                filename="BSRNet.pth",
                url="https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
            ),
            WeightFile(
                filename="scunet_color_real_psnr.pth",
                url="https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
            ),
            WeightFile(
                filename="codeformer_swinir.ckpt",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/codeformer_swinir.ckpt",
            ),
            # Stable Diffusion backbones
            WeightFile(
                filename="v2-1_512-ema-pruned.ckpt",
                url="https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
            ),
            WeightFile(
                filename="sd2.1-base-zsnr-laionaes5.ckpt",
                url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
            ),
        ],
    ),
    "diffir": RepoSpec(
        name="diffir",
        local_dirname="diffir",
        git_url="https://github.com/ChaofWang/DiffIR.git",
        zip_urls=(
            "https://codeload.github.com/ChaofWang/DiffIR/zip/refs/heads/main",
            "https://codeload.github.com/ChaofWang/DiffIR/zip/refs/heads/master",
            "https://github.com/ChaofWang/DiffIR/archive/refs/heads/main.zip",
            "https://github.com/ChaofWang/DiffIR/archive/refs/heads/master.zip",
            "https://github.com/ChaofWang/DiffIR/archive/refs/tags/v1.0.zip",
        ),
        weight_files=[
            WeightFile(
                filename="DiffIR_s128_nf64_bs32.pth",
                url="https://github.com/ChaofWang/DiffIR/releases/download/v1.0/DiffIR_s128_nf64_bs32.pth",
            ),
        ],
    ),
    "kvdegformer": RepoSpec(
        name="kvdegformer",
        local_dirname="kv-deformer",
        git_url="https://github.com/zzh-tech/KV-Deformer.git",
        zip_urls=(
            "https://codeload.github.com/zzh-tech/KV-Deformer/zip/refs/heads/main",
            "https://codeload.github.com/zzh-tech/KV-Deformer/zip/refs/heads/master",
            "https://github.com/zzh-tech/KV-Deformer/archive/refs/heads/main.zip",
            "https://github.com/zzh-tech/KV-Deformer/archive/refs/heads/master.zip",
            "https://github.com/zzh-tech/KV-Deformer/archive/refs/tags/v1.0.zip",
        ),
        weight_files=[
                WeightFile(
                    filename="kvdeformer_motion.pth",
                    url="https://github.com/zzh-tech/KV-Deformer/releases/download/v1.0/kvdeformer_motion.pth",
                ),
                WeightFile(
                    filename="kvdeformer_gaussian.pth",
                    url="https://github.com/zzh-tech/KV-Deformer/releases/download/v1.0/kvdeformer_gaussian.pth",
                ),
        ],
    ),
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            with urllib.request.urlopen(url) as response:
                chunk = response.read(8192)
                while chunk:
                    tmp_file.write(chunk)
                    chunk = response.read(8192)
        except urllib.error.HTTPError as exc:
            tmp_file.close()
            tmp_path.unlink(missing_ok=True)
            if exc.code == 401 and "huggingface" in url:
                raise RuntimeError(
                    "Hugging Face returned HTTP 401 Unauthorized while downloading "
                    f"{destination.name}. This model requires accepting the license and "
                    "setting an access token. Set the HUGGINGFACE_TOKEN or "
                    "HUGGINGFACE_HUB_TOKEN environment variable and rerun, or download "
                    "the file manually."
                ) from exc
            raise RuntimeError(f"Download failed ({exc.code}): {url}") from exc
    try:
        if destination.exists():
            destination.unlink()
        shutil.move(str(tmp_path), str(destination))
    finally:
        tmp_path.unlink(missing_ok=True)


def _verify_checksum(path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _ensure_repo(spec: RepoSpec) -> Path:
    target_dir = THIRD_PARTY_DIR / spec.local_dirname
    if target_dir.exists():
        return target_dir

    if spec.deprecation_reason:
        raise RuntimeError(spec.deprecation_reason)

    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Optional[urllib.error.HTTPError] = None

    for url in spec.zip_urls:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            urllib.request.urlretrieve(url, tmp_path)
        except urllib.error.HTTPError as exc:
            tmp_path.unlink(missing_ok=True)
            last_error = exc
            continue

        try:
            with zipfile.ZipFile(tmp_path, "r") as archive:
                root_entries = []
                for member in archive.namelist():
                    if not member or member.startswith("__MACOSX"):
                        continue
                    root_name = member.split("/", 1)[0]
                    if root_name and root_name not in root_entries:
                        root_entries.append(root_name)
                archive.extractall(THIRD_PARTY_DIR)
        except zipfile.BadZipFile as exc:  # pragma: no cover - network data issues are rare
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded archive for '{spec.name}' was invalid: {url}"
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        for root_name in root_entries:
            candidate = THIRD_PARTY_DIR / root_name
            if not candidate.exists():
                continue
            if candidate == target_dir:
                return target_dir
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(candidate), str(target_dir))
            return target_dir

    git_exe = shutil.which("git")
    if git_exe and spec.git_url:
        try:
            subprocess.run(
                [git_exe, "clone", "--depth", "1", spec.git_url, str(target_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return target_dir
        except subprocess.CalledProcessError as exc:
            clone_log = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                "Failed to clone repository for "
                f"'{spec.name}' using git.\n"
                f"Command: {' '.join(exc.cmd)}\n"
                f"Exit code: {exc.returncode}\n"
                f"Output: {clone_log}"
            ) from exc

    if last_error is not None:
        raise RuntimeError(
            "Failed to download repository archive for "
            f"'{spec.name}'. Tried: {', '.join(spec.zip_urls)}\n"
            f"Last error: HTTP {last_error.code} ({last_error.reason}). "
            "Install git to enable an automatic clone or download manually."
        ) from last_error

    raise RuntimeError(
        f"Failed to download repository archive for '{spec.name}' due to an unknown error."
    )


def _ensure_weights(spec: RepoSpec) -> None:
    for directory in WEIGHT_WRITE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    for weight in spec.weight_files:
        existing_path = None
        for directory in WEIGHT_SEARCH_DIRS:
            candidate = directory / weight.filename
            if candidate.exists():
                existing_path = candidate
                break
        if existing_path is not None:
            if weight.sha256 and not _verify_checksum(existing_path, weight.sha256):
                existing_path.unlink()
            else:
                continue
        dest = PREFERRED_WEIGHT_DIR / weight.filename
        _download_file(weight.url, dest)
        if weight.sha256 and not _verify_checksum(dest, weight.sha256):
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {dest.name}")


def ensure_model_assets(model: str) -> Path:
    if model not in REPO_SPECS:
        raise ValueError(f"Unknown model '{model}'. Valid options: {', '.join(REPO_SPECS)}")
    spec = REPO_SPECS[model]
    repo_path = _ensure_repo(spec)
    _ensure_weights(spec)
    return repo_path


def cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download third-party transformer model assets")
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(REPO_SPECS.keys()) + ["all"],
        default=["all"],
        help="Which model repos to download (default: all)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    models = set(args.models)
    if "all" in models:
        models = set(REPO_SPECS.keys())

    for model in sorted(models):
        print(f"Fetching assets for {model}...")
        repo_path = ensure_model_assets(model)
        print(f"  • Repo:    {repo_path}")
        print(f"  • Weights: {PREFERRED_WEIGHT_DIR}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())
