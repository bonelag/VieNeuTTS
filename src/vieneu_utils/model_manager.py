"""
Portable Model Manager for VieNeu-TTS.
======================================
- Manages models locally in `model/` (backbone, codec, tokenizers).
- Completely portable: no reliance on ~/.cache/huggingface.
- Checks commit hash on HuggingFace Hub on model load; auto-downloads/replaces when updated.
- Works 100% offline if network is unavailable and local model exists.
- Cleans any temporary HuggingFace .cache metadata.
"""

from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

# Set portable environment settings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

logger = logging.getLogger("Vieneu.ModelManager")

# Root directory of the repository: parent of `src/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(os.getenv("VIENEU_MODEL_DIR", PROJECT_ROOT / "model")).resolve()

# Known mapping of HuggingFace repos -> (category, local_folder_name)
KNOWN_MODELS = {
    "pnnbao-ump/VieNeu-TTS-v3-Turbo": ("backbone", "VieNeu-TTS-v3-Turbo"),
    "pnnbao-ump/VieNeu-TTS-v3-Nano": ("backbone", "VieNeu-TTS-v3-Nano"),
    "pnnbao-ump/VieNeu-TTS-v2": ("backbone", "VieNeu-TTS-v2"),
    "pnnbao-ump/VieNeu-TTS": ("backbone", "VieNeu-TTS"),
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano": ("codec", "MOSS-Audio-Tokenizer-Nano"),
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX": ("codec", "MOSS-Audio-Tokenizer-Nano-ONNX"),
    "neuphonic/distill-neucodec": ("codec", "distill-neucodec"),
    "neuphonic/neucodec": ("codec", "neucodec"),
    "neuphonic/neucodec-onnx-decoder-int8": ("codec", "neucodec-onnx-decoder-int8"),
    "pnnbao-ump/VieNeu-Codec": ("codec", "VieNeu-Codec"),
}


def get_model_destination(repo_or_name: str, category: Optional[str] = None) -> Tuple[str, Path]:
    """Determine the category and local folder Path under `model/` for a given repo or model name."""
    clean_id = repo_or_name.strip()
    if clean_id in KNOWN_MODELS:
        cat, folder_name = KNOWN_MODELS[clean_id]
        if category:
            cat = category
        return cat, MODEL_DIR / cat / folder_name

    # Check if user already placed it directly under model/<clean_id>
    flat_path = MODEL_DIR / clean_id.split("/")[-1]
    if flat_path.is_dir():
        cat = category or ("codec" if "codec" in clean_id.lower() or "moss" in clean_id.lower() else "backbone")
        return cat, flat_path

    # Automatic category detection
    folder_name = clean_id.split("/")[-1]
    if not category:
        low = clean_id.lower()
        if any(k in low for k in ["codec", "moss", "tokenizer", "vocos"]):
            cat = "codec"
        else:
            cat = "backbone"
    else:
        cat = category

    return cat, MODEL_DIR / cat / folder_name


def clean_hf_cache(target_dir: Path) -> None:
    """Remove any .cache or download artifacts created by huggingface_hub inside target_dir."""
    try:
        cache_dir = target_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception as e:
        logger.debug(f"Error cleaning {cache_dir}: {e}")


def get_local_commit_hash(target_dir: Path) -> Optional[str]:
    """Read the recorded commit hash from target_dir/.commit_hash."""
    hash_file = target_dir / ".commit_hash"
    if hash_file.is_file():
        try:
            return hash_file.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def set_local_commit_hash(target_dir: Path, commit_hash: str) -> None:
    """Save the commit hash to target_dir/.commit_hash."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / ".commit_hash").write_text(commit_hash.strip(), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not save .commit_hash in {target_dir}: {e}")


def get_remote_commit_hash(repo_id: str, hf_token: Optional[str] = None, timeout: float = 4.0) -> Optional[str]:
    """Fetch the latest commit SHA from HuggingFace Hub. Returns None on network error."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        info = api.repo_info(repo_id=repo_id, repo_type="model", timeout=timeout)
        return info.sha
    except Exception as e:
        logger.info(f"Could not fetch remote commit hash for '{repo_id}' ({e})")
        return None


def is_model_dir_valid(target_dir: Path, category: str = "backbone") -> bool:
    """Check if target_dir has essential files indicating a complete download."""
    if not target_dir.is_dir():
        return False
    # Look for at least one weights or config file
    indicators = [
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
        "meta.yaml",
        "constants.npz",
        "flow.onnx",
        "model.onnx",
        "decoder.onnx",
        "update",
        "onnx_update",
    ]
    for ind in indicators:
        if (target_dir / ind).exists():
            return True
    return False


def ensure_model(
    repo_or_path: str,
    category: Optional[str] = None,
    check_update: bool = True,
    hf_token: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Ensure the model is present in `model/<category>/<name>` and up to date.
    
    Args:
        repo_or_path: HuggingFace repo ID or local directory path.
        category: 'backbone' or 'codec' (inferred if None).
        check_update: If True, queries HF Hub for the latest commit SHA.
        hf_token: Optional Hugging Face auth token.
        status_callback: Callable accepting status message strings.
        
    Returns:
        Path to the local directory containing model files.
    """
    clean_id = repo_or_path.strip()

    # 1. If it is already an existing local directory (e.g. finetune/output/hoat_ngon/merged), use it directly
    local_p = Path(clean_id)
    if local_p.is_dir():
        if status_callback:
            status_callback(f"📂 Dùng model local: {local_p}")
        return local_p.resolve()

    # Also check if it's relative to PROJECT_ROOT
    rel_p = PROJECT_ROOT / clean_id
    if rel_p.is_dir():
        if status_callback:
            status_callback(f"📂 Dùng model local: {rel_p}")
        return rel_p.resolve()

    # 2. Determine target path under model/
    cat, target_dir = get_model_destination(clean_id, category=category)
    local_hash = get_local_commit_hash(target_dir)
    dir_valid = is_model_dir_valid(target_dir, category=cat)

    remote_hash = None
    if check_update:
        if status_callback:
            status_callback(f"🔍 Đang kiểm tra phiên bản '{clean_id}' trên Hugging Face...")
        remote_hash = get_remote_commit_hash(clean_id, hf_token=hf_token)

    # 3. Decide whether download/update is required
    needs_download = False
    reason = ""

    if not dir_valid:
        needs_download = True
        reason = "model chưa có trong thư mục local"
    elif remote_hash and local_hash and remote_hash != local_hash:
        needs_download = True
        reason = f"có phiên bản mới trên Hugging Face ({remote_hash[:7]} != {local_hash[:7]})"
    elif remote_hash and not local_hash:
        # Files are already valid, write the hash file
        set_local_commit_hash(target_dir, remote_hash)
        clean_hf_cache(target_dir)

    # 4. Download / update if needed
    if needs_download:
        if status_callback:
            status_callback(f"⬇️ {reason.capitalize()}. Đang tải vào: {target_dir} ...")
        logger.info(f"Downloading/updating '{clean_id}' -> {target_dir} ({reason})")

        from huggingface_hub import snapshot_download

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=clean_id,
                local_dir=str(target_dir),
                token=hf_token,
            )
            clean_hf_cache(target_dir)
            if remote_hash:
                set_local_commit_hash(target_dir, remote_hash)
            elif not local_hash:
                # If remote_hash wasn't fetched earlier, try fetching once to save
                rh = get_remote_commit_hash(clean_id, hf_token=hf_token)
                if rh:
                    set_local_commit_hash(target_dir, rh)

            if status_callback:
                status_callback(f"✅ Đã tải và lưu thành công vào: {target_dir}")
        except Exception as e:
            # If download fails, check if we have a valid offline version
            clean_hf_cache(target_dir)
            if is_model_dir_valid(target_dir, category=cat):
                logger.warning(f"Download failed ({e}), falling back to existing local copy in {target_dir}")
                if status_callback:
                    status_callback(f"⚠️ Không tải được bản mới ({e}). Sử dụng bản hiện có trong {target_dir}")
            else:
                raise RuntimeError(
                    f"Không thể tải model '{clean_id}' về {target_dir}. Vui lòng kiểm tra kết nối mạng: {e}"
                ) from e
    else:
        # Up-to-date or offline valid
        clean_hf_cache(target_dir)
        if status_callback:
            if remote_hash and local_hash and remote_hash == local_hash:
                status_callback(f"✅ Model '{clean_id}' đã ở phiên bản mới nhất ({local_hash[:7]}).")
            else:
                status_callback(f"⚡ Đang dùng model local: {target_dir}")

    return target_dir.resolve()


def migrate_from_hf_cache(status_callback: Optional[Callable[[str], None]] = None) -> None:
    """
    Helper to copy pre-downloaded snapshots from ~/.cache/huggingface/hub into model/
    so the user does not need to re-download gigabytes of data.
    """
    hf_hub_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    if not hf_hub_cache.is_dir():
        return

    for repo_id, (cat, folder_name) in KNOWN_MODELS.items():
        target_dir = MODEL_DIR / cat / folder_name
        if is_model_dir_valid(target_dir, category=cat):
            clean_hf_cache(target_dir)
            continue

        # Find repo in HF cache
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        repo_cache_dir = hf_hub_cache / repo_cache_name / "snapshots"
        if not repo_cache_dir.is_dir():
            continue

        # Find the latest snapshot folder
        snapshots = [s for s in repo_cache_dir.iterdir() if s.is_dir()]
        if not snapshots:
            continue

        # Sort by modification time descending
        snapshots.sort(key=lambda s: s.stat().st_mtime, reverse=True)
        latest_snapshot = snapshots[0]
        commit_sha = latest_snapshot.name

        msg = f"📦 Đang chuyển model có sẵn '{folder_name}' sang {target_dir}..."
        if status_callback:
            status_callback(msg)
        logger.info(msg)

        target_dir.mkdir(parents=True, exist_ok=True)
        # Copy files (dereferencing symlinks so they become real standalone files)
        for item in latest_snapshot.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item.resolve(), dest, symlinks=False)
            else:
                shutil.copy2(item.resolve(), dest)

        set_local_commit_hash(target_dir, commit_sha)
        clean_hf_cache(target_dir)
        done_msg = f"✅ Đã chuyển '{folder_name}' sang {target_dir} (version: {commit_sha[:7]})"
        if status_callback:
            status_callback(done_msg)
        logger.info(done_msg)
