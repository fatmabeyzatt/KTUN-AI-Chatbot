import os
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _env_bool(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_source():
    local_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    if local_model_path:
        resolved = Path(local_model_path).expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(
                "EMBEDDING_MODEL_PATH bulundu ancak klasor mevcut degil: "
                f"{resolved}"
            )
        return str(resolved), True

    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL), False


def _default_hf_home():
    return os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def _find_cached_snapshot(model_name):
    cache_root = Path(_default_hf_home()) / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not cache_root.exists():
        return None

    candidates = [path for path in cache_root.iterdir() if path.is_dir()]
    if not candidates:
        return None

    # Select the newest snapshot that has a model config.
    for snapshot in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        if (snapshot / "config.json").exists():
            return str(snapshot.resolve())
    return None


def _build_error_message(exc, source, is_local):
    error_text = str(exc).lower()
    connection_tokens = (
        "winerror 10013",
        "forbidden by its access permissions",
        "cannot send a request, as the client has been closed",
        "failed to establish a new connection",
        "connection timed out",
        "name or service not known",
        "https://huggingface.co",
    )
    is_connection_issue = any(token in error_text for token in connection_tokens)

    if is_connection_issue and not is_local:
        return (
            "Embedding modeli HuggingFace uzerinden indirilemedi (ag/proxy/firewall engeli).\n"
            "Cozum:\n"
            "1) EMBEDDING_MODEL_PATH ortam degiskenine local model klasoru verin.\n"
            "2) Veya internet erisimini acip modeli bir kez indirerek cache olusturun.\n"
            f"Kullanilan kaynak: {source}"
        )

    if is_local:
        return (
            "Local embedding modeli yuklenemedi.\n"
            f"Kontrol edin: EMBEDDING_MODEL_PATH={source}\n"
            "Model klasorunde config dosyalari ve agirliklarin tam oldugundan emin olun."
        )

    return f"Embedding modeli yuklenemedi: {source}\nOrijinal hata: {exc}"


def _load_embeddings_quietly(kwargs):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return HuggingFaceEmbeddings(**kwargs)


def create_embedding_model():
    source, is_local_source = _resolve_model_source()
    offline_mode = _env_bool("EMBEDDING_OFFLINE", default=False)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    if not is_local_source:
        cached_snapshot = _find_cached_snapshot(source)
        if cached_snapshot:
            source = cached_snapshot
            is_local_source = True

    model_kwargs = {}
    if offline_mode:
        if not is_local_source:
            cached_snapshot = _find_cached_snapshot(source)
            if cached_snapshot:
                source = cached_snapshot
                is_local_source = True
            else:
                raise RuntimeError(
                    "EMBEDDING_OFFLINE=true ancak local model bulunamadi.\n"
                    "Cozum:\n"
                    "1) EMBEDDING_MODEL_PATH ile model klasoru verin veya\n"
                    "2) once internet acikken modeli indirip cache olusturun."
                )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_kwargs["local_files_only"] = True

    kwargs = {"model_name": source}
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    cache_folder = os.getenv("HF_HOME")
    if cache_folder:
        kwargs["cache_folder"] = cache_folder

    try:
        return _load_embeddings_quietly(kwargs)
    except Exception as exc:
        if not is_local_source:
            cached_snapshot = _find_cached_snapshot(source)
            if cached_snapshot:
                retry_kwargs = dict(kwargs)
                retry_model_kwargs = dict(retry_kwargs.get("model_kwargs", {}))
                retry_model_kwargs["local_files_only"] = True
                retry_kwargs["model_name"] = cached_snapshot
                retry_kwargs["model_kwargs"] = retry_model_kwargs
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                try:
                    return _load_embeddings_quietly(retry_kwargs)
                except Exception as retry_exc:
                    raise RuntimeError(
                        _build_error_message(retry_exc, cached_snapshot, True)
                    ) from retry_exc
        raise RuntimeError(_build_error_message(exc, source, is_local_source)) from exc
