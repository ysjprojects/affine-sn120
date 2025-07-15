from __future__ import annotations

import json, gzip, io, os, datetime as dt
from typing import Any

import boto3

_BUCKET = os.getenv("TRACE_BUCKET", "affine-traces")
_S3 = boto3.client("s3")

__all__ = ["put_trace", "get_trace", "put_json"]


def put_json(prefix: str, name: str, obj: Any) -> None:
    """Upload *obj* as gz-compressed JSON to `{prefix}/{name}.json.gz`."""
    payload = gzip.compress(json.dumps(obj).encode("utf-8"), compresslevel=9)

    trace_dir = os.getenv("TRACE_DIR")
    if trace_dir:
        path = os.path.join(trace_dir, f"{prefix}/{name}.json.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
        return

    _S3.put_object(Bucket=_BUCKET, Key=f"{prefix}/{name}.json.gz", Body=payload, ContentEncoding="gzip")


def _key(env: str, hotkey: str, block: int) -> str:
    return f"{env}/{hotkey}/{block}.json.gz"


def put_trace(env: str, hotkey: str, block: int, eval: "af.Evaluation", r_before: float, r_after: float) -> None:  # type: ignore
    """Compress + upload a trace JSON to S3 (or local FS if $TRACE_DIR set)."""
    data: dict[str, Any] = {
        "ts": int(dt.datetime.utcnow().timestamp()),
        "env": env,
        "hotkey": hotkey,
        "block": block,
        "score": eval.score,
        "rating_before": r_before,
        "rating_after": r_after,
        "extra": eval.extra,
    }
    payload = gzip.compress(json.dumps(data).encode("utf-8"), compresslevel=9)

    # If TRACE_DIR env var provided → write to local dir (easier for local tests)
    trace_dir = os.getenv("TRACE_DIR")
    if trace_dir:
        path = os.path.join(trace_dir, _key(env, hotkey, block))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
        return

    _S3.put_object(Bucket=_BUCKET, Key=_key(env, hotkey, block), Body=payload, ContentEncoding="gzip")


def get_trace(env: str, hotkey: str, block: int) -> dict | None:
    try:
        trace_dir = os.getenv("TRACE_DIR")
        if trace_dir:
            path = os.path.join(trace_dir, _key(env, hotkey, block))
            if not os.path.exists(path):
                return None
            with gzip.open(path, "rb") as f:
                return json.load(f)
        obj = _S3.get_object(Bucket=_BUCKET, Key=_key(env, hotkey, block))
        with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as z:
            return json.load(z)
    except Exception:
        return None 