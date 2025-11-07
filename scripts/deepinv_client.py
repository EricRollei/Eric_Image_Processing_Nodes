"""Utility client for communicating with the external DeepInv FastAPI service."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests

_DEFAULT_URL = os.environ.get("DEEPINV_SERVICE_URL", "http://127.0.0.1:6112")
_HEALTH_ENDPOINT = "/health"
_DENOISE_ENDPOINT = "/denoise"


class DeepInvServiceError(RuntimeError):
    pass


class DeepInvServiceClient:
    def __init__(self, base_url: str = _DEFAULT_URL, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._last_health_check: float = 0.0
        self._health_cache_success: bool = False

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:  # pragma: no cover - network error path
            detail = ""
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    text = resp.text
                    if text:
                        detail = f" | Response: {text.strip()}"
                except Exception:  # pragma: no cover - best effort logging
                    detail = ""
            raise DeepInvServiceError(f"DeepInv service request failed: {exc}{detail}") from exc

    def ensure_alive(self, force: bool = False) -> None:
        now = time.time()
        if not force and self._health_cache_success and (now - self._last_health_check) < 15.0:
            return
        response = self._request("GET", _HEALTH_ENDPOINT)
        payload = response.json()
        if payload.get("status") != "ok":
            raise DeepInvServiceError(f"DeepInv service unhealthy: {payload}")
        self._last_health_check = now
        self._health_cache_success = True

    def denoise(self, input_path: Path, output_path: Path, model: str, sigma: float, prefer_gpu: bool) -> Dict[str, Any]:
        self.ensure_alive()
        json_payload = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model": model,
            "sigma": sigma,
            "prefer_gpu": prefer_gpu,
        }
        response = self._request("POST", _DENOISE_ENDPOINT, json=json_payload)
        return response.json()


_client: Optional[DeepInvServiceClient] = None


def get_client() -> DeepInvServiceClient:
    global _client
    if _client is None:
        _client = DeepInvServiceClient()
    return _client


def save_tensor(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def load_tensor(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    return np.ascontiguousarray(data)
