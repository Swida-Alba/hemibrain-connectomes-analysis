"""Lightweight client for the public NeuronBridge data API.

The upstream ``neuronbridge-python`` package currently pulls in Ray, Memray,
and Pydantic 2.9 even though DROCAT only needs the read-only metadata client.
Those dependencies conflict with DROCAT's current validation and UI runtime.
This module implements the small, stable subset of that client locally using
``requests`` and Pillow, both of which are already core DROCAT dependencies.

The returned JSON objects support attribute access (``image.files.CDSResults``)
so callers remain compatible with the upstream client without depending on its
Pydantic models.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional
from urllib.parse import quote, urljoin

import requests
from PIL import Image


DEFAULT_DATA_BUCKET = "janelia-neuronbridge-data-prod"
DEFAULT_TIMEOUT = 30


class APIObject:
    """Recursively attribute-accessible representation of a JSON object."""

    def __init__(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self, key, _to_api_object(value))

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"APIObject({fields})"

    # NeuronBridge config contains named mappings (stores, prefixes). Supporting
    # both attribute and mapping access keeps those structures convenient while
    # preserving the object API used for images and matches.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, _to_api_object(value))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return vars(self).items()


def _to_api_object(value: Any) -> Any:
    if isinstance(value, dict):
        return APIObject(**value)
    if isinstance(value, list):
        return [_to_api_object(item) for item in value]
    return value


class Client:
    """Read-only NeuronBridge client compatible with DROCAT's usage."""

    def __init__(
        self,
        data_bucket: str = DEFAULT_DATA_BUCKET,
        version: str = "current",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        data_url_prefix = f"https://{data_bucket}.s3.us-east-1.amazonaws.com"

        if version == "current":
            response = self._get(f"{data_url_prefix}/current.txt")
            try:
                version = response.text.strip()
            finally:
                response.close()
            if not version:
                raise RuntimeError("NeuronBridge current.txt returned an empty version")

        self.version = version
        self.data_url = f"{data_url_prefix}/{version}"
        self.config = _to_api_object(self._get_json(f"{self.data_url}/config.json"))

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        response = self.session.get(url, **kwargs)
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            response.close()
            raise RuntimeError(f"Could not retrieve {url}: {exc}") from exc
        return response

    def _get_json(self, url: str) -> Any:
        response = self._get(url)
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"NeuronBridge returned invalid JSON from {url}") from exc
        finally:
            response.close()

    def _get_image(self, url: str) -> Image.Image:
        response = self._get(url, stream=True)
        try:
            # Pillow loads lazily by default. Detach the image from the HTTP
            # response so callers can safely inspect or save it later.
            image = Image.open(BytesIO(response.content))
            image.load()
            return image
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    def _get_text(self, url: str) -> str:
        response = self._get(url)
        try:
            return response.text
        finally:
            response.close()

    def _get_files_url(self, files: APIObject, file_key: str) -> Optional[str]:
        path = getattr(files, file_key, None)
        if not path:
            return None
        if str(path).startswith(("http://", "https://")):
            return str(path)

        store_name = getattr(files, "store", "")
        stores = getattr(self.config, "stores", {})
        store = stores.get(store_name) if hasattr(stores, "get") else None
        prefixes = getattr(store, "prefixes", {}) if store else {}
        prefix = prefixes.get(file_key, "") if hasattr(prefixes, "get") else ""
        if not prefix:
            raise RuntimeError(f"NeuronBridge config has no prefix for '{file_key}'")
        return urljoin(prefix.rstrip("/") + "/", str(path).lstrip("/"))

    def _get_match_url(self, match: APIObject, file_key: str) -> str:
        for owner in (match, getattr(match, "image", None)):
            files = getattr(owner, "files", None) if owner is not None else None
            if files is not None:
                url = self._get_files_url(files, file_key)
                if url:
                    return url
        raise RuntimeError(f"Match contains no file with type '{file_key}'")

    def get_em_images(self, body_id: Any) -> list[APIObject]:
        encoded_id = quote(str(body_id), safe="")
        payload = self._get_json(f"{self.data_url}/metadata/by_body/{encoded_id}.json")
        return _to_api_object(payload).results

    def get_em_image(self, body_id: Any) -> Optional[APIObject]:
        images = self.get_em_images(body_id)
        return images[0] if images else None

    def get_lm_images(self, line_id: str) -> list[APIObject]:
        encoded_id = quote(str(line_id), safe="")
        payload = self._get_json(f"{self.data_url}/metadata/by_line/{encoded_id}.json")
        return _to_api_object(payload).results

    def _get_matches(self, neuron_image: APIObject, file_key: str) -> list[APIObject]:
        url = self._get_files_url(neuron_image.files, file_key)
        if not url:
            return []
        return _to_api_object(self._get_json(url)).results

    def get_cds_matches(self, neuron_image: APIObject) -> list[APIObject]:
        return self._get_matches(neuron_image, "CDSResults")

    def get_ppp_matches(self, neuron_image: APIObject) -> list[APIObject]:
        return self._get_matches(neuron_image, "PPPMResults")

    def get_cds_image(self, match: APIObject, thumbnail: bool = False) -> Image.Image:
        return self._get_image(self._get_match_url(match, "CDMThumbnail" if thumbnail else "CDM"))

    def get_target_searchable_image(self, match: APIObject) -> Image.Image:
        return self._get_image(self._get_match_url(match, "CDMInput"))

    def get_match_searchable_image(self, match: APIObject) -> Image.Image:
        return self._get_image(self._get_match_url(match, "CDMMatch"))

    def get_ppp_image(self, match: APIObject, thumbnail: bool = False) -> Image.Image:
        key = "CDMBestThumbnail" if thumbnail else "CDMBest"
        return self._get_image(self._get_match_url(match, key))

    def get_swc_skeleton(self, match: APIObject) -> str:
        # SWC is a plain-text skeleton format, not an image; return the raw
        # text so callers can parse or save it directly.
        return self._get_text(self._get_match_url(match, "AlignedBodySWC"))

    def get_image_stack(self, match: APIObject) -> Image.Image:
        return self._get_image(self._get_match_url(match, "VisuallyLosslessStack"))
