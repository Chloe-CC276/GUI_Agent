"""
base_vlm.py

Provider-independent base interface for Vision-Language Models (VLMs).

Goals
-----
1. Provide one unified interface for text-only and multimodal requests.
2. Keep provider-specific code outside the base class.
3. Normalize local images, URLs, bytes, and data URLs.
4. Support synchronous and asynchronous generation.
5. Support retries, timeout configuration, structured JSON extraction,
   usage statistics, request/response metadata, and conversation history.
6. Avoid forcing a particular HTTP client or SDK on subclasses.

Typical inheritance
-------------------
class OpenAIVLM(BaseVLM):
    def _generate_once(self, request: VLMRequest) -> VLMResponse:
        ...

class QwenVLM(BaseVLM):
    def _generate_once(self, request: VLMRequest) -> VLMResponse:
        ...
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class VLMError(RuntimeError):
    """Base exception for VLM-related failures."""


class VLMConfigurationError(VLMError):
    """Raised when model or request configuration is invalid."""


class VLMRequestError(VLMError):
    """Raised when a VLM request is invalid."""


class VLMProviderError(VLMError):
    """Raised when the remote/local provider call fails."""


class VLMResponseError(VLMError):
    """Raised when the provider returns an invalid response."""


class VLMStructuredOutputError(VLMResponseError):
    """Raised when structured output cannot be parsed."""


# ============================================================
# Enums
# ============================================================

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class ImageSourceType(str, Enum):
    URL = "url"
    DATA_URL = "data_url"
    LOCAL_FILE = "local_file"
    BYTES = "bytes"


# ============================================================
# Image input
# ============================================================

@dataclass(frozen=True)
class ImageInput:
    """
    Provider-independent image input.

    Only one of ``url``, ``path``, ``data``, or ``data_url`` should be set.
    Use the factory methods instead of constructing this class manually.
    """

    source_type: ImageSourceType
    url: str | None = None
    path: Path | None = None
    data: bytes | None = None
    data_url: str | None = None
    mime_type: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        detail: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ImageInput":
        if not isinstance(url, str) or not url.strip():
            raise VLMRequestError("Image URL must be a non-empty string.")

        text = url.strip()

        if text.startswith("data:"):
            return cls.from_data_url(
                text,
                detail=detail,
                metadata=metadata,
            )

        if not re.match(r"^https?://", text, flags=re.IGNORECASE):
            raise VLMRequestError(
                "Remote image URL must start with http:// or https://."
            )

        return cls(
            source_type=ImageSourceType.URL,
            url=text,
            detail=detail,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        detail: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ImageInput":
        resolved = Path(path).expanduser().resolve()

        if not resolved.is_file():
            raise FileNotFoundError(f"Image file not found: {resolved}")

        mime_type, _ = mimetypes.guess_type(str(resolved))

        if mime_type is None:
            mime_type = "application/octet-stream"

        if not mime_type.startswith("image/"):
            raise VLMRequestError(
                f"File does not appear to be an image: {resolved}"
            )

        return cls(
            source_type=ImageSourceType.LOCAL_FILE,
            path=resolved,
            mime_type=mime_type,
            detail=detail,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime_type: str = "image/png",
        detail: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ImageInput":
        if not isinstance(data, bytes) or not data:
            raise VLMRequestError("Image bytes must be non-empty bytes.")

        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise VLMRequestError(
                "mime_type must be a valid image MIME type."
            )

        return cls(
            source_type=ImageSourceType.BYTES,
            data=data,
            mime_type=mime_type,
            detail=detail,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_data_url(
        cls,
        data_url: str,
        *,
        detail: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ImageInput":
        if not isinstance(data_url, str) or not data_url.startswith("data:"):
            raise VLMRequestError("Invalid image data URL.")

        match = re.match(
            r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$",
            data_url,
            flags=re.DOTALL,
        )

        if match is None:
            raise VLMRequestError(
                "Only base64 encoded image data URLs are supported."
            )

        mime_type = match.group(1)

        return cls(
            source_type=ImageSourceType.DATA_URL,
            data_url=data_url,
            mime_type=mime_type,
            detail=detail,
            metadata=dict(metadata or {}),
        )

    def read_bytes(self) -> bytes:
        if self.source_type == ImageSourceType.LOCAL_FILE:
            if self.path is None:
                raise VLMRequestError("Local image path is missing.")

            return self.path.read_bytes()

        if self.source_type == ImageSourceType.BYTES:
            if self.data is None:
                raise VLMRequestError("Image byte content is missing.")

            return self.data

        if self.source_type == ImageSourceType.DATA_URL:
            if self.data_url is None:
                raise VLMRequestError("Image data URL is missing.")

            encoded = self.data_url.split(",", maxsplit=1)[1]

            try:
                return base64.b64decode(encoded, validate=True)
            except Exception as error:
                raise VLMRequestError(
                    "Image data URL contains invalid base64 data."
                ) from error

        raise VLMRequestError(
            "Remote URL images cannot be read without a network client."
        )

    def to_data_url(self) -> str:
        """
        Convert local file or bytes input to a base64 data URL.

        Remote URLs are returned unchanged by ``provider_value`` rather than
        downloaded here. This keeps network logic inside provider subclasses.
        """
        if self.source_type == ImageSourceType.DATA_URL:
            if self.data_url is None:
                raise VLMRequestError("Image data URL is missing.")

            return self.data_url

        if self.source_type == ImageSourceType.URL:
            raise VLMRequestError(
                "Remote URL cannot be converted to a data URL without download."
            )

        mime_type = self.mime_type or "image/png"
        encoded = base64.b64encode(self.read_bytes()).decode("ascii")

        return f"data:{mime_type};base64,{encoded}"

    def provider_value(
        self,
        *,
        local_as_data_url: bool = True,
    ) -> str:
        if self.source_type == ImageSourceType.URL:
            if self.url is None:
                raise VLMRequestError("Image URL is missing.")

            return self.url

        if local_as_data_url:
            return self.to_data_url()

        if self.path is not None:
            return str(self.path)

        return self.to_data_url()

    def to_dict(
        self,
        *,
        include_binary: bool = False,
    ) -> dict[str, Any]:
        result = {
            "source_type": self.source_type.value,
            "url": self.url,
            "path": str(self.path) if self.path is not None else None,
            "data_url": self.data_url,
            "mime_type": self.mime_type,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }

        if include_binary and self.data is not None:
            result["data_base64"] = base64.b64encode(
                self.data
            ).decode("ascii")

        return result


ImageLike = ImageInput | str | Path | bytes


def normalise_image(
    image: ImageLike,
    *,
    mime_type: str = "image/png",
    detail: str | None = None,
) -> ImageInput:
    """Convert a supported image value into ``ImageInput``."""
    if isinstance(image, ImageInput):
        return image

    if isinstance(image, bytes):
        return ImageInput.from_bytes(
            image,
            mime_type=mime_type,
            detail=detail,
        )

    if isinstance(image, Path):
        return ImageInput.from_path(
            image,
            detail=detail,
        )

    if isinstance(image, str):
        text = image.strip()

        if text.startswith("data:"):
            return ImageInput.from_data_url(
                text,
                detail=detail,
            )

        if re.match(r"^https?://", text, flags=re.IGNORECASE):
            return ImageInput.from_url(
                text,
                detail=detail,
            )

        return ImageInput.from_path(
            text,
            detail=detail,
        )

    raise TypeError(
        "image must be ImageInput, str, pathlib.Path, or bytes."
    )


# ============================================================
# Message content
# ============================================================

@dataclass(frozen=True)
class TextContent:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise VLMRequestError("Text content must be a non-empty string.")

    @property
    def type(self) -> ContentType:
        return ContentType.TEXT

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "text": self.text,
        }


@dataclass(frozen=True)
class ImageContent:
    image: ImageInput

    @property
    def type(self) -> ContentType:
        return ContentType.IMAGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "image": self.image.to_dict(),
        }


MessageContent = TextContent | ImageContent


@dataclass
class VLMMessage:
    role: MessageRole | str
    content: list[MessageContent]
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            try:
                self.role = MessageRole(self.role.strip().lower())
            except ValueError as error:
                raise VLMRequestError(
                    f"Unsupported message role: {self.role!r}"
                ) from error

        if not isinstance(self.content, list) or not self.content:
            raise VLMRequestError(
                "Message content must be a non-empty list."
            )

        if not all(
            isinstance(item, (TextContent, ImageContent))
            for item in self.content
        ):
            raise VLMRequestError(
                "Message content contains unsupported item types."
            )

    @classmethod
    def text(
        cls,
        role: MessageRole | str,
        text: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VLMMessage":
        return cls(
            role=role,
            content=[TextContent(text)],
            name=name,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def user(
        cls,
        text: str,
        images: Sequence[ImageLike] | None = None,
        *,
        image_detail: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VLMMessage":
        parts: list[MessageContent] = [TextContent(text)]

        for image in images or []:
            parts.append(
                ImageContent(
                    normalise_image(
                        image,
                        detail=image_detail,
                    )
                )
            )

        return cls(
            role=MessageRole.USER,
            content=parts,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def assistant(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VLMMessage":
        return cls.text(
            MessageRole.ASSISTANT,
            text,
            metadata=metadata,
        )

    @classmethod
    def system(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VLMMessage":
        return cls.text(
            MessageRole.SYSTEM,
            text,
            metadata=metadata,
        )

    def text_content(self) -> str:
        return "\n".join(
            item.text
            for item in self.content
            if isinstance(item, TextContent)
        )

    def images(self) -> list[ImageInput]:
        return [
            item.image
            for item in self.content
            if isinstance(item, ImageContent)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "name": self.name,
            "content": [
                item.to_dict()
                for item in self.content
            ],
            "metadata": dict(self.metadata),
        }


# ============================================================
# Request and response schema
# ============================================================

@dataclass
class VLMGenerationConfig:
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    timeout: float = 60.0
    max_retries: int = 2
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0
    response_format: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature,
            (int, float),
        ):
            raise VLMConfigurationError(
                "temperature must be numeric."
            )

        if self.temperature < 0:
            raise VLMConfigurationError(
                "temperature must be non-negative."
            )

        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(
                self.top_p,
                (int, float),
            ):
                raise VLMConfigurationError(
                    "top_p must be numeric or None."
                )

            if not 0 < self.top_p <= 1:
                raise VLMConfigurationError(
                    "top_p must be in (0, 1]."
                )

        if self.max_tokens is not None:
            if (
                not isinstance(self.max_tokens, int)
                or isinstance(self.max_tokens, bool)
                or self.max_tokens <= 0
            ):
                raise VLMConfigurationError(
                    "max_tokens must be a positive integer or None."
                )

        if (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or self.timeout <= 0
        ):
            raise VLMConfigurationError(
                "timeout must be a positive number."
            )

        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise VLMConfigurationError(
                "max_retries must be a non-negative integer."
            )

        if (
            not isinstance(self.retry_base_delay, (int, float))
            or self.retry_base_delay < 0
        ):
            raise VLMConfigurationError(
                "retry_base_delay must be non-negative."
            )

        if (
            not isinstance(self.retry_max_delay, (int, float))
            or self.retry_max_delay < self.retry_base_delay
        ):
            raise VLMConfigurationError(
                "retry_max_delay must be >= retry_base_delay."
            )

        if not isinstance(self.stop, list) or not all(
            isinstance(item, str)
            for item in self.stop
        ):
            raise VLMConfigurationError(
                "stop must be a list of strings."
            )

    def merged(
        self,
        **overrides: Any,
    ) -> "VLMGenerationConfig":
        """
        Return a validated copy with selected fields overridden.

        Unknown override keys are stored in ``extra``.
        """
        known_fields = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "timeout",
            "max_retries",
            "retry_base_delay",
            "retry_max_delay",
            "response_format",
        }

        replacement: dict[str, Any] = {}
        extra = dict(self.extra)

        for key, value in overrides.items():
            if key in known_fields:
                replacement[key] = value
            else:
                extra[key] = value

        replacement["extra"] = extra

        config = replace(self, **replacement)
        config.validate()

        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "seed": self.seed,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "response_format": self.response_format,
            "extra": dict(self.extra),
        }


@dataclass
class VLMRequest:
    messages: list[VLMMessage]
    model: str
    config: VLMGenerationConfig
    request_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise VLMRequestError("model must be a non-empty string.")

        if not isinstance(self.messages, list) or not self.messages:
            raise VLMRequestError(
                "messages must be a non-empty list."
            )

        if not all(
            isinstance(message, VLMMessage)
            for message in self.messages
        ):
            raise VLMRequestError(
                "messages must contain only VLMMessage objects."
            )

        self.config.validate()

    def image_count(self) -> int:
        return sum(
            len(message.images())
            for message in self.messages
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class VLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    image_tokens: int | None = None
    cost: float | None = None
    currency: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalise_total(self) -> None:
        if self.total_tokens is None:
            values = [
                value
                for value in (
                    self.input_tokens,
                    self.output_tokens,
                )
                if value is not None
            ]

            if values:
                self.total_tokens = sum(values)

    def to_dict(self) -> dict[str, Any]:
        self.normalise_total()

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "image_tokens": self.image_tokens,
            "cost": self.cost,
            "currency": self.currency,
            "metadata": dict(self.metadata),
        }


@dataclass
class VLMResponse:
    text: str
    model: str
    request_id: str
    provider: str
    usage: VLMUsage = field(default_factory=VLMUsage)
    finish_reason: str | None = None
    latency_seconds: float = 0.0
    raw_response: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.text, str):
            raise VLMResponseError(
                "Response text must be a string."
            )

        if not isinstance(self.model, str) or not self.model:
            raise VLMResponseError(
                "Response model must be a non-empty string."
            )

        if not isinstance(self.request_id, str) or not self.request_id:
            raise VLMResponseError(
                "Response request_id must be a non-empty string."
            )

        if not isinstance(self.provider, str) or not self.provider:
            raise VLMResponseError(
                "Response provider must be a non-empty string."
            )

    def json(
        self,
        *,
        allow_markdown_fence: bool = True,
    ) -> Any:
        return parse_json_text(
            self.text,
            allow_markdown_fence=allow_markdown_fence,
        )

    def to_dict(
        self,
        *,
        include_raw_response: bool = False,
    ) -> dict[str, Any]:
        result = {
            "text": self.text,
            "model": self.model,
            "request_id": self.request_id,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "latency_seconds": self.latency_seconds,
            "metadata": dict(self.metadata),
        }

        if include_raw_response:
            result["raw_response"] = self.raw_response

        return result


# ============================================================
# JSON helpers
# ============================================================

_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_json_text(
    text: str,
    *,
    allow_markdown_fence: bool = True,
) -> Any:
    if not isinstance(text, str):
        raise VLMStructuredOutputError(
            "Structured response must be a string."
        )

    candidate = text.strip()

    if not candidate:
        raise VLMStructuredOutputError(
            "Structured response is empty."
        )

    if allow_markdown_fence:
        match = _JSON_FENCE_PATTERN.search(candidate)

        if match is not None:
            candidate = match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the widest JSON object or array in surrounding prose.
    object_start = candidate.find("{")
    object_end = candidate.rfind("}")
    array_start = candidate.find("[")
    array_end = candidate.rfind("]")

    possible: list[str] = []

    if object_start >= 0 and object_end > object_start:
        possible.append(candidate[object_start : object_end + 1])

    if array_start >= 0 and array_end > array_start:
        possible.append(candidate[array_start : array_end + 1])

    for item in sorted(possible, key=len, reverse=True):
        try:
            return json.loads(item)
        except json.JSONDecodeError:
            continue

    raise VLMStructuredOutputError(
        "Model response does not contain valid JSON."
    )


# ============================================================
# Base VLM
# ============================================================

class BaseVLM(ABC):
    """
    Abstract VLM interface.

    Subclasses must implement ``_generate_once``.

    Async providers may override ``_agenerate_once``. If they do not, the
    default implementation runs the synchronous method in a worker thread.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_system_prompt: str | None = None,
        generation_config: VLMGenerationConfig | None = None,
        keep_history: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise VLMConfigurationError(
                "model must be a non-empty string."
            )

        if not isinstance(provider, str) or not provider.strip():
            raise VLMConfigurationError(
                "provider must be a non-empty string."
            )

        self.model = model.strip()
        self.provider = provider.strip().lower()
        self.api_key = api_key
        self.base_url = base_url
        self.default_system_prompt = (
            default_system_prompt.strip()
            if isinstance(default_system_prompt, str)
            and default_system_prompt.strip()
            else None
        )

        self.generation_config = (
            generation_config
            if generation_config is not None
            else VLMGenerationConfig()
        )
        self.generation_config.validate()

        self.keep_history = keep_history
        self.extra_headers = dict(extra_headers or {})
        self.metadata = dict(metadata or {})

        self._history: list[VLMMessage] = []
        self._response_history: list[VLMResponse] = []

    # ------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------

    def generate(
        self,
        prompt: str | None = None,
        *,
        images: Sequence[ImageLike] | None = None,
        messages: Sequence[VLMMessage | Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        config: VLMGenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        **generation_overrides: Any,
    ) -> VLMResponse:
        request = self.build_request(
            prompt=prompt,
            images=images,
            messages=messages,
            system_prompt=system_prompt,
            config=config,
            metadata=metadata,
            **generation_overrides,
        )

        return self.generate_request(request)

    async def agenerate(
        self,
        prompt: str | None = None,
        *,
        images: Sequence[ImageLike] | None = None,
        messages: Sequence[VLMMessage | Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        config: VLMGenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        **generation_overrides: Any,
    ) -> VLMResponse:
        request = self.build_request(
            prompt=prompt,
            images=images,
            messages=messages,
            system_prompt=system_prompt,
            config=config,
            metadata=metadata,
            **generation_overrides,
        )

        return await self.agenerate_request(request)

    def generate_json(
        self,
        prompt: str,
        *,
        images: Sequence[ImageLike] | None = None,
        schema: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        config: VLMGenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        **generation_overrides: Any,
    ) -> tuple[Any, VLMResponse]:
        structured_prompt = self._build_structured_prompt(
            prompt=prompt,
            schema=schema,
        )

        response = self.generate(
            prompt=structured_prompt,
            images=images,
            system_prompt=system_prompt,
            config=config,
            metadata=metadata,
            response_format="json",
            **generation_overrides,
        )

        return response.json(), response

    async def agenerate_json(
        self,
        prompt: str,
        *,
        images: Sequence[ImageLike] | None = None,
        schema: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        config: VLMGenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        **generation_overrides: Any,
    ) -> tuple[Any, VLMResponse]:
        structured_prompt = self._build_structured_prompt(
            prompt=prompt,
            schema=schema,
        )

        response = await self.agenerate(
            prompt=structured_prompt,
            images=images,
            system_prompt=system_prompt,
            config=config,
            metadata=metadata,
            response_format="json",
            **generation_overrides,
        )

        return response.json(), response

    # ------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------

    def build_request(
        self,
        prompt: str | None = None,
        *,
        images: Sequence[ImageLike] | None = None,
        messages: Sequence[VLMMessage | Mapping[str, Any]] | None = None,
        system_prompt: str | None = None,
        config: VLMGenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        **generation_overrides: Any,
    ) -> VLMRequest:
        if prompt is None and messages is None:
            raise VLMRequestError(
                "Either prompt or messages must be provided."
            )

        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise VLMRequestError(
                    "prompt must be a non-empty string."
                )

        resolved_config = (
            config if config is not None else self.generation_config
        )

        if generation_overrides:
            resolved_config = resolved_config.merged(
                **generation_overrides
            )
        else:
            resolved_config.validate()

        request_messages: list[VLMMessage] = []

        resolved_system_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str)
            and system_prompt.strip()
            else self.default_system_prompt
        )

        if resolved_system_prompt:
            request_messages.append(
                VLMMessage.system(
                    resolved_system_prompt
                )
            )

        if self.keep_history:
            request_messages.extend(self._history)

        if messages is not None:
            request_messages.extend(
                self._normalise_messages(messages)
            )

        if prompt is not None:
            request_messages.append(
                VLMMessage.user(
                    prompt.strip(),
                    images=images,
                )
            )
        elif images:
            raise VLMRequestError(
                "images can only be supplied together with prompt."
            )

        request = VLMRequest(
            messages=request_messages,
            model=self.model,
            config=resolved_config,
            metadata={
                **self.metadata,
                **dict(metadata or {}),
            },
        )
        request.validate()

        return request

    def _normalise_messages(
        self,
        messages: Sequence[VLMMessage | Mapping[str, Any]],
    ) -> list[VLMMessage]:
        result: list[VLMMessage] = []

        for index, message in enumerate(messages):
            if isinstance(message, VLMMessage):
                result.append(message)
                continue

            if not isinstance(message, Mapping):
                raise VLMRequestError(
                    f"Message {index} must be VLMMessage or mapping."
                )

            role = message.get("role")
            content = message.get("content")

            if isinstance(content, str):
                result.append(
                    VLMMessage.text(
                        role=role,
                        text=content,
                        name=self._optional_string(
                            message.get("name")
                        ),
                        metadata=message.get("metadata")
                        if isinstance(message.get("metadata"), Mapping)
                        else None,
                    )
                )
                continue

            if not isinstance(content, Sequence):
                raise VLMRequestError(
                    f"Message {index} has invalid content."
                )

            parts: list[MessageContent] = []

            for part_index, part in enumerate(content):
                if isinstance(part, (TextContent, ImageContent)):
                    parts.append(part)
                    continue

                if not isinstance(part, Mapping):
                    raise VLMRequestError(
                        f"Message {index} content {part_index} is invalid."
                    )

                part_type = str(part.get("type", "")).strip().lower()

                if part_type == ContentType.TEXT.value:
                    text = part.get("text")

                    if not isinstance(text, str):
                        raise VLMRequestError(
                            "Text content requires a string 'text' field."
                        )

                    parts.append(TextContent(text))
                    continue

                if part_type == ContentType.IMAGE.value:
                    image_value = part.get(
                        "image",
                        part.get("url", part.get("path")),
                    )

                    parts.append(
                        ImageContent(
                            normalise_image(
                                image_value,
                                detail=self._optional_string(
                                    part.get("detail")
                                ),
                            )
                        )
                    )
                    continue

                raise VLMRequestError(
                    f"Unsupported content type: {part_type!r}"
                )

            result.append(
                VLMMessage(
                    role=role,
                    content=parts,
                    name=self._optional_string(
                        message.get("name")
                    ),
                    metadata=dict(
                        message.get("metadata", {})
                    )
                    if isinstance(
                        message.get("metadata", {}),
                        Mapping,
                    )
                    else {},
                )
            )

        return result

    # ------------------------------------------------------------
    # Retry execution
    # ------------------------------------------------------------

    def generate_request(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        request.validate()
        start_time = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(
            request.config.max_retries + 1
        ):
            attempt_start = time.perf_counter()

            try:
                response = self._generate_once(request)

                if not isinstance(response, VLMResponse):
                    raise VLMResponseError(
                        "_generate_once must return VLMResponse."
                    )

                response.latency_seconds = (
                    response.latency_seconds
                    or time.perf_counter() - attempt_start
                )
                response.validate()

                self._record_success(
                    request=request,
                    response=response,
                )

                return response

            except Exception as error:
                last_error = error

                if not self._is_retryable(error):
                    break

                if attempt >= request.config.max_retries:
                    break

                delay = self._retry_delay(
                    attempt=attempt,
                    config=request.config,
                )

                LOGGER.warning(
                    "VLM request failed; retrying provider=%s "
                    "model=%s attempt=%d delay=%.2fs error=%s",
                    self.provider,
                    self.model,
                    attempt + 1,
                    delay,
                    error,
                )

                time.sleep(delay)

        elapsed = time.perf_counter() - start_time

        raise VLMProviderError(
            f"VLM request failed after "
            f"{request.config.max_retries + 1} attempt(s), "
            f"elapsed={elapsed:.3f}s: {last_error}"
        ) from last_error

    async def agenerate_request(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        request.validate()
        start_time = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(
            request.config.max_retries + 1
        ):
            attempt_start = time.perf_counter()

            try:
                response = await asyncio.wait_for(
                    self._agenerate_once(request),
                    timeout=request.config.timeout,
                )

                if not isinstance(response, VLMResponse):
                    raise VLMResponseError(
                        "_agenerate_once must return VLMResponse."
                    )

                response.latency_seconds = (
                    response.latency_seconds
                    or time.perf_counter() - attempt_start
                )
                response.validate()

                self._record_success(
                    request=request,
                    response=response,
                )

                return response

            except Exception as error:
                last_error = error

                if not self._is_retryable(error):
                    break

                if attempt >= request.config.max_retries:
                    break

                delay = self._retry_delay(
                    attempt=attempt,
                    config=request.config,
                )

                LOGGER.warning(
                    "Async VLM request failed; retrying provider=%s "
                    "model=%s attempt=%d delay=%.2fs error=%s",
                    self.provider,
                    self.model,
                    attempt + 1,
                    delay,
                    error,
                )

                await asyncio.sleep(delay)

        elapsed = time.perf_counter() - start_time

        raise VLMProviderError(
            f"Async VLM request failed after "
            f"{request.config.max_retries + 1} attempt(s), "
            f"elapsed={elapsed:.3f}s: {last_error}"
        ) from last_error

    @abstractmethod
    def _generate_once(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        """
        Perform one provider call.

        Subclasses should:
        1. Convert ``request.messages`` to provider format.
        2. Send the request.
        3. Convert the provider response to ``VLMResponse``.
        4. Raise ``VLMProviderError`` for provider/network failures.
        """

    async def _agenerate_once(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        """
        Default async implementation.

        Provider subclasses with a native async SDK should override this
        method. Otherwise the synchronous call runs in a worker thread.
        """
        return await asyncio.to_thread(
            self._generate_once,
            request,
        )

    # ------------------------------------------------------------
    # Retry policy hooks
    # ------------------------------------------------------------

    def _is_retryable(
        self,
        error: Exception,
    ) -> bool:
        """
        Override in subclasses when provider-specific status codes are known.

        Configuration, request, response-format, and structured-output errors
        are treated as non-retryable. Other failures are retryable by default.
        """
        return not isinstance(
            error,
            (
                VLMConfigurationError,
                VLMRequestError,
                VLMResponseError,
                VLMStructuredOutputError,
            ),
        )

    @staticmethod
    def _retry_delay(
        attempt: int,
        config: VLMGenerationConfig,
    ) -> float:
        exponential = config.retry_base_delay * (2 ** attempt)
        capped = min(exponential, config.retry_max_delay)

        # Small jitter prevents multiple workers retrying simultaneously.
        return capped + random.uniform(0.0, min(0.25, capped))

    # ------------------------------------------------------------
    # History
    # ------------------------------------------------------------

    @property
    def history(self) -> tuple[VLMMessage, ...]:
        return tuple(self._history)

    @property
    def response_history(self) -> tuple[VLMResponse, ...]:
        return tuple(self._response_history)

    def clear_history(self) -> None:
        self._history.clear()
        self._response_history.clear()

    def _record_success(
        self,
        request: VLMRequest,
        response: VLMResponse,
    ) -> None:
        self._response_history.append(response)

        if not self.keep_history:
            return

        # Do not duplicate default/system/history messages. Only append the
        # newest user-side messages plus the assistant response.
        request_messages = request.messages

        last_user_index = None

        for index in range(len(request_messages) - 1, -1, -1):
            if request_messages[index].role == MessageRole.USER:
                last_user_index = index
                break

        if last_user_index is not None:
            self._history.append(
                request_messages[last_user_index]
            )

        self._history.append(
            VLMMessage.assistant(response.text)
        )

    # ------------------------------------------------------------
    # Utility methods for provider subclasses
    # ------------------------------------------------------------

    def make_response(
        self,
        *,
        request: VLMRequest,
        text: str,
        usage: VLMUsage | None = None,
        finish_reason: str | None = None,
        raw_response: Any = None,
        metadata: Mapping[str, Any] | None = None,
        latency_seconds: float = 0.0,
        model: str | None = None,
    ) -> VLMResponse:
        response = VLMResponse(
            text=text,
            model=model or request.model,
            request_id=request.request_id,
            provider=self.provider,
            usage=usage or VLMUsage(),
            finish_reason=finish_reason,
            latency_seconds=latency_seconds,
            raw_response=raw_response,
            metadata=dict(metadata or {}),
        )
        response.validate()

        return response

    @staticmethod
    def flatten_text(
        messages: Iterable[VLMMessage],
    ) -> str:
        """
        Produce a readable text-only representation for logging or
        providers that do not support chat messages directly.
        """
        lines: list[str] = []

        for message in messages:
            text = message.text_content()
            image_count = len(message.images())

            suffix = (
                f" [images={image_count}]"
                if image_count
                else ""
            )

            lines.append(
                f"{message.role.value}: {text}{suffix}"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_structured_prompt(
        prompt: str,
        schema: Mapping[str, Any] | None,
    ) -> str:
        if schema is None:
            return (
                f"{prompt.strip()}\n\n"
                "Return only valid JSON. Do not include Markdown fences "
                "or explanatory text."
            )

        schema_text = json.dumps(
            dict(schema),
            ensure_ascii=False,
            indent=2,
        )

        return (
            f"{prompt.strip()}\n\n"
            "Return only valid JSON that follows this schema:\n"
            f"{schema_text}\n"
            "Do not include Markdown fences or explanatory text."
        )

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if isinstance(value, str):
            text = value.strip()
            return text or None

        return str(value)