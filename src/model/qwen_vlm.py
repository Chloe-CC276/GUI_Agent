"""
qwen_vlm.py

Qwen-VL implementation supports:

- Text-only and multimodal messages
- Local images, image bytes, data URLs, and remote image URLs
- Synchronous and asynchronous calls
- BaseVLM retry, history, JSON parsing, and usage tracking
- Region-aware endpoint configuration
- Provider-specific ``extra_body`` parameters


Example
-------
from src.models.qwen_vlm import QwenVLM

vlm = QwenVLM(
    model="qwen3-vl-plus",
    region="beijing",
)

response = vlm.generate(
    prompt="Describe the current GUI and identify the Submit button.",
    images=["screen.png"],
)

print(response.text)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        OpenAI,
        RateLimitError,
    )
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(
        "QwenVLM requires the OpenAI Python SDK. "
        'Install it with: pip install "openai>=1.30.0"'
    ) from error

from .base_vlm import (
    BaseVLM,
    ImageContent,
    MessageRole,
    TextContent,
    VLMConfigurationError,
    VLMGenerationConfig,
    VLMMessage,
    VLMProviderError,
    VLMRequest,
    VLMRequestError,
    VLMResponse,
    VLMResponseError,
    VLMUsage,
)


LOGGER = logging.getLogger(__name__)


class QwenVLMConfigurationError(VLMConfigurationError):
    """Raised when Qwen-VL configuration is invalid."""


class QwenVLMProviderError(VLMProviderError):
    """Raised when Model Studio returns an API failure."""


class QwenVLMResponseError(VLMResponseError):
    """Raised when a Qwen-VL response cannot be normalized."""


class QwenVLM(BaseVLM):
    """
    Qwen Vision-Language Model adapter.

    Parameters
    ----------
    model:
        Model Studio model name, for example ``qwen3-vl-plus``.

    api_key:
        Model Studio API key. When omitted, ``DASHSCOPE_API_KEY`` is used.

    base_url:
        Complete OpenAI-compatible base URL. When omitted, the loader first
        reads ``DASHSCOPE_BASE_URL`` and then constructs an endpoint from
        ``region`` and ``workspace_id``.

    region:
        Endpoint region alias. Supported aliases:
        ``beijing``, ``singapore``, ``japan``, ``germany``, ``virginia``.

    workspace_id:
        Model Studio workspace ID. When omitted,
        ``DASHSCOPE_WORKSPACE_ID`` is used. It is required by the
        workspace-specific endpoint templates.

    default_system_prompt:
        Default system message inserted before each request.

    generation_config:
        Shared BaseVLM generation settings.

    keep_history:
        Preserve user/assistant messages between calls.

    image_detail:
        Optional provider image detail hint such as ``low``, ``high``,
        or ``auto``. The actual supported values depend on the selected model.

    enable_thinking:
        Optional Qwen reasoning switch. It is sent through ``extra_body``.
        Leave as None unless the selected model supports the parameter.

    extra_body:
        Provider-specific body fields included in every request.

    client / async_client:
        Optional preconfigured OpenAI-compatible clients, useful for tests,
        proxies, or custom transports.
    """

    REGION_ALIASES = {
        "cn": "beijing",
        "china": "beijing",
        "china-beijing": "beijing",
        "cn-beijing": "beijing",
        "beijing": "beijing",

        "sg": "singapore",
        "ap-southeast-1": "singapore",
        "singapore": "singapore",

        "jp": "japan",
        "tokyo": "japan",
        "ap-northeast-1": "japan",
        "japan": "japan",

        "de": "germany",
        "frankfurt": "germany",
        "eu-central-1": "germany",
        "germany": "germany",

        "us": "virginia",
        "usa": "virginia",
        "us-east": "virginia",
        "virginia": "virginia",
    }

    WORKSPACE_BASE_URLS = {
        "beijing": (
            "https://{workspace_id}.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        "singapore": (
            "https://{workspace_id}.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        "japan": (
            "https://{workspace_id}.ap-northeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        "germany": (
            "https://{workspace_id}.eu-central-1.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
    }

    GLOBAL_BASE_URLS = {
        "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    }

    def __init__(
        self,
        model: str = "qwen3-vl-plus",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: str = "beijing",
        workspace_id: str | None = None,
        default_system_prompt: str | None = None,
        generation_config: VLMGenerationConfig | None = None,
        keep_history: bool = False,
        image_detail: str | None = None,
        enable_thinking: bool | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        resolved_api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
        )

        if not resolved_api_key:
            raise QwenVLMConfigurationError(
                "Model Studio API key is missing. Set DASHSCOPE_API_KEY "
                "or pass api_key explicitly."
            )

        resolved_region = self._normalise_region(region)
        resolved_workspace_id = (
            workspace_id
            or os.getenv("DASHSCOPE_WORKSPACE_ID")
        )
        resolved_base_url = self._resolve_base_url(
            explicit_base_url=base_url,
            region=resolved_region,
            workspace_id=resolved_workspace_id,
        )

        if image_detail is not None:
            if (
                not isinstance(image_detail, str)
                or not image_detail.strip()
            ):
                raise QwenVLMConfigurationError(
                    "image_detail must be a non-empty string or None."
                )

            image_detail = image_detail.strip().lower()

        if (
            enable_thinking is not None
            and not isinstance(enable_thinking, bool)
        ):
            raise QwenVLMConfigurationError(
                "enable_thinking must be bool or None."
            )

        self.region = resolved_region
        self.workspace_id = resolved_workspace_id
        self.image_detail = image_detail
        self.enable_thinking = enable_thinking
        self.default_extra_body = dict(extra_body or {})

        super().__init__(
            model=model,
            provider="qwen",
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            default_system_prompt=default_system_prompt,
            generation_config=generation_config,
            keep_history=keep_history,
            extra_headers=extra_headers,
            metadata={
                "region": resolved_region,
                "workspace_id": resolved_workspace_id,
                **dict(metadata or {}),
            },
        )

        self._client = client or OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            default_headers=dict(extra_headers or {}),
            timeout=90.0,
            max_retries=0,
        )

        self._async_client = async_client or AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            default_headers=dict(extra_headers or {}),
            timeout=30.0,
            max_retries=0,
        )

    # ============================================================
    # Base URL configuration
    # ============================================================

    @classmethod
    def _normalise_region(
        cls,
        region: str,
    ) -> str:
        if not isinstance(region, str) or not region.strip():
            raise QwenVLMConfigurationError(
                "region must be a non-empty string."
            )

        key = region.strip().lower()
        resolved = cls.REGION_ALIASES.get(key)

        if resolved is None:
            raise QwenVLMConfigurationError(
                f"Unsupported Model Studio region: {region!r}. "
                f"Supported regions: "
                f"{sorted(set(cls.REGION_ALIASES.values()))}"
            )

        return resolved

    @classmethod
    def _resolve_base_url(
        cls,
        *,
        explicit_base_url: str | None,
        region: str,
        workspace_id: str | None,
    ) -> str:
        environment_url = os.getenv("DASHSCOPE_BASE_URL")

        candidate = explicit_base_url or environment_url

        if candidate:
            if not isinstance(candidate, str) or not candidate.strip():
                raise QwenVLMConfigurationError(
                    "base_url must be a non-empty string."
                )

            return candidate.strip().rstrip("/")

        if region in cls.GLOBAL_BASE_URLS:
            return cls.GLOBAL_BASE_URLS[region]

        template = cls.WORKSPACE_BASE_URLS.get(region)

        if template is None:
            raise QwenVLMConfigurationError(
                f"No endpoint template configured for region {region!r}."
            )

        if not workspace_id:
            raise QwenVLMConfigurationError(
                f"workspace_id is required for region {region!r}. "
                "Set DASHSCOPE_WORKSPACE_ID, pass workspace_id, or provide "
                "DASHSCOPE_BASE_URL/base_url directly."
            )

        workspace_text = str(workspace_id).strip()

        if not workspace_text:
            raise QwenVLMConfigurationError(
                "workspace_id must not be empty."
            )

        return template.format(
            workspace_id=workspace_text
        )

    # ============================================================
    # Provider calls
    # ============================================================

    def _generate_once(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        payload = self._build_completion_payload(request)
        start_time = time.perf_counter()

        try:
            completion = self._client.chat.completions.create(
                **payload
            )
        except Exception as error:
            raise self._convert_provider_error(error) from error

        latency = time.perf_counter() - start_time

        return self._completion_to_response(
            request=request,
            completion=completion,
            latency_seconds=latency,
        )

    async def _agenerate_once(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        payload = self._build_completion_payload(request)
        start_time = time.perf_counter()

        try:
            completion = await self._async_client.chat.completions.create(
                **payload
            )
        except Exception as error:
            raise self._convert_provider_error(error) from error

        latency = time.perf_counter() - start_time

        return self._completion_to_response(
            request=request,
            completion=completion,
            latency_seconds=latency,
        )

    # ============================================================
    # Request conversion
    # ============================================================

    def _build_completion_payload(
        self,
        request: VLMRequest,
    ) -> dict[str, Any]:
        config = request.config

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                self._message_to_openai(message)
                for message in request.messages
            ],
            "temperature": config.temperature,
            "timeout": config.timeout,
        }

        if config.top_p is not None:
            payload["top_p"] = config.top_p

        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens

        if config.stop:
            payload["stop"] = list(config.stop)

        if config.seed is not None:
            payload["seed"] = config.seed

        response_format = config.response_format

        if response_format:
            payload["response_format"] = (
                {"type": "json_object"}
                if response_format.lower()
                in {"json", "json_object"}
                else {"type": response_format}
            )

        extra_body = dict(self.default_extra_body)

        if self.enable_thinking is not None:
            extra_body.setdefault(
                "enable_thinking",
                self.enable_thinking,
            )

        # BaseVLM stores unknown generation options in config.extra.
        request_extra_body = config.extra.get("extra_body")

        if request_extra_body is not None:
            if not isinstance(request_extra_body, Mapping):
                raise VLMRequestError(
                    "generation option extra_body must be a mapping."
                )

            extra_body.update(dict(request_extra_body))

        for key, value in config.extra.items():
            if key == "extra_body":
                continue

            # Standard OpenAI SDK arguments may be supplied directly.
            if key in {
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "logprobs",
                "n",
                "parallel_tool_calls",
                "service_tier",
                "tools",
                "tool_choice",
                "top_logprobs",
                "user",
            }:
                payload[key] = value
            else:
                # Unknown provider fields are safest in extra_body.
                extra_body[key] = value

        if extra_body:
            payload["extra_body"] = extra_body

        return payload

    def _message_to_openai(
        self,
        message: VLMMessage,
    ) -> dict[str, Any]:
        role = message.role.value

        if (
            message.role in {
                MessageRole.SYSTEM,
                MessageRole.ASSISTANT,
                MessageRole.TOOL,
            }
            and not message.images()
        ):
            result: dict[str, Any] = {
                "role": role,
                "content": message.text_content(),
            }

            if message.name:
                result["name"] = message.name

            return result

        content: list[dict[str, Any]] = []

        for item in message.content:
            if isinstance(item, TextContent):
                content.append(
                    {
                        "type": "text",
                        "text": item.text,
                    }
                )
                continue

            if isinstance(item, ImageContent):
                image = item.image
                detail = image.detail or self.image_detail
                image_url: dict[str, Any] = {
                    "url": image.provider_value(
                        local_as_data_url=True
                    )
                }

                if detail:
                    image_url["detail"] = detail

                content.append(
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    }
                )
                continue

            raise VLMRequestError(
                f"Unsupported message content: {type(item).__name__}"
            )

        result = {
            "role": role,
            "content": content,
        }

        if message.name:
            result["name"] = message.name

        return result

    # ============================================================
    # Response conversion
    # ============================================================

    def _completion_to_response(
        self,
        *,
        request: VLMRequest,
        completion: Any,
        latency_seconds: float,
    ) -> VLMResponse:
        choices = getattr(completion, "choices", None)

        if not choices:
            raise QwenVLMResponseError(
                "Qwen-VL response contains no choices."
            )

        choice = choices[0]
        message = getattr(choice, "message", None)

        if message is None:
            raise QwenVLMResponseError(
                "Qwen-VL response choice contains no message."
            )

        text = self._extract_message_text(
            getattr(message, "content", None)
        )

        usage = self._extract_usage(
            getattr(completion, "usage", None)
        )

        response_model = (
            getattr(completion, "model", None)
            or request.model
        )

        finish_reason = getattr(
            choice,
            "finish_reason",
            None,
        )

        provider_request_id = getattr(
            completion,
            "id",
            None,
        )

        return self.make_response(
            request=request,
            text=text,
            model=str(response_model),
            usage=usage,
            finish_reason=(
                str(finish_reason)
                if finish_reason is not None
                else None
            ),
            raw_response=completion,
            latency_seconds=latency_seconds,
            metadata={
                "provider_request_id": provider_request_id,
                "region": self.region,
                "base_url": self.base_url,
            },
        )

    @staticmethod
    def _extract_message_text(
        content: Any,
    ) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue

                if isinstance(item, Mapping):
                    text = item.get("text")

                    if isinstance(text, str):
                        text_parts.append(text)

                    continue

                text = getattr(item, "text", None)

                if isinstance(text, str):
                    text_parts.append(text)

            return "\n".join(text_parts)

        return str(content)

    @staticmethod
    def _extract_usage(
        provider_usage: Any,
    ) -> VLMUsage:
        if provider_usage is None:
            return VLMUsage()

        def get_value(*names: str) -> int | None:
            for name in names:
                if isinstance(provider_usage, Mapping):
                    value = provider_usage.get(name)
                else:
                    value = getattr(provider_usage, name, None)

                if isinstance(value, int):
                    return value

            return None

        usage = VLMUsage(
            input_tokens=get_value(
                "prompt_tokens",
                "input_tokens",
            ),
            output_tokens=get_value(
                "completion_tokens",
                "output_tokens",
            ),
            total_tokens=get_value(
                "total_tokens",
            ),
        )
        usage.normalise_total()

        return usage

    # ============================================================
    # Error and retry policy
    # ============================================================

    @staticmethod
    def _convert_provider_error(
        error: Exception,
    ) -> QwenVLMProviderError:
        status_code = getattr(error, "status_code", None)
        request_id = getattr(error, "request_id", None)

        details = [str(error)]

        if status_code is not None:
            details.append(f"status_code={status_code}")

        if request_id is not None:
            details.append(f"request_id={request_id}")

        converted = QwenVLMProviderError(
            "Qwen-VL provider request failed: "
            + ", ".join(details)
        )

        setattr(converted, "status_code", status_code)
        setattr(converted, "provider_request_id", request_id)
        setattr(converted, "original_error", error)

        return converted

    def _is_retryable(
        self,
        error: Exception,
    ) -> bool:
        if not isinstance(error, QwenVLMProviderError):
            return super()._is_retryable(error)

        original_error = getattr(
            error,
            "original_error",
            None,
        )

        if isinstance(
            original_error,
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
            ),
        ):
            return True

        status_code = getattr(
            error,
            "status_code",
            None,
        )

        if status_code in {
            408,
            409,
            429,
            500,
            502,
            503,
            504,
        }:
            return True

        if isinstance(original_error, APIStatusError):
            return False

        return False

    # ============================================================
    # Lifecycle
    # ============================================================

    def close(self) -> None:
        """Close the synchronous OpenAI-compatible client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous OpenAI-compatible client."""
        await self._async_client.close()

    def __enter__(self) -> "QwenVLM":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    async def __aenter__(self) -> "QwenVLM":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"region={self.region!r}, "
            f"base_url={self.base_url!r}, "
            f"keep_history={self.keep_history}"
            f")"
        )