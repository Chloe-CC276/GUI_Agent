"""
Local HuggingFace Qwen2.5-VL (+ optional LoRA/PEFT) backend for GUI Agent eval.

Designed for Colab / HPC GPU runs. Does not fall back to cloud DashScope.
Enable via ``--allow-local-adapter`` (see ``src.evaluation.variants``).
"""

from __future__ import annotations

import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from .base_vlm import (
    BaseVLM,
    ImageContent,
    ImageInput,
    ImageSourceType,
    MessageRole,
    TextContent,
    VLMConfigurationError,
    VLMGenerationConfig,
    VLMProviderError,
    VLMRequest,
    VLMResponse,
    VLMUsage,
)

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_torch_dtype(torch_mod: Any, name: str | None) -> Any:
    key = (name or os.environ.get("GUI_AGENT_TORCH_DTYPE") or "bfloat16").strip().lower()
    mapping = {
        "bf16": torch_mod.bfloat16,
        "bfloat16": torch_mod.bfloat16,
        "fp16": torch_mod.float16,
        "float16": torch_mod.float16,
        "fp32": torch_mod.float32,
        "float32": torch_mod.float32,
    }
    if key not in mapping:
        raise VLMConfigurationError(
            f"Unsupported torch dtype {key!r}; use bfloat16|float16|float32"
        )
    return mapping[key]


def image_input_to_hf(image: ImageInput) -> Any:
    """Convert ImageInput to a value accepted by qwen_vl_utils / AutoProcessor."""

    if image.source_type == ImageSourceType.LOCAL_FILE:
        if image.path is None:
            raise VLMProviderError("Local image path is missing")
        return str(image.path)

    if image.source_type == ImageSourceType.URL:
        if not image.url:
            raise VLMProviderError("Remote image URL is missing")
        return image.url

    # BYTES / DATA_URL → PIL
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise VLMProviderError(
            "Pillow is required to pass in-memory images to LocalPeftVLM"
        ) from exc

    data = image.read_bytes()
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise VLMProviderError(f"Cannot decode in-memory image: {exc}") from exc


def request_to_hf_messages(request: VLMRequest) -> list[dict[str, Any]]:
    """Map BaseVLM chat messages to Qwen2.5-VL HF chat template format."""

    messages: list[dict[str, Any]] = []
    for message in request.messages:
        role = (
            message.role.value
            if isinstance(message.role, MessageRole)
            else str(message.role)
        )
        parts: list[dict[str, Any]] = []
        for item in message.content:
            if isinstance(item, TextContent):
                parts.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                parts.append({"type": "image", "image": image_input_to_hf(item.image)})
            else:
                raise VLMProviderError(f"Unsupported content type: {type(item)!r}")

        if len(parts) == 1 and parts[0]["type"] == "text":
            messages.append({"role": role, "content": parts[0]["text"]})
        else:
            messages.append({"role": role, "content": parts})
    return messages


class LocalPeftVLM(BaseVLM):
    """Qwen2.5-VL local inference with optional PEFT/LoRA adapter."""

    def __init__(
        self,
        base_model: str,
        *,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool | None = None,
        torch_dtype: str | None = None,
        local_files_only: bool | None = None,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
        device_map: str | Mapping[str, Any] = "auto",
        trust_remote_code: bool = True,
        generation_config: VLMGenerationConfig | None = None,
        keep_history: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_4bit = (
            _env_flag("GUI_AGENT_LOAD_IN_4BIT", True)
            if load_in_4bit is None
            else bool(load_in_4bit)
        )
        # Skip Hub network: use env HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE or
        # GUI_AGENT_LOCAL_FILES_ONLY=1 after the base model is already cached.
        if local_files_only is None:
            resolved_local_only = (
                _env_flag("GUI_AGENT_LOCAL_FILES_ONLY", False)
                or _env_flag("HF_HUB_OFFLINE", False)
                or _env_flag("TRANSFORMERS_OFFLINE", False)
            )
        else:
            resolved_local_only = bool(local_files_only)
        adapter = Path(adapter_path).expanduser().resolve() if adapter_path else None
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            raise VLMConfigurationError(
                f"adapter_config.json missing under {adapter}"
            )

        super().__init__(
            model=str(base_model),
            provider="local_peft",
            generation_config=generation_config,
            keep_history=keep_history,
            metadata={
                "adapter_path": str(adapter) if adapter else None,
                "load_in_4bit": resolved_4bit,
                "local_files_only": resolved_local_only,
                **dict(metadata or {}),
            },
        )
        self.adapter_path = adapter
        self.load_in_4bit = resolved_4bit
        self.local_files_only = resolved_local_only
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        # Colab: prefer whole model on cuda:0 (bnb 4-bit forbids CPU/disk shards).
        # Set GUI_AGENT_DEVICE_MAP=auto|cuda:0|{"": 0}
        env_map = (os.environ.get("GUI_AGENT_DEVICE_MAP") or "").strip()
        if env_map:
            if env_map.lower() == "auto":
                self.device_map = "auto"
            elif env_map.startswith("{"):
                import ast

                self.device_map = ast.literal_eval(env_map)
            else:
                # "cuda:0" / "0" → force single device
                self.device_map = {"": env_map}
        else:
            self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._torch_dtype_name = torch_dtype
        self._torch: Any = None
        self.model_obj: Any = None
        self.processor: Any = None
        self._load_model()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise VLMConfigurationError(
                "LocalPeftVLM requires torch, transformers, peft "
                "(and bitsandbytes when load_in_4bit=True). "
                f"Import error: {exc}"
            ) from exc

        try:
            from qwen_vl_utils import process_vision_info  # noqa: F401
        except ImportError as exc:
            raise VLMConfigurationError(
                "Please install qwen-vl-utils: pip install qwen-vl-utils"
            ) from exc

        self._torch = torch
        dtype = _resolve_torch_dtype(torch, self._torch_dtype_name)

        processor_src = str(self.model)
        if self.adapter_path is not None and (
            (self.adapter_path / "preprocessor_config.json").is_file()
            or (self.adapter_path / "processor_config.json").is_file()
        ):
            processor_src = str(self.adapter_path)

        self.processor = AutoProcessor.from_pretrained(
            processor_src,
            trust_remote_code=self.trust_remote_code,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            local_files_only=self.local_files_only,
        )

        quant_config = None
        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }

        if self.load_in_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quant_config
            except Exception as exc:
                LOGGER.warning(
                    "bitsandbytes 4-bit unavailable (%s); falling back to %s",
                    exc,
                    self._torch_dtype_name or "bfloat16",
                )
                self.load_in_4bit = False
                model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype

        # Free stale CUDA allocations from a previous notebook cell / failed load.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass

        LOGGER.info(
            "Loading local VLM base=%s adapter=%s load_in_4bit=%s",
            self.model,
            self.adapter_path,
            self.load_in_4bit,
        )
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model,
                **model_kwargs,
            )
        except Exception as exc:
            msg = str(exc).lower()
            oomish = any(
                token in msg
                for token in (
                    "disk",
                    "cpu or the disk",
                    "out of memory",
                    "cuda out of memory",
                    "gpu ram",
                    "offload",
                )
            )
            # Falling back to bf16/fp16 after a 4-bit OOM makes Colab worse
            # (full 7B + disk/meta offload → broken LoRA attach).
            allow_fallback = _env_flag("GUI_AGENT_ALLOW_4BIT_FALLBACK", False)
            if quant_config is not None and allow_fallback and not oomish:
                LOGGER.warning(
                    "4-bit load failed (%s); retrying without quantization",
                    exc,
                )
                self.load_in_4bit = False
                model_kwargs.pop("quantization_config", None)
                model_kwargs["torch_dtype"] = dtype
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model,
                    **model_kwargs,
                )
            else:
                hint = ""
                if oomish:
                    hint = (
                        " GPU RAM insufficient for quantized load. "
                        "In Colab: Runtime → Disconnect and delete runtime, "
                        "then reconnect (T4/L4), clear GPU, and reload in 4-bit only "
                        "(do not fall back to full precision)."
                    )
                raise VLMProviderError(
                    f"Failed to load base model {self.model!r}: {exc}.{hint}"
                ) from exc

        if self.adapter_path is not None:
            try:
                print(
                    f"Attaching LoRA from {self.adapter_path} (1–3 min on T4)...",
                    flush=True,
                )
                model = PeftModel.from_pretrained(model, str(self.adapter_path))
                print("LoRA attach complete", flush=True)
            except Exception as exc:
                raise VLMProviderError(
                    f"Failed to load LoRA adapter from {self.adapter_path}: {exc}"
                ) from exc

        model.eval()
        self.model_obj = model
        self.metadata["adapter_loaded"] = self.adapter_path is not None
        self.metadata["load_in_4bit"] = self.load_in_4bit
        print(f"LocalPeftVLM ready load_in_4bit={self.load_in_4bit}", flush=True)

    @property
    def adapter_loaded(self) -> bool:
        return self.adapter_path is not None and self.model_obj is not None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _model_device(self) -> Any:
        params = getattr(self.model_obj, "parameters", None)
        if callable(params):
            first = next(params(), None)
            if first is not None:
                return first.device
        return getattr(self.model_obj, "device", "cpu")

    def _generate_once(self, request: VLMRequest) -> VLMResponse:
        if self.model_obj is None or self.processor is None or self._torch is None:
            raise VLMProviderError("LocalPeftVLM model is not loaded")

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover
            raise VLMProviderError("qwen_vl-utils is required at inference time") from exc

        started = time.perf_counter()
        hf_messages = request_to_hf_messages(request)
        max_new_tokens = int(request.config.max_tokens or 512)
        temperature = float(request.config.temperature)

        try:
            text = self.processor.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = process_vision_info(hf_messages)
            kwargs: dict[str, Any] = {
                "text": [text],
                "padding": True,
                "return_tensors": "pt",
            }
            if images:
                kwargs["images"] = images
            if videos:
                kwargs["videos"] = videos
            inputs = self.processor(**kwargs)
            device = self._model_device()
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
            if temperature and temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            with self._torch.inference_mode():
                out = self.model_obj.generate(**inputs, **gen_kwargs)
            trimmed = out[:, inputs["input_ids"].shape[1] :]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True
            )[0]
        except Exception as exc:
            raise VLMProviderError(f"LocalPeftVLM generation failed: {exc}") from exc

        usage = VLMUsage(
            input_tokens=int(inputs["input_ids"].shape[1])
            if "input_ids" in inputs
            else None,
            output_tokens=int(trimmed.shape[1]) if trimmed is not None else None,
        )
        usage.normalise_total()
        return self.make_response(
            request=request,
            text=decoded,
            usage=usage,
            finish_reason="stop",
            latency_seconds=time.perf_counter() - started,
            metadata={
                "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                "load_in_4bit": self.load_in_4bit,
            },
        )


__all__ = [
    "LocalPeftVLM",
    "image_input_to_hf",
    "request_to_hf_messages",
]
