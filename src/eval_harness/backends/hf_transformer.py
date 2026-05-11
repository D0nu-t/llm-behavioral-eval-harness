"""
backends/hf_transformer.py

HuggingFace local backend with forward-hook activation capture.

Changes from previous version
------------------------------
- 4-bit NF4 quantization via BitsAndBytesConfig (enable with quantize=True).
  Required for Qwen2.5-7B on consumer hardware (~4-5 GB VRAM).
- nla_layer parameter: the layer index whose final-token vector is stored
  separately as `response.nla_activation` for NLA verbalization.
  Default None = no NLA activation extracted.
  For Qwen2.5-7B + kitft NLA: set nla_layer=20.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval_harness.backends.base import ModelBackend
from eval_harness.interpretability.live_capture import make_hook, save_activations
from eval_harness.schemas.responses import ModelResponse


def _quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _register_hooks(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers       # Qwen2.5, LLaMA, Mistral, Gemma
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h      # GPT-2
    else:
        raise RuntimeError(
            f"Cannot locate transformer layers on {type(model).__name__}. "
            "Add the attribute path to _register_hooks()."
        )
    for i, block in enumerate(layers):
        block.register_forward_hook(make_hook(f"layer_{i}"))
    return len(layers)


class HFTransformerBackend(ModelBackend):

    @property
    def model_name(self) -> str:
        return self._model_name

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        quantize: bool = False,
        nla_layer: int | None = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID, e.g. "Qwen/Qwen2.5-7B-Instruct".
        device : str
            "cpu" or "cuda". Use "cuda" for quantized Qwen2.5-7B.
        quantize : bool
            Load in 4-bit NF4. Requires bitsandbytes. Use for 7B+ models.
        nla_layer : int | None
            If set, the final-token hidden state at this layer index is stored
            as ModelResponse.nla_activation for NLA verbalization.
            For Qwen2.5-7B + kitft NLA checkpoints: use 20.
        """
        self._model_name = model_name
        self.nla_layer = nla_layer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {
            "device_map": "auto" if device == "cuda" else device,
            "torch_dtype": torch.float32 if not quantize else torch.float16,
        }
        if quantize:
            load_kwargs["quantization_config"] = _quant_config()

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        n = _register_hooks(self.model)
        print(f"[HFTransformerBackend] loaded {model_name} "
              f"({'4-bit' if quantize else 'fp32'}), "
              f"hooks on {n} layers"
              + (f", NLA extraction at layer {nla_layer}" if nla_layer is not None else ""))

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> ModelResponse:
        if self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
        prompt += "\nASSISTANT:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        latency_ms = (time.time() - start) * 1000

        save_activations()

        gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # All layers — step 0 = prompt context representation
        hidden_states = [
            layer_hs[0, -1, :].detach().cpu()
            for layer_hs in outputs.hidden_states[0]
        ]

        # NLA extraction — single layer vector for verbalization
        nla_activation = None
        if self.nla_layer is not None and self.nla_layer < len(hidden_states):
            nla_activation = hidden_states[self.nla_layer]

        return ModelResponse(
            text=text,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_ids),
            latency_ms=latency_ms,
            hidden_states=hidden_states,
            nla_activation=nla_activation,
            raw={},
        )
