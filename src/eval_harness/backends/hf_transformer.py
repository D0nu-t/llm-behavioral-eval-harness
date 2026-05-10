import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness.backends.base import ModelBackend
from eval_harness.interpretability.live_capture import make_hook, save_activations
from eval_harness.schemas.responses import ModelResponse


def _register_hooks(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers          # Qwen2.5, LLaMA, Mistral, Gemma
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h         # GPT-2
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

    def __init__(self, model_name: str, device: str = "cpu"):
        self._model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        self.model.eval()

        n = _register_hooks(self.model)
        print(f"[HFTransformerBackend] loaded {model_name}, registered hooks on {n} layers")

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

        hidden_states = [
            layer_hs[0, -1, :].detach().cpu()
            for layer_hs in outputs.hidden_states[0]
        ]

        return ModelResponse(
            text=text,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_ids),
            latency_ms=latency_ms,
            hidden_states=hidden_states,
            raw={},
        )
