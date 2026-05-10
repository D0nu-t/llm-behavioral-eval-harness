"""
backends/hf_transformer.py

HuggingFace local backend with activation capture via forward hooks.


BUG THAT WAS HERE (2): the backend ran self.model() AND self.model.generate()
as two separate calls — two full forward passes. Hidden states were captured
from the prompt-only forward pass, then discarded. generate() ran without
capturing anything useful.

FIX: single model.generate() call with return_dict_in_generate=True and
output_hidden_states=True. Extract prefix hidden states from the first
generation step (outputs.hidden_states[0]) — shape (num_layers+1, batch,
seq_len, hidden_dim). This gives us activations over the full prompt context.

FIX (hook save): save_activations() is now called once here after generate()
returns, not inside each hook. See live_capture.py.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness.backends.base import ModelBackend
from eval_harness.interpretability.live_capture import make_hook, save_activations
from eval_harness.schemas.responses import ModelResponse


def _register_hooks(model) -> int:
    """
    Register hooks on the correct layer list for the model family.
    Supports Qwen2.5 / LLaMA-style (model.model.layers) and
    falls back to GPT-2-style (model.transformer.h) for legacy models.
    Returns the number of hooks registered.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers          # Qwen2.5, LLaMA, Mistral, Gemma
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h         # GPT-2, older Qwen
    else:
        raise RuntimeError(
            f"Cannot locate transformer layers on {type(model).__name__}. "
            "Add the attribute path to _register_hooks()."
        )

    for i, block in enumerate(layers):
        block.register_forward_hook(make_hook(f"layer_{i}"))

    return len(layers)


class HFTransformerBackend(ModelBackend):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name

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
    # GPT-2 and other base models have no chat template.
    # Concatenate turns manually.
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

        # Hooks fire during generate(). Persist to disk once, atomically.
        save_activations()

        gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # outputs.hidden_states: tuple[step] of tuple[layer] of [batch, seq, dim]
        # Take the first generation step's layers for the prompt representation.
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
