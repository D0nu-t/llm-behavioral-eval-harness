import time
from dataclasses import dataclass
from typing import Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from eval_harness.backends.base import ModelBackend
from eval_harness.interpretability.live_capture import (
    make_hook,
)


@dataclass
class ModelResponse:
    text: str

    input_tokens: int
    output_tokens: int

    latency_ms: float

    hidden_states: Any | None

    raw: dict


class HFTransformerBackend(ModelBackend):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            # hf_token=os.getenv("HF_TOKEN")
        )

        self.model.eval()
        print("MODEL LOADED")
        for i, block in enumerate(self.model.transformer.h):
            block.register_forward_hook(make_hook(f"layer_{i}"))

    def complete(
        self,
        messages,
        temperature=0.0,
        max_tokens=128,
    ):
        # prompt = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        prompt = messages[-1]["content"]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.time()

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
            print("STARTING GENERATION")
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            print("GENERATION COMPLETE")

        latency_ms = (time.time() - start) * 1000

        generated_tokens = generated[0][inputs.input_ids.shape[1] :]

        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        hidden_states = outputs.hidden_states

        return ModelResponse(
            text=text,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(generated_tokens),
            latency_ms=latency_ms,
            hidden_states=hidden_states,
            raw={},
        )
