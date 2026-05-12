"""
interpretability/nla_client.py

NLAVerbalizer: converts a residual-stream activation vector into a
natural-language description using a released AV (Activation Verbalizer)
checkpoint from the kitft/nla-models collection.

Design constraints
------------------
- Pure transformers inference — no SGLang server required.
  We replicate the input_embeds injection path manually:
  tokenize the prompt, get embeddings, overwrite the injection slot,
  call model.generate(inputs_embeds=...).
- Sequential loading. The verbalizer loads the AV model only when
  explain() is first called, and can be explicitly unloaded via unload()
  to free VRAM before reloading the subject model.
- Optional. The orchestrator checks `NLA_MODEL` env var; if unset,
  nla_explanation is None in every SSE event and no AV model is loaded.

Supported checkpoints (all from kitft/nla-models collection)
-------------------------------------------------------------
  kitft/nla-qwen2.5-7b-L20-av   — pair with Qwen2.5-7B-Instruct layer 20
  kitft/nla-gemma3-12b-L32-av   — pair with Gemma-3-12B-IT layer 32
  kitft/nla-gemma3-27b-L41-av   — pair with Gemma-3-27B-IT layer 41
  kitft/Llama-3.3-70B-NLA-L53-av — pair with Llama-3.3-70B-Instruct layer 53

Usage
-----
    verbalizer = NLAVerbalizer("kitft/nla-qwen2.5-7b-L20-av")
    explanation = verbalizer.explain(activation_tensor)   # torch.Tensor [d]
    verbalizer.unload()   # free VRAM before reloading subject model
"""

import math
import os
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_QUANT_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


class NLAVerbalizer:
    """
    Loads an NLA AV checkpoint and verbalizes activation vectors.

    Parameters
    ----------
    av_model_id : str
        HuggingFace repo ID for the AV checkpoint.
    device : str
        Target device. Use "cuda" for GPU inference.
    quantize : bool
        Load in 4-bit (NF4) to save VRAM. Requires bitsandbytes.
    """

    def __init__(
        self,
        av_model_id: str,
        device: str = "cuda",
        quantize: bool = True,
    ):
        self.av_model_id = av_model_id
        self.device = device
        self.quantize = quantize
        self._model = None
        self._tokenizer = None
        self._meta = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return

        print(f"[NLAVerbalizer] loading {self.av_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.av_model_id)

        load_kwargs = dict(
            device_map="auto" if self.device == "cuda" else self.device,
            torch_dtype=torch.float16,
        )
        if self.quantize and self.device == "cuda":
            load_kwargs["quantization_config"] = _QUANT_4BIT

        self._model = AutoModelForCausalLM.from_pretrained(
            self.av_model_id, **load_kwargs
        )
        self._model.eval()

        # Load sidecar — nla_meta.yaml ships with every checkpoint
        sidecar = self._resolve_sidecar()
        with open(sidecar) as f:
            self._meta = yaml.safe_load(f)

        print(f"[NLAVerbalizer] ready — d_model={self._meta['d_model']}, "
              f"injection_scale={self._meta['extraction']['injection_scale']}")

    def _resolve_sidecar(self) -> Path:
        """
        Look for nla_meta.yaml in the HF cache for this model.
        Falls back to downloading via huggingface_hub if not cached.
        """
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.av_model_id, "nla_meta.yaml")
            return Path(path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load nla_meta.yaml for {self.av_model_id}: {e}"
            )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def explain(self, activation: torch.Tensor, max_new_tokens: int = 200) -> str:
        """
        Verbalize a residual-stream activation vector.

        Parameters
        ----------
        activation : torch.Tensor
            Shape [d_model]. Raw hidden state — not pre-normalized.
            Must come from the layer and model family the AV was trained on.
        max_new_tokens : int
            Generation budget.

        Returns
        -------
        str
            The AV's natural-language description, stripped of XML tags.
            Returns a fallback string on extraction failure.
        """
        self._ensure_loaded()
        meta = self._meta

        d_model         = meta["d_model"]
        injection_scale = meta["extraction"]["injection_scale"]
        inj_token_id    = meta["tokens"]["injection_token_id"]
        left_id         = meta["tokens"]["injection_left_neighbor_id"]
        right_id        = meta["tokens"]["injection_right_neighbor_id"]
        inj_char        = meta["tokens"]["injection_char"]
        template        = meta["prompt_templates"]["av"]

        # 1. Build prompt from sidecar template
        content = template.format(injection_char=inj_char)
        messages = [{"role": "user", "content": content}]

        # 2. Tokenize — one-step to avoid double-BOS on Gemma/Llama
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )  # [1, T]

        # 3. Embed
        embed_fn = self._model.get_input_embeddings()
        embeds = embed_fn(input_ids.to(self._model.device)).float()  # [1, T, d]

        # Gemma-3 multiplies embeddings by √d in its forward — replicate here
        model_type = self._model.config.model_type
        if "gemma" in model_type.lower():
            embeds = embeds * math.sqrt(d_model)

        # 4. Scale activation vector to injection_scale L2-norm
        v = activation.float()
        norm = v.norm()
        if norm < 1e-8:
            return "[NLA: zero-norm activation — skipped]"
        v_scaled = v * (injection_scale / norm)

        # 5. Find injection position (token ID + neighbor verification)
        ids_list = input_ids[0].tolist()
        inj_pos = None
        for i in range(1, len(ids_list) - 1):
            if (ids_list[i] == inj_token_id
                    and ids_list[i - 1] == left_id
                    and ids_list[i + 1] == right_id):
                inj_pos = i
                break

        if inj_pos is None:
            return "[NLA: injection position not found — template drift?]"

        # 6. Overwrite injection slot
        embeds[0, inj_pos] = v_scaled.to(embeds.device)

        # 7. Generate
        with torch.no_grad():
            out_ids = self._model.generate(
                inputs_embeds=embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        full_text = self._tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # 8. Extract <explanation>...</explanation>
        m = re.search(r"<explanation>(.*?)</explanation>", full_text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Fallback — return raw output truncated (injection may have worked partially)
        return full_text.strip()[:400]

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def unload(self):
        """Delete the AV model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[NLAVerbalizer] unloaded")
