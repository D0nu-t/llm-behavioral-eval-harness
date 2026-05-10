import json
import torch

from eval_harness.dashboard.state import (
    LIVE_ACTIVATIONS,
)


def save_activations():
    serializable = {}

    for k, tensor in LIVE_ACTIVATIONS.items():
        serializable[k] = tensor.tolist()

    with open(
        "activation_dump.json",
        "w",
    ) as f:
        json.dump(
            serializable,
            f,
        )

        f.flush()
        print("Written ", f, " to file")


def make_hook(name):
    def hook(
        module,
        inputs,
        outputs,
    ):
        print(f"HOOK FIRING: {name}")

        if isinstance(
            outputs,
            tuple,
        ):
            tensor = outputs[0]

        else:
            tensor = outputs

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            print(f"NOT A TENSOR: {type(tensor)}")

            return

        tensor = tensor[:, -1, :].detach().cpu()

        print(f"{name} -> {tensor.shape}")

        LIVE_ACTIVATIONS[name] = tensor

        save_activations()
        print(f"Activations saved for {name}")

    return hook
