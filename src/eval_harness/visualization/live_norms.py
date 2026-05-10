import matplotlib.pyplot as plt
import torch


def plot_layer_norms(states):
    norms = [s.norm().item() for s in states]

    plt.plot(norms)
    plt.xlabel("Layer")
    plt.ylabel("Activation Norm")
    plt.title("Residual Stream Norms")
    plt.show()
