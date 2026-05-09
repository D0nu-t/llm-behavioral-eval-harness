def extract_final_token_states(hidden_states):
    return [layer[0, -1, :].detach().cpu() for layer in hidden_states]


def extract_mean_pooled_states(hidden_states):
    return [layer[0].mean(dim=0).detach().cpu() for layer in hidden_states]
