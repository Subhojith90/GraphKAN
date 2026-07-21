from graphkan_amo_nqs.exceptions import DependencyUnavailable


def require_rnn_dependencies():
    raise DependencyUnavailable("Autoregressive RNN baseline is optional and not implemented in the CPU artifact.")
