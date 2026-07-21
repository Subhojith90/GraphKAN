from graphkan_amo_nqs.exceptions import DependencyUnavailable


def require_transformer_dependencies():
    raise DependencyUnavailable("Patched transformer baseline requires a separate GPU/NetKet-compatible implementation.")
