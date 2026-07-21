DEFAULT_OPTIMIZER_GRID = {
    "exact_fullsum": {"method": "L-BFGS-B", "maxiter": [25, 45, 55]},
    "vmc": {"optimizers": ["Adam", "SR"], "learning_rates": [0.01, 0.003, 0.001]},
}
