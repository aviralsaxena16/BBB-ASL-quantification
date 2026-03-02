import os


def load_stan_model(stan_filename):
    """
    Load Stan model code from file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stan_path = os.path.join(base_dir, "models", "stan", stan_filename)

    if not os.path.exists(stan_path):
        raise FileNotFoundError(f"Stan model file not found: {stan_path}")

    with open(stan_path, "r") as f:
        return f.read()