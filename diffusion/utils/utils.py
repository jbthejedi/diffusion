import os
from omegaconf import OmegaConf

def load_config(path, env="local"):
    base_config = OmegaConf.load(path)

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        env_config = OmegaConf.load(env_path)
        # Merges env_config into base_config (env overrides base)
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config
    return config