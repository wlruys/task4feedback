from omegaconf import OmegaConf

if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver(
        "mul",
        lambda a, b: int(a) * int(b),
    )
