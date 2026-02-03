from .baseline import get_model as get_baseline_model
from .lsnet import get_model as get_lsnet_model

def get_model(name: str = "baseline"):
    """Factory to return a model by name.

    Args:
        name: one of 'baseline' or 'lsnet'
    """
    name = (name or "baseline").lower()
    if name in ("lsnet", "ls-net", "large-small", "large_focus_small"):
        return get_lsnet_model()
    return get_baseline_model()
