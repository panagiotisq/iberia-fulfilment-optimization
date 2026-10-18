"""Single entry point to the simulated world shared by all experiments."""
from functools import lru_cache

from . import config
from .demand import simulate
from .network import iberia


@lru_cache(maxsize=None)
def world():
    net = iberia()
    hist = simulate(net, config.HISTORY_START, config.HISTORY_END, seed=config.DGP_SEED)
    return net, hist
