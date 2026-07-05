"""Memory leak guard for batch dataloader."""
import gc
def clear_cache():
    gc.collect()
