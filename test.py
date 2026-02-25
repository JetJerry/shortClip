
import torch, importlib
print("torch:", torch.__version__, "cuda:", getattr(torch.version,'cuda',None), "cuda_available:", torch.cuda.is_available())
try:
    import torch.utils._pytree as pt
    print("pytree.register_pytree_node:", getattr(pt,"register_pytree_node",None) is not None)
except Exception as e:
    print("pytree import error:", e)
try:
    import transformers
    from transformers import CLIPProcessor, CLIPModel
    print("transformers:", transformers.__version__, "CLIPProcessor OK")
except Exception as e:
    print("transformers import error:", e)