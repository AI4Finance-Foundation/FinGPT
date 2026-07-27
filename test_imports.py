#!/usr/bin/env python
import sys
print("Python version:", sys.version)

try:
    import numpy
    print("NumPy:", numpy.__version__)
except Exception as e:
    print("NumPy import failed:", e)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch import failed:", e)

try:
    import transformers
    print("Transformers:", transformers.__version__)
except Exception as e:
    print("Transformers import failed:", e)

try:
    import peft
    print("PEFT:", peft.__version__)
except Exception as e:
    print("PEFT import failed:", e)

try:
    import datasets
    print("Datasets:", datasets.__version__)
except Exception as e:
    print("Datasets import failed:", e)

try:
    import bitsandbytes
    print("Bitsandbytes imported successfully")
except Exception as e:
    print("Bitsandbytes import failed:", e)

print("All imports completed!")