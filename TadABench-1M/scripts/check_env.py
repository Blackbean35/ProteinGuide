import os
import sys

print("=== Environment Check ===")
print(f"python: {sys.version.split()[0]}")

try:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
except Exception as e:
    print(f"[warn] torch import failed: {e}")

for pkg in ["datasets", "transformers", "esm"]:
    try:
        __import__(pkg)
        print(f"{pkg}: OK")
    except Exception as e:
        print(f"{pkg}: missing or import failed ({e})")

# Offline dataset presence check (AA split required by default configs)
required = [
    os.path.join("data", f"all.AA.{sp}") for sp in ("train", "val", "test")
]
missing = [p for p in required if not os.path.isdir(p)]
if missing:
    print("[warn] missing required local dataset dirs:")
    for p in missing:
        print(" -", p)
    print("Place the offline dataset under data/ before running.")
else:
    print("local dataset: OK (data/all.AA.{train,val,test} present)")

print("=== Check Complete ===")
