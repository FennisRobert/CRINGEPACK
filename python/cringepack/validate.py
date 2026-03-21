import numpy as np


def val_result(x1: np.ndarray, x2: np.ndarray):
    err = np.linalg.norm(x2-x1)
    if err < 1e-12:
        print("✅ Correct!")
    else:
        print(f"❌ Wrong! error = {err}")