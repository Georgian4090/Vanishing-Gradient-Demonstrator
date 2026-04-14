import torch
import numpy as np
from sklearn.datasets import make_classification
from typing import Tuple, Optional

def get_xor_dataset(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates the XOR problem dataset."""
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)

def get_synthetic_dataset(n_samples: int = 1000, n_features: int = 10, n_classes: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a synthetic classification dataset using sklearn."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=min(n_features, 5), 
        n_redundant=0, 
        n_classes=n_classes, 
        random_state=42
    )
    if n_classes == 2:
        return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()

def get_dataset(name: str, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factory function for datasets."""
    name_clean = name.lower().replace(" ", "").replace("_", "")
    
    if name_clean == "xor":
        return get_xor_dataset(n_samples=kwargs.get("n_samples", 1000))
    elif name_clean == "synthetic":
        return get_synthetic_dataset(
            n_samples=kwargs.get("n_samples", 1000),
            n_features=kwargs.get("n_features", 10),
            n_classes=kwargs.get("n_classes", 2)
        )
    else:
        raise ValueError(f"Dataset '{name}' not supported. Choose from: XOR, Synthetic.")

if __name__ == "__main__":
    X, y = get_dataset("xor")
    print(f"XOR shape: {X.shape}, labels: {y.shape}")
    X, y = get_dataset("synthetic", n_features=5)
    print(f"Synthetic shape: {X.shape}, labels: {y.shape}")
