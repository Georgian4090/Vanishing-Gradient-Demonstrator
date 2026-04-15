# Vanishing Gradient Demonstrator (VGD)
##  Overview

A **modular, research-oriented educational library** for diagnosing and understanding the **vanishing gradient problem** in deep neural networks.
Built with a strong emphasis on **transparency and experimentation**, VGD provides a controlled environment to observe how gradients evolve across layers under different architectural and functional choices.
The vanishing gradient problem arises when gradients shrink exponentially as they propagate backward through deep networks. This leads to:

* Minimal weight updates in early layers
* Slow or stalled training
* Poor convergence in deep architectures

While widely discussed in theory, it is rarely *observed directly*.

**VGD addresses this gap** by exposing the internal dynamics of neural networks through structured experiments and precise instrumentation.

##  Core Features

###  Probe Network with Gradient Instrumentation

A custom-built feedforward network that automatically logs:

* Gradient L2 norms per layer (per epoch)
* Weight update magnitudes
* Activation distributions

---

###  Activation Function Engine

Supports both standard and custom activations:

**Built-in:**

* Sigmoid
* Tanh
* ReLU
* Leaky ReLU
* ELU
* Swish

**Custom (via SymPy):**

```python
x / (1 + abs(x))
```

* Symbolic differentiation handled automatically
* Seamless integration into PyTorch pipeline

---

### Structured Metric Collection

During training, the system records:

* Loss trajectory
* Gradient norms across layers
* Activation distributions
* Weight update magnitudes

All outputs are cleanly packaged into a `TrainingResult` object.

---

### Visualization Engine

Automatically generates:

* Gradient flow plots (layer vs magnitude)
* Loss curves (epoch vs loss)
* Activation histograms

Saved to `results/` for reproducibility and analysis.

---

###  Experiment Comparison

Run multiple configurations and directly compare:

* Gradient decay patterns
* Convergence behavior
* Stability across architectures

---

##  Architecture

The project follows a **layered, modular design** to ensure clarity and extensibility:

```text
vanishing_gradients/
│
├── core/                   # Fundamental ML components
│   ├── activations.py
│   ├── datasets.py
│   ├── model.py
│   └── trainer.py
│
├── experiment/             # Orchestration layer
│   ├── config.py
│   └── experiment.py
│
├── analysis/               # Post-training analysis
│   ├── visualizer.py
│   └── metrics.py
│
├── utils/                  # Shared utilities
│   ├── logging.py
│   └── seed.py
│
├── results/                # Generated outputs
└── main.py                 # Example usage
```

---

## Installation

### Requirements

* Python 3.8+
* PyTorch
* NumPy
* scikit-learn
* SymPy
* Matplotlib / Plotly

### Setup

```bash
git clone <repo-url>
cd vanishing_gradients
pip install -r requirements.txt
```

---

##  Usage

### Run a Baseline Experiment

```python
from experiment.config import ExperimentConfig
from experiment.experiment import Experiment

config = ExperimentConfig(
    activation="Sigmoid",
    loss="cross_entropy",
    n_layers=6,
    hidden_dim=64,
    dataset="synthetic",
    epochs=50,
    lr=0.01,
    label="sigmoid_baseline"
)

result = Experiment(config).run()
```

---

### Compare Two Configurations

```python
from experiment.experiment import compare

compare(
    ExperimentConfig(activation="Sigmoid", ...),
    ExperimentConfig(activation="ReLU", ...)
)
```

---

### Custom Activation Example

```python
config = ExperimentConfig(
    activation="Custom",
    custom_expr="x * sigmoid(x)",  # Swish
    n_layers=10,
    label="custom_activation_run"
)
```

---

##  Output Artifacts

Each run generates a dedicated folder in `results/` containing:

* Gradient flow plots
* Loss curves
* Activation distributions
* Metrics summary (`metrics.json`)

---

##  Sample Insight

| Sigmoid Network                             | ReLU Network                  |
| ------------------------------------------- | ----------------------------- |
| Gradients decay exponentially across layers | Gradients remain stable       |
| Early layers receive near-zero updates      | Consistent updates throughout |
| Training stagnates                          | Converges efficiently         |


As example, the results for Sigmoid vanishing:

<img width="600" height="300" alt="gradient_flow" src="https://github.com/user-attachments/assets/4bf2046e-481a-4bbd-91d0-d792a296b0cb" />

<img width="500" height="250" alt="loss_curve" src="https://github.com/user-attachments/assets/3a29bd53-1bce-4195-bed1-697f9d924bce" />

<img width="500" height="250" alt="weight_updates" src="https://github.com/user-attachments/assets/973d9985-f806-4f6c-998f-cbd669942725" />


##  Future Extensions

* Per-layer activation configuration
* Optimizer comparisons
* Gradient clipping experiments
* Optional lightweight UI layer


Modify components. Break things. Run comparisons.

Understanding emerges not from reading about gradients—but from **watching them fail and fixing them**.
