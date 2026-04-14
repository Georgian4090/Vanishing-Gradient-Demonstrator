# Vanishing Gradient Demonstrator (VGD)

An educational tool to visualize and understand the vanishing gradient problem in deep neural networks.

## 🚀 Features
- **Probe Network**: MLP with built-in hooks to capture gradients, activations, and weight updates.
- **Custom Activations**: Define your own activation functions as strings using SymPy syntax.
- **Deep Network Analysis**: Compare Sigmoid, ReLU, and Tanh across many layers to see where learning stalls.
- **Automatic Visualization**: Generates loss curves, gradient flow charts, and activation histograms.

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🛠️ Usage
Run the demo to compare deep Sigmoid vs ReLU networks:
```bash
python main.py
```

### Example: Custom Activation
You can use the experimental API to test your own functions:
```python
from vgd import ExperimentConfig, Experiment

config = ExperimentConfig(
    activation="Custom",
    custom_expr="x / (1 + x**2)**0.5",
    n_layers=6,
    label="custom_run"
)
Experiment(config).run()
```

## 📂 Project Structure
- `vgd/`: Core library package.
  - `activations.py`: Symbolic and built-in activations.
  - `model.py`: MLP with diagnostic hooks.
  - `trainer.py`: Training orchestration.
  - `visualizer.py`: Plotting engine.
  - `experiment.py`: Orchestrator.
- `main.py`: Entry point for demo.
- `results/`: Directory where plots and metrics are saved.

## 📊 Sample Output
The tool generates plots like `gradient_flow.png` which shows the exponential decay of gradients in Sigmoid networks compared to the healthy flow in ReLU networks.
