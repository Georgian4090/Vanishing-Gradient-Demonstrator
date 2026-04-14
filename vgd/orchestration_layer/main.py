from .. import ExperimentConfig, compare, Experiment
import os
import numpy as np

def run_demo():
    """
    Demonstrates the Vanishing Gradient problem by comparing Sigmoid and ReLU
    in a deep network architecture.
    """
    print("=== Vanishing Gradient Demonstrator Demo ===")
    
    # Configuration 1: Deep Sigmoid Network (Expecting Vanishing Gradients)
    # With 10 layers, the gradients in early layers should be extremely small.
    config_sigmoid = ExperimentConfig(
        activation="Sigmoid",
        n_layers=10,
        hidden_dim=32,
        dataset="synthetic",
        epochs=150,
        lr=0.01,
        label="deep_sigmoid_vanishing"
    )
    
    # Configuration 2: Deep ReLU Network (Expecting Healthier Gradients)
    # ReLU helps mitigate the vanishing gradient problem.
    config_relu = ExperimentConfig(
        activation="ReLU",
        n_layers=10,
        hidden_dim=32,
        dataset="synthetic",
        epochs=150,
        lr=0.01,
        label="deep_relu_healthy"
    )
    
    # 3. Run Baseline Comparisons
    print("\n[Demo] Running standard baseline comparisons...")
    all_results = compare([config_sigmoid, config_relu])

    # 4. Interactive Custom Input
    print("\n" + "="*45)
    print(" INTERACTIVE MODE: Test your own activation")
    print("="*45)
    print("Examples: 'x / (1 + x**2)**0.5', 'exp(-x**2)', 'sigmoid(x) * x'")
    
    try:
        choice = input("\nWould you like to try a custom activation? (y/n): ").lower().strip()
        if choice == 'y':
            expr = input("Enter expression (variable 'x'): ").strip()
            layers_in = input("Number of hidden layers [default 6]: ").strip()
            layers = int(layers_in) if layers_in else 6
            
            custom_config = ExperimentConfig(
                activation="Custom",
                custom_expr=expr,
                n_layers=layers,
                label="user_custom_run"
            )
            
            print(f"\n[Demo] Launching custom experiment with: {expr}")
            custom_res = Experiment(custom_config).run()
            all_results.append(custom_res)
            print("\n[Success] Custom experiment complete. Check 'results/user_custom_run/'.")
    except EOFError:
        # Handle cases where input is not available (e.g. non-interactive environments)
        print("\n[Notice] Running in non-interactive mode. Skipping custom input.")
    except Exception as e:
        print(f"\n[Error] Error: {e}")

    # 5. Final Comparison Table
    if all_results:
        print("\n" + "="*65)
        print(" FINAL COMPARISON SUMMARY")
        print("="*65)
        print(f"{'Experiment Label':<25} | {'Final Loss':<10} | {'Grad Health':<12}")
        print("-" * 65)
        
        for r in all_results:
            layers = sorted(r.gradient_norms.keys())
            if layers:
                ratio = np.mean(r.gradient_norms[layers[0]]) / (np.mean(r.gradient_norms[layers[-1]]) + 1e-12)
                health = f"{ratio:.2e}"
            else:
                health = "N/A"
            
            label = r.config.get('label', 'unnamed')
            print(f"{label:<25} | {r.final_loss:<10.4f} | {health:<12}")
        print("="*65)
            
    print("\nDemo complete. Thank you for using VGD!")

if __name__ == "__main__":
    run_demo()
