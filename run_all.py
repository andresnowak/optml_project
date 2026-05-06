import subprocess
import os

# The list of all experiments defined in your project
experiments = [
    "linear_regression",
    "ill_conditioned_linear_regression",
    "matrix_factorization",
    "matrix_completion",
    "sylvester_equation",
    "orthogonal_procrustes",
    "shakespeare"
]

# Create a directory to store the generated graphs
os.makedirs("results", exist_ok=True)

# Adjust this learning rate if some experiments diverge
learning_rate = "1e-2"

for exp in experiments:
    print(f"=== Running {exp} ===")
    
    cmd = [
        "python", "main.py",
        "--experiment", exp,
        "--compare-all",
        "--lr", learning_rate,
        "--backend", "matplotlib",
        "--save-plot", f"results/{exp}_comparison.png"
    ]
    
    # We use subprocess.run to execute the command exactly as if you typed it
    subprocess.run(cmd)
    print(f"Saved graph to results/{exp}_comparison.png\n")

print("All experiments completed.")