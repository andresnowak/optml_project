import re
import matplotlib.pyplot as plt
import os

def parse_log_file(file_path):
    steps = []
    losses = []
    
    # Regex to find lines like: "step  2001/20000 | loss 6.3192 | lr 5.87e-04"
    log_pattern = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss\s+([\d.]+)")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find {file_path}")
                
    return steps, losses

# 1. Update the paths to point to your new downloaded files
base_steps, base_losses = parse_log_file("results/baseline_dryrun_log.txt")
spatial_steps, spatial_losses = parse_log_file("results/spatial_buggy_dryrun_log.txt")

# Create the plots
plt.figure(figsize=(10, 6))

# 2. Update the plotting logic and labels
if base_steps:
    plt.plot(base_steps, base_losses, label="Baseline (Log1p Full Network)", color="blue", alpha=0.8)

if spatial_steps:
    # Explicitly labeled as "Buggy" so it doesn't end up in the final paper!
    plt.plot(spatial_steps, spatial_losses, label="Spatial Ablation (Dry Run - Buggy LR)", linestyle="--", color="red")

plt.xlabel("Training Steps")
plt.ylabel("Training Loss")
plt.title("Optimizer Convergence Comparison (Dry Run Evaluation)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# 3. Create the img directory if it doesn't exist to prevent crashes
os.makedirs("img", exist_ok=True)

# Save the plot image
plt.savefig("img/dryrun_comparison.png", dpi=300)
plt.show()

print("📊 Graph generated successfully as img/dryrun_comparison.png!")