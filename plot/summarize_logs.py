import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    steps = []
    losses = []
    
    # Matches lines like: step  2001/20000 | loss 6.3192
    log_pattern = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss\s+([\d.]+)")
    
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                
    if not steps:
        return None
    return steps, losses

# Define your directory and file dictionary mapping
results_dir = "results"
experiments = {
    "Baseline (AdamW/Muon)": "baseline_dryrun_log.txt",
    "Standard RelMuon": "single_results_1.txt",
    "Dynamic Depth Routing": "log_route_finished.txt",
    "RelMuon Log1p Marathon": "log_log1p_marathon.txt",
    "Low LR Rescue (0.005)": "log_relmuon_low_lr.txt",
    "Stabilized Spatial (Log1p)": "spatial_buggy_dryrun_log.txt"
}

# Containers for metrics
summary_stats = {}

plt.figure(figsize=(12, 7))

print("=" * 60)
print("📌 PROCESSING EXPERIMENT LOG DATA")
print("=" * 60)

for name, filename in experiments.items():
    full_path = os.path.join(results_dir, filename)
    data = parse_log_file(full_path)
    
    if data is None:
        print(f"⚠️ Skipping {name}: File not found or empty ({full_path})")
        continue
        
    steps, losses = data
    
    # Generate stats
    start_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    min_step = steps[losses.index(min_loss)]
    
    # Variance in final stages to capture divergence/instability
    tail_variance = np.var(losses[-50:]) if len(losses) >= 50 else np.var(losses)
    
    print(f"\n✅ Parsed: {name}")
    print(f"  • Steps:      {steps[-1]}")
    print(f"  • Min Loss:   {min_loss} at step {min_step}")
    print(f"  • Final Loss: {final_loss}")
    print(f"  • Stability:  {tail_variance:.6f} (tail variance)")
    
    # Add to plot (using alpha and distinct line styles to prevent overcrowding)
    linestyle = "--" if "Ablation" in name or "Sprint" in name else "-"
    plt.plot(steps, losses, label=f"{name} (Final: {final_loss:.3f})", linestyle=linestyle, alpha=0.85)

# Plot formatting
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Comprehensive Optimizer Evaluation & Ablation Analysis")
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc="upper right")

# Save results
output_image = "comprehensive_optimizer_comparison.png"
plt.savefig(output_image, dpi=300, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print(f"📊 Graphical analysis exported successfully to: {output_image}")
print("=" * 60)