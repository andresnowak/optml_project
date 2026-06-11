import os
import re
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    steps, losses = [], []
    log_pattern = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss\s+([\d.]+)")
    if not os.path.exists(file_path): return None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return (steps, losses) if steps else None

results_dir = "results"
experiments = {
    "Baseline (AdamW/Muon)": "baseline_results.txt",
    "Standard RelMuon": "single_results_1.txt",
    "Dynamic Depth Routing": "log_route_finished.txt",
    "RelMuon Log1p Marathon": "log_log1p_marathon.txt",
    "Low LR Rescue (0.005)": "log_relmuon_low_lr.txt",
    "Stabilized Spatial (Log1p)": "log_spatial_log1p.txt"
}

# Colors and line styles to keep the lines visually distinct
styles = {
    "Baseline (AdamW/Muon)": {"color": "#1f77b4", "linestyle": "-"},
    "Standard RelMuon": {"color": "#ff7f0e", "linestyle": "-"},
    "Dynamic Depth Routing": {"color": "#2ca02c", "linestyle": "-"},
    "RelMuon Log1p Marathon": {"color": "#d62728", "linestyle": "-"},
    "Low LR Rescue (0.005)": {"color": "#9467bd", "linestyle": "--"},
    "Stabilized Spatial (Log1p)": {"color": "#8c564b", "linestyle": "--"}
}

# Create a two-column figure (Left: Log Scale Overview, Right: Late-Stage Zoom)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, filename in experiments.items():
    data = parse_log_file(os.path.join(results_dir, filename))
    if not data: continue
    steps, losses = data
    
    # Left Plot: Full curve with log scale on Y
    ax1.plot(steps, losses, label=f"{name} ({losses[-1]:.3f})", **styles[name], alpha=0.9)
    
    # Right Plot: Zoomed look at steps 2,000 to 20,000, and loss 3.0 to 8.0
    ax2.plot(steps, losses, **styles[name], alpha=0.9)

# Format Left Plot (Log Scale overview)
ax1.set_yscale("log")
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Loss (Log Scale)")
ax1.set_title("Full Training History (Logarithmic Y-Axis)")
ax1.grid(True, which="both", linestyle=":", alpha=0.5)
ax1.legend(loc="upper right", fontsize=9)

# Format Right Plot (Late-Stage detailed focus)
ax2.set_xlim(1000, 20000)
ax2.set_ylim(3.0, 8.0)
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Loss")
ax2.set_title("Convergence Details (Steps 1k-20k, Loss 3-8)")
ax2.grid(True, linestyle=":", alpha=0.5)

plt.suptitle("Comprehensive Optimizer Evaluation & Ablation Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()

output_image = "professional_optimizer_comparison.png"
plt.savefig(output_image, dpi=300, bbox_inches="tight")
print(f"✨ Professional dual-view graph exported to: {output_image}")