import torch
import src.trainer
from src.cli import main

def spatial_ablation_builder(model, cfg):
    # 1. Let the codebase build the native optimizers using its own registry logic
    # This prevents any ModuleNotFound or config initialization crashes!
    dynmuon, adamw = src.trainer.build_optimizers_original(model, cfg)
    
    print("\n🚀 [SPATIAL ABLATION] Intercepting Built Optimizers!")
    
    if dynmuon is None:
        raise ValueError("Spatial ablation expects a matrix optimizer (like relmuon or muon) to be active in the config!")

    muon_params = []
    adamw_params = []
    
    # 2. Re-route the parameters based on names
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.ndim >= 2 and 'attn' in name:
            muon_params.append(param)
            print(f" Routed Attention to Matrix Optimizer: {name}")
        else:
            adamw_params.append(param)
            print(f" Routed MLP/Bias/Norm to AdamW:        {name}")

    # 3. Clear whatever parameters your team's script assigned and overwrite them
    dynmuon.param_groups = [{"params": muon_params, "lr": cfg.get("muon_lr", 0.02)}]
    
    if adamw is not None:
        adamw.param_groups = [{"params": adamw_params, "lr": cfg.get("adam_lr", 6e-4), "weight_decay": cfg.get("weight_decay", 0.1)}]
    else:
        # If the config didn't build an AdamW optimizer natively, spin one up for the MLP parameters
        print("Creating fallback AdamW instance for routed layers...")
        adamw = torch.optim.AdamW(
            adamw_params, 
            lr=cfg.get("adam_lr", 6e-4), 
            weight_decay=cfg.get("weight_decay", 0.1),
            betas=(0.9, 0.95)
        )
        
    return dynmuon, adamw

if __name__ == "__main__":
    # Save a reference to the original factory function so we can use it inside our interceptor
    src.trainer.build_optimizers_original = src.trainer.build_optimizers
    
    # Inject our monkey-patch
    src.trainer.build_optimizers = spatial_ablation_builder
    
    # Hand control back over to the CLI
    main()