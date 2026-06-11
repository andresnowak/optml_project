import torch
import src.trainer
from src.cli import main

def spatial_ablation_builder(model, cfg):
    # 1. Let the codebase build the native optimizers using its own registry logic
    # This prevents any ModuleNotFound or config initialization crashes!
    dynmuon, adamw = src.trainer.build_optimizers_original(model, cfg)
    
    print("\n[SPATIAL ABLATION] Intercepting Built Optimizers!")
    
    if dynmuon is None:
        raise ValueError("Spatial ablation expects a matrix optimizer (like relmuon or muon) to be active in the config!")

    muon_params = []
    
    # 2. Re-route the parameters based on names
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.ndim >= 2 and 'attn' in name:
            muon_params.append(param)
            print(f" Routed Attention to Matrix Optimizer: {name}")
        else:
            print(f" Routed MLP/Bias/Norm to AdamW:        {name}")

    # 3. Preserve the original optimizer metadata and only filter the parameter lists.
    # Filter the parameters in each of its param groups to only keep those in muon_params.
    # This keeps group-specific hyperparameters/states intact while routing only attn params to it.
    muon_param_ids = {id(p) for p in muon_params}
    for group in dynmuon.param_groups:
        group["params"] = [p for p in group["params"] if id(p) in muon_param_ids]
    
    # 4. Reconstruct the AdamW optimizer to optimize MLP, embedding, and scalar parameters without duplicates.
    # Separate the non-attention parameters into the correct sub-groups for AdamW:
    embed_params = []
    mlp_params = []
    scalar_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and 'attn' in name:
            continue
            
        if name == "embed.weight":
            embed_params.append(param)
        elif param.ndim >= 2:
            mlp_params.append(param)
        else:
            scalar_params.append(param)
            
    adamw_groups = []
    if embed_params:
        embed_lr = cfg.get("embed_lr", cfg.get("adam_lr", 6e-4))
        adamw_groups.append({
            "params": embed_params,
            "name": "embed",
            "lr": embed_lr,
            "initial_lr": embed_lr,
            "weight_decay": cfg.get("scalar_weight_decay", 0.0),
        })
    if mlp_params:
        adam_lr = cfg.get("adam_lr", 6e-4)
        adamw_groups.append({
            "params": mlp_params,
            "name": "mlp",
            "lr": adam_lr,
            "initial_lr": adam_lr,
            "weight_decay": cfg.get("weight_decay", 0.1),
        })
    if scalar_params:
        adam_lr = cfg.get("adam_lr", 6e-4)
        adamw_groups.append({
            "params": scalar_params,
            "name": "aux",
            "lr": adam_lr,
            "initial_lr": adam_lr,
            "weight_decay": cfg.get("scalar_weight_decay", 0.0),
        })
        
    adamw = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95))
    
    return dynmuon, adamw


if __name__ == "__main__":
    # Save a reference to the original factory function so we can use it inside our interceptor
    src.trainer.build_optimizers_original = src.trainer.build_optimizers
    
    # Inject our monkey-patch
    src.trainer.build_optimizers = spatial_ablation_builder
    
    # Hand control back over to the CLI
    main()