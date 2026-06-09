import torch
import src.trainer
from src.cli import main

# 1. We need to import the RelMuon class. 
# Check your src/optimizers folder if this path needs a slight adjustment!
from src.optimizers.relmuon import RelMuon 

def spatial_ablation_builder(model, cfg):
    print("\n🚀 [SPATIAL ABLATION] Intercepting Optimizers!")
    print("-> Routing Attention matrices to RelMuon.")
    print("-> Routing MLP matrices, biases, and norms to AdamW.\n")
    
    muon_params = []
    adamw_params = []
    
    # 2. Iterate through every single tensor in the neural network
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 3. The Routing Logic:
        # If it's a 2D matrix (like a linear layer) AND it belongs to the 
        # Attention mechanism ('attn' in the name), give it to RelMuon.
        if param.ndim >= 2 and 'attn' in name:
            muon_params.append(param)
            print(f"[RelMuon] {name}")
        else:
            # Give the MLP layers and 1D tensors to AdamW
            adamw_params.append(param)
            print(f"[AdamW]   {name}")
            
    print("\nInitializing parameter groups...")
    
    # 4. Build the optimizers using the learning rates from your YAML configs
    opt_muon = RelMuon(muon_params, lr=cfg.get("muon_lr", 0.02))
    opt_adamw = torch.optim.AdamW(
        adamw_params, 
        lr=cfg.get("adam_lr", 6e-4), 
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95)
    )
    
    return opt_muon, opt_adamw

if __name__ == "__main__":
    # 5. The Monkey-Patch: We overwrite the team's standard optimizer builder
    # with our custom spatial router inside the trainer module.
    src.trainer.build_optimizers = spatial_ablation_builder
    
    # 6. Call the standard CLI. It will load the config, build the model, 
    # but when it tries to build optimizers, it will trigger our code above!
    main()