# SAV isolation — linear_regression

lr = 1.00e-01, steps = 300, seed = 0

| variant | score (min over last 10%) | final loss |
|---|---|---|
| top_k=0 (no-SAV / paper tail) | 1.333e-03 | 1.354e-03 |
| top_k=1 | 3.966e-03 | 3.966e-03 |
| top_k=6 (paper default) | 6.777e-03 | 6.777e-03 |
| top_k=32 | 2.800e-02 | 2.800e-02 |
| gated (τ=0.05, k=6) | 1.515e-03 | 1.515e-03 |
