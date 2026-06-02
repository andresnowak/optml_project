# SAV isolation — shakespeare

lr = 3.00e-04, steps = 2000, seed = 0

| variant | score (min over last 10%) | final loss |
|---|---|---|
| top_k=0 (no-SAV / paper tail) | 2.432e+00 | 2.440e+00 |
| top_k=1 | 2.430e+00 | 2.438e+00 |
| top_k=6 (paper default) | 2.334e+00 | 2.339e+00 |
| top_k=32 | 1.847e+00 | 1.903e+00 |
| gated (τ=0.05, k=6) | 2.337e+00 | 2.342e+00 |
