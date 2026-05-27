# SAV isolation — matrix_factorization

lr = 1.00e-01, steps = 30, seed = 0

| variant | score (min over last 10%) | final loss |
|---|---|---|
| top_k=0 (no-SAV / paper tail) | 2.574e+00 | 2.574e+00 |
| top_k=1 | 2.782e+00 | 2.782e+00 |
| top_k=6 (paper default) | 7.042e-01 | 7.042e-01 |
| top_k=32 | 1.514e+00 | 1.514e+00 |
| gated (τ=0.05, k=6) | 2.038e+00 | 2.038e+00 |
