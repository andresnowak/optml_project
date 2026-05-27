# SAV isolation — matrix_factorization

lr = 1.00e-01, steps = 300, seed = 0

| variant | score (min over last 10%) | final loss |
|---|---|---|
| top_k=0 (no-SAV / paper tail) | 1.656e-04 | 1.656e-04 |
| top_k=1 | 3.749e-03 | 3.749e-03 |
| top_k=6 (paper default) | 2.315e-02 | 2.315e-02 |
| top_k=32 | 1.674e-02 | 1.674e-02 |
| gated (τ=0.05, k=6) | 5.533e-03 | 5.533e-03 |
