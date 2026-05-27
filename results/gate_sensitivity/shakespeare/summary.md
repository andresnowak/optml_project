# Gate sensitivity — shakespeare

lr = 3.00e-04, steps = 1000, top_k = 6, seed = 0

score = min loss over last 10% of training; τ=0 ⇒ gate disabled (paper default).

| τ \ window | 5 | 10 | 20 |
|---|---|---|---|
| 0 | 2.438e+00 | 2.438e+00 | 2.438e+00 |
| 0.01 | 2.453e+00 | 2.458e+00 | 2.465e+00 |
| 0.05 | 2.438e+00 | 2.438e+00 | 2.446e+00 |
| 0.1 | 2.438e+00 | 2.438e+00 | 2.439e+00 |
| 0.2 | 2.438e+00 | 2.438e+00 | 2.439e+00 |
