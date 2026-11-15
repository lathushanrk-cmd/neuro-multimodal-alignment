# Benchmarking

To measure runtime on *your own computer*, run:

```bash
python reproduce/benchmark.py
```

The script measures RSA, five-fold PCA-preprocessed CCA and RSA permutations at several synthetic sample counts. Printed numbers represent the executing environment only. Benchmarks are illustrative and not a guarantee of large-scale data capability. RDM memory is quadratic in the number of units, which makes RSA a poor choice for unrestricted large numbers of trials without adaptation.
