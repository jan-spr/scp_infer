# scp-infer: Causal Iference using perturbed single-cell gene expression data
Apply a number of causal inference algorithms to experimental perturbed single-cell gene exression data.
Built using scanpy

## Contents:
### 1. [scp_infer](scp_infer)
Module that cointains all of the functionality:
-adata:
  Data manipulation and annotation using Anndata/scanpy
-inference:
  Apllication of inference algorithms

### 2. [algorithm_implementations](algorithm_implementations)

folder that houses local files for algorithms where they are not available as a package
- [GIES](https://github.com/juangamella/gies)
- [DCDI](https://github.com/slachapelle/dcdi/tree/master/dcdi)

