# Introduction

This repo contains the code for "Generative Flow Model on Distance Geometry for Predicting Transition States of Chemical Reactions" published at *Nature Communications*.

# Dataset and pretrained checkpoints

Before reproducing the results on Transition1x dataset, please download the `.h5` file in [Item - Transition1x - figshare - Figshare](https://figshare.com/articles/dataset/Transition1x/19614657/4?file=36035789).

The preprocessed RGD1 subset used in our work can be downloaded from [https://doi.org/10.5281/zenodo.20179684](https://doi.org/10.5281/zenodo.20179684).

We have provided the trained checkpoints of baseline methods and our proposed TS-DFM. They can be downloaded from [https://doi.org/10.5281/zenodo.20179684](https://doi.org/10.5281/zenodo.20179684)

# Reproduction

For reproducing the results of TS-DFM, please run the Python scripts in the `Scripts` folder. The files containing hyperparameters are in the `Configs` folder.

The baseline methods are provided in the following folders:

````bash
React-OT on ts1x:react-ot-main/
React-OT on rgd1:react-ot-rgd1/
PSI-based:learnts_main/
OA-ReactDiff: OAReactDiff-main/
NeuralNEB: NeuralNEB/
````

# Citation

If you find this repo useful, please cite our article as follows
````latex
@article{luoGenerativeFlowModel2026,
  title = {Generative Flow Model on Distance Geometry for Predicting Transition States of Chemical Reactions},
  author = {Luo, Yufei and Gu, Xiang and Sun, Jian},
  year = 2026,
  month = jun,
  journal = {Nature Communications},
  volume = {17},
  number = {1},
  pages = {7497},
  issn = {2041-1723},
  doi = {10.1038/s41467-026-74101-0},
}
````
