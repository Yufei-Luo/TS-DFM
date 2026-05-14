# Introduction

This repo contains the codes for "Generative Flow Model on Distance Geometry for Predicting Transition States of Chemical Reactions" to be published at *Nature Communications*.

# Dataset and pretrained checkpoints

Before reproducing the results on Transition1x dataset, please download the `.h5` file in [Item - Transition1x - figshare - Figshare](https://figshare.com/articles/dataset/Transition1x/19614657/4?file=36035789).

We have provided the trained checkpoints of baseline methods and our proposed TS-DFM. They can be downloaded from **https://doi.org/10.5281/zenodo.17672638**

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

If you find this repo useful, please cite our article (The citation at *Nature Communications* will be updated after official publishment.)
````latex
@misc{luo2025generatingtransitionstateschemical,
      title={Generating transition states of chemical reactions via distance-geometry-based flow matching}, 
      author={Yufei Luo and Xiang Gu and Jian Sun},
      year={2025},
      eprint={2511.17229},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.17229}, 
}
````