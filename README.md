# Graph Convolutional Kernel Machine GCKM
GCKM is introduced in "Unsupervised Neighborhood Propagation Kernel Layers for Semi-supervised Node Classification". This repository contains the code to be used to reproduce results and/or as a starting point for future work. Links to the paper are provided below.


## Abstract

We present a deep Graph Convolutional Kernel Machine(GCKM) for semi-supervised node classification in graphs. The method is built of two main types of blocks: (i) We introduce unsupervised kernel machine layers propagating the node features in a one-hop neighborhood, using implicit node feature mappings. (ii) We specify a semi-supervised classification kernel machine through the lens of the Fenchel-Young inequality. We derive an effective initialization scheme and efficient end-to-end training algorithm in the dual variables for the full architecture. The main idea underlying GCKM is that, because of the unsupervised core, the final model can achieve higher performance in semi-supervised node classification when few labels are available for training. Experimental results demonstrate the effectiveness of the proposed framework. 

## Useful links
[preprint](https://arxiv.org/abs/2301.13764), includes Appendix

[proceedings AAAI-24](link coming soon)

[project page](http://www.sonnyachten.com/gckm)

## Cite 
[BibTex](https://github.com/sonnyachten/GCKM/blob/main/achten_gckm_2024.bib)
```
@article{achten_gckm_2024,
	title = {Unsupervised {Neighborhood} {Propagation} {Kernel} {Layers} for {Semi-supervised} {Node} {Classification}},
	volume = {38},
	url = {},
	doi = {},
	number = {},
	journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
	author = {Achten, Sonny and Tonin, Francesco and Patrinos, Panagiotis and Suykens, Johan A.K.},
	month = {},
	year = {2024},
	pages = {},
}
```

## Code Usage

### Download

First, navigate to the unzipped directory and install required python packages with the provided `requirements.txt` file. This is explained in the following section.

### Install packages in conda environment

Run the following in terminal. This will create a conda environment named *gckm*.

```
conda create --name gckm python=3.10
```

Activate the conda environment with the command `conda activate gckm`. To install the required dependencies, run:

```R
pip install -r requirements.txt
```

Remember to install `pytorch-geometric` separately from their website according to your setup.

### Train

Activate the conda environment `conda activate gckm`.

The configuration is done using YAML files with [hydra](https://hydra.cc/). Available configurations are in the `conf` directory. You may edit the provided default_config.yaml file or write your own YAML files.

To reproduce the reported experiments, choose one of the config files (e.g., `table2_cora_gckm.yaml`) and run the following command:
```R
python run_experiment.py --config-name table2_cora_gckm
```

