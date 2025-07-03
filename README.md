# Code accompanying the preprint:
## ["Redundant prefrontal hemispheres adapt storage strategy to working memory demands"](https://www.biorxiv.org/content/10.1101/2025.01.15.633176)

This repository contains the code used to generate the main figures and supplementary analyses presented in the preprint. The code includes simulations, data analysis, and plotting routines.

🔒 Note: The dataset used in the study is not yet publicly available, but will be shared via this repository as soon as possible.
To run the notebooks, it is recommended to install the environment contained in the environment.yml file with conda. The same environment can be used for models and data analysis.

### Getting started:
To reproduce the results, we recommend using the environment provided in environment.yml. You can create the environment using:
```
conda env create --name <env-name> --file=environment.yml
conda activate <env-name>
```
The environment is compatible with both, data analysis and model simulations.

### Repository structure:
- **Jupyter notebooks**: Notebooks named FigureX.ipynb replicate specific figures (i.e. Figure1.ipynb replicates Figure 1 of the main manuscript). These notebooks create figures directly, using preprocessed data and simulation results.
- **Python scripts**: Scripts are prefixed by matching figure number. Scripts contain more computationally expensive preprocessing steps or simulations

### Notes
While the notebooks should run as-is (once the data is available), some files assume a specific folder structure and data location, which will be clarified once the dataset is released.
All figures should be reproducible from the notebooks.

### Contact
If you have any questions, encounter issues, or would like to discuss the methods or paper, feel free to reach out via github or email (mel.tschiersch@gmail.com)
