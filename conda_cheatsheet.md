### Conda Cheatsheet
---
I recommend using conda for managing Python environments and packages. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Miniforge](https://github.com/conda-forge/miniforge) or [Anaconda](https://www.anaconda.com/products/distribution) as per your needs. Miniconda or Miniforge are preferred for most users as they are lightweight and allow you to install only the packages you need. Anaconda comes with a lot of pre-installed packages which may not be necessary for everyone.

### Basic conda commands
list all conda environments
```bash
conda env list
```

list all packages in the current conda environment
```bash
conda list
```

list package start with "sunpy" in the current conda environment
```bash
conda list sunpy*
```

create an environment named sunpy with Python 3.12
```bash
conda create -n sunpy python=3.12 -c conda-forge
```

activate the sunpy environment
```bash
conda activate sunpy
```

install sunpy and other dependencies (not necessary to type them, left as a reference) in the sunpy environment
```bash
conda install -c conda-forge numpy scipy matplotlib astropy sunpy
```

install jupyterlab in the sunpy environment
```bash
conda install -c conda-forge jupyterlab
```

upgrade all packages in the sunpy environment
```bash
conda update --all -c conda-forge -n sunpy
```

remove the sunpy environment with all its packages
```bash
conda remove -n sunpy --all
```

remove a package (e.g. sunpy) from the current conda environment
```bash
conda remove sunpy
```