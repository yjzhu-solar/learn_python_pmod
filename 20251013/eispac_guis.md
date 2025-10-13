### GUIs provided by EISPAC 
---
Remember to activate your anaconda environment 
```bash
conda activate name_of_your_env
```
Install EISPAC 
It is recommended to install all dependencies through anaconda 
```bash
conda install -c conda-forge h5py ndcube parfive packaging tomli pyqt
```
then we install eispac through pip 
```bash
python -m pip install eispac
```
upgrade EISPAC 
```bash
python -m pip install --upgrade eispac
```

Search and download the h5 data files (calibrated to level 1, 1" and 2" slit data only). It will search for the EIS catalog sqlite file (usually the one in your SolarSoftware installation). 
```bash
eis_catalog
```

Explore the raster and fitting files 
```bash
eis_explore_raster
```

Browse and copy the fitting template 
```bash
eis_browse_templates 
```