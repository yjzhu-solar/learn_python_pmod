It would be helpful to create a new conda environment for this package.

```bash
conda create -n spiceitup python=3.13
```
Then we activate the environment.

```bash
conda activate spiceitup
```

Although SPICEITUP is destributed via PyPI, it is recommended to install most of its dependencies via conda-forge channel.

```bash
conda install -c conda-forge sunraster sunpy astropy matplotlib numpy scipy pandas pyqt 
```

Finally, we can install SPICEITUP via pip.
Go to a directory (e.g., ~/Downloads) and clone the repository from Gitlab.
```bash
cd ~/Downloads
git clone https://git.ias.u-psud.fr/spice/data_quicklook.git
cd data_quicklook
```

There is a hard-coded path in the setup.py file that needs to be changed.
Open setup.py in a text editor and change the line 7 from
```python
with open('/srv/data_quicklook/' + filename, 'r') as f:
```
to
```python
with open('./' + filename, 'r') as f:
```

Then we can install the package.
```bash
python3 -m pip install .
```

Now you can run the package by executing
```bash
spiceitup
```

Note that in some cases (e.g., Krzysztof's Mac Silicon), you may need to install the packge in editable mode, which means you need to add the `-e` option to the pip install command. And you should always keep the directory (do not delete it!) where you cloned the repository when you run the command.
```bash
python3 -m pip install -e .
```