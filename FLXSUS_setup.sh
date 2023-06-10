# script to init FLXSUS environment

conda create --name flxsus
conda activate flxsus
conda install -c anaconda spyder
conda install -c anaconda openpyxl
conda install -c conda-forge pingouin
conda install -c anaconda seaborn
conda install -c anaconda jupyter

# does not work with spyder for some reason...
