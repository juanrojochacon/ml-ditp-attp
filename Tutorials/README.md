- to check the version of your tensorflow:
python3 -c 'import tensorflow as tf; print(tf.__version__)'
(we need 1.13)

- instructions (linux/mac):
make sure you have anaconda/conda:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

- for linux and macs:
```
conda create -n ml-tutorials python=3.6
conda activate ml-tutorials
pip install ipython jupyter pandas sklearn matplotlib
conda install -c conda-forge tensorflow=1.13
```