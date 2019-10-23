## GAGAN
Unofficial implementation of [this paper](https://arxiv.org/pdf/1904.08144.pdf)
## How to get started ##

#### 1. Download Dataset & Preprocess it
- Download [Openbabel](http://openbabel.org/wiki/Category:Installation)
  - For Linux, run apt install.
  
```
sudo apt-get install openbabel
```
- Download and preprocess dataset.

```
./download.sh
```


#### 2. Create conda environment and install dependencies
WARNING: RDKit seems to crash when pytorch is installed with pip ([reference](https://github.com/molecularsets/moses/issues/40)). Download pytorch with conda to resolve this issue.

```
conda create -c rdkit -n {NAME} rdkit
conda activate {NAME}
conda install -c pytorch pytorch==1.2.0
pip install -r requirements.txt
```

#### 3. Train.
```
python train.py
```
