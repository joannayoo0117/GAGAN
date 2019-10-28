## GAGAN
Unofficial implementation of [this paper](https://arxiv.org/pdf/1904.08144.pdf)
## How to get started ##

#### 1. Download Dataset & Preprocess it
- Download [Openbabel](http://openbabel.org/wiki/Category:Installation)
  - For Linux, run apt install.
  
```
sudo apt-get install openbabel
```
- Download and preprocess dataset (PDBBind, DUD-E).
- **Warning** This will take about 3Gb of your space.  

```
./download.sh
```
*Optional*
- To run docking calculation across DUD-E and PDBBind, download [smina](https://sourceforge.net/projects/smina/) and place the `smina.static` file to `./data/docking`

```
cd ./data/docking
./generate_docking.sh
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
