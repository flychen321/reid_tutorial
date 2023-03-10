********************
Source Code is released for the paper entitled Unsupervised Person Re-identification via Multi-domain Joint Leatning.
********************
## Prerequisites
- Python 3.6
- GPU Memory >= 20G
- Numpy
- Pytorch 0.4+

Preparation 1: create folder for dataset.

first, download Market-1501 and DukeMTMC-reID dataset from the links below:

google drive: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing
              https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O

baidu disk: https://pan.baidu.com/s/1ntIi2Op
            https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw

second,
```bash
mkdir data
unzip Market-1501-v15.09.15.zip
ln -s Market-1501-v15.09.15 market
unzip DukeMTMC-reID.zip
ln -s DukeMTMC-reID duke
``` 
then, get the directory structure
``` 
├── MDJL
    ├── data
            ├── market
            ├── Market-1501-v15.09.15
            ├── duke
            ├── DukeMTMC-reID
``` 


Preparation 2: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```

Finally, train, test and evaluate the re-ID model with the below command:
```bash
python train.py
```

