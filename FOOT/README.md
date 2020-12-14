## Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Numpy
- Pytorch 0.4+

Preparation 1: create folder for dataset.
first, obtain foot data

then, get the directory structure
``` 
├── FOOT
　　　├── data
　　　　　　├── foot
``` 
Preparation 2: Put the images with the same id in one folder. You may use 
```bash
python prepare_for_foot.py
```

Finally, conduct training, testing and evaluating
```bash
python train.py
python test.py
python evaluate.py
```


