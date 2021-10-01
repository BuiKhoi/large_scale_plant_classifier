# Large scale plant recognition

This is a repository of training large scale plant recognition model, the codes in this repo is mostly inspired by [TreB1eN's repo](https://github.com/TreB1eN/InsightFace_Pytorch)

# 0. Environment setup
I used this repo to train on [Google Colab](https://research.google.com/colaboratory/), if you want to setup the environment on colab, run:
```bash
pip install -r requirements_colab.txt
```

Otherwise, if you're running on a Linux computer, run:
```bash
pip install -r requirements_local.txt
```

# 1. Data preparation
This section will guide you how to prepare your dataset to be trained, and also create a validation dataset for training process

First you have to prepare your dataset matching the following structure:
```
dataset_folder
    ├── class_1
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   ├── 00003.jpg
    │   ├── 00004.jpg
    │   .
    │   .
    │   .
    │   └── 00099.jpg
    └── class_2
        ├── 00001.jpg
        ├── 00002.jpg
        ├── 00003.jpg
        .
        .
        .
        └── 00099.jpg
```

Then, to prepare data for training, checkout `prepare_data.py`:
```bash
python prepare_data.py -h
```

To run the prepare script:
```bash
python prepare_data.py --dataset_dir /content/training_data/Flower --min_set_size 3 \
    --resize --make_val_data --val_dir /content/training_data/val/
```

Now you should have your data prepared. And process to training phase

# 2. Training model
After preparing the data following the previous instructions, you can take a look at training options:
```bash
python train.py -h
```

You can start training with:
```bash
python train.py -net mobilefacenet -b 256 -w 4 -e 500 \
    -d /content/training_data/Flower -v /content/training_data/val \
    -load '2021-10-01-15-44_loss:39.85_step:205_None.pth'
```

Note: 
* If you want to train from scratch, please remove the argument `-load`
* Training loss and accuracy will be printed along the way of training, remember to take a look
