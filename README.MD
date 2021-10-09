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

## 2.1 Modify training size
The default training image size is `224x224`, if you want to change it, just simply change the size when running `python prepare_data.py ...` process, but then, when you run training, you should have an error looks like:
```
RuntimeError: Function MmBackward returned an invalid gradient at index 0 - got [4, 32768] but expected shape compatible with [4, 247808]
```

This is because when you change the image size, it should affect the flatten function, to solve this, head to `model.py`, and change line 209, which will look like this:
```
208|    self.conv_6_flatten = Flatten()
209|    self.linear = Linear(32768, embedding_size, bias=False)
210|    self.bn = BatchNorm1d(embedding_size)
```
You can change the `32768` number to `247808`, which our model requires.
And then start training again, the code should be working smoothly. Remember whenever you change image size, you will have to modify this number


# 3. Make serving model
For the best inference with this model, we should convert the model to tensorflow serving format, which will allow us to use [gRPC](https://grpc.io/) to inference model.
To do so, run:
```bash
python make_serving_model.py --model_file /path/to/trained/model/file
```

You can checkout options for making serving model with:
```bash
python make_serving_model.py -h
```
Now you should have your serving model ready, follow the steps to [Examine and deploy serving model](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#examine_your_saved_model)

Note: Tensorflow serving model running on docker is highly recommended

# 4. Make databank
For the final process, we will find image embedding in the databank of embeddings, which means we have to calculate them first, to do so, run:
```bash
python make_databank.py --model_path path/to/your/model --data_path path/to/data/folder
    --save_path path/to/save/folder --dataset_name plant_entire
```

You should have a folder containing embedding for individual images, and a `.npz` file with all these images compressed for ease of use

# 5. Inference
After creating the databank, you should be able to run inference on your images, first, prepare your test dataset with the structer
being the same as `1. Data preparation` section, then run:
```bash
python infer_on_image.py --model_path path/to/your/model \
    --images_path path/to/your/test/dataset/ \
    --databank path/to/your/databank
```

It should show you the files and the predicted classes.
