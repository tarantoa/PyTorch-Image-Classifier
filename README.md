# PyTorch-Image-Classifier

A basic dense neural network implemented with PyTorch to classify images in this flower dataset https://archive.ics.uci.edu/ml/datasets/iris

Required libraries: argparse numpy Pillow torch torchvision

## Training the model

The model can be trained with the `train.py` script.

Usage: `python3 train.py [--options] [data_directory]`

Default: `python3 train.py flowers`

### Optional arguments
```
-s, --save-directory specifies the directory to which trained models will be saved (./checkpoints by default)
-a, --arch specifies the architecture to use for transfer learning (vgg16 (default), alexnet, googlenet)
-g, --gpu trains the model using the gpu (disabled by default)
-l, --learn-rate specify learn rate (0.003 by default)
-H, --hidden-units specify the number of nodes in the hidden layer (2508 by default)
-e, --epochs specify the number of training epochs (20 by default)
```

The flowers subdirectory folder structure (`train, test, valid`) is important and should be maintained if the `flowers` folder is renamed.


Once trained, the model is saved to the specified save directory.

## Making predictions

Predictions with the trained model are made with the `predict.py` script.

Usage: `python3 predict.py [--options] [image_path] [checkpoint_path]`

Where `[image_path]` is the relative path to an image to predict and `[checkpoint_path]` is the relative path to the saved model.

Example: `python3 predict.py flowers/test/1/image_06743.jpg checkpoints/model_vgg16_1.pth` 

Sample output:
```
image_06743
1. pink primrose: 31.488%
2. hibiscus: 24.475%
3. tree mallow: 11.602%`
```
### Optional arguments
```
-k, --top_k specify the number of category predictions to show (3 by default)
-g, --gpu make predictions using the gpu (disabled by default)
-c, --category_names path to json file specifying the map of category numbers to names (cat_to_names.json by default)
```
