# Gaze Correction Module

This is a pytorch re-implementation of the paper [Dual In-painting Model for Unsupervised Gaze Correction and Animation in the Wild](https://arxiv.org/abs/2008.03834). The official tensorflow implementation can be found [here](https://github.com/zhangqianhui/GazeAnimation).

## Set up

### convert tensorflow checkpoints to pytorch state dicts

I followed the instructions from [here](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28).

I provide a [converting notebook](./load_pretrained_tf2pt.ipynb) that you can use easily.

### load a pretrained pytorch model

To use the GazeGan, you need to first download the pre-trained model from [here](https://drive.google.com/drive/folders/1mhuOYrjdSmYEcpyG2prtAcdELKVRu750?usp=sharing) and put it under the `pretrained` folder.

The folder structure should be:

```
pretrained/
    checkpointv1.pt
    checkpointv2.pt
```

## Models

The GazeGan has the following fundamental model components:
1. An `ContentEncoder` class which extracts angle-invariant features from the local region.

2. An `GazeCorrection` class which takes the angle-invariant features and the corrupted image (eyes masked) as input and outputs the inpainted image with eyes looking at the camera.

3. A `Discriminator` class which is used for adversarial training.


The `GazeCorrection` class uses UNet architecture.

These components are combined in the `GazeGan` class, which encapsulates the inpainting logic.

These components are organized in the following folder structure:

```
models/
    content_encoder.py
    gaze_correction.py
    discriminator.py
model.py
```

## Usage

To use this library, you only need to import the `load_model` function to load a pretrained model.

```python
from gazegan import load_model

model = load_model()

# Then you can use the model to correct the gaze of an image.

xr = model(x, x_mask, left_eye, right_eye)
# x: torch.Tensor, shape=(1, 3, H=256, W=256)
# x_mask: torch.Tensor, shape=(1, 1, H=256, W=256)
# left_eye: torch.Tensor, shape=(1, 3, H=64, W=64)
# right_eye: torch.Tensor, shape=(1, 3, H=64, W=64)
# xr: torch.Tensor, shape=(1, 3, H=256, W=256)
```
