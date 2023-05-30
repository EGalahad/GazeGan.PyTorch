# Gaze Correction Module

This is the gaze correction module (GCM) for LAME project.

This is also a pytorch re-implementation of the paper [Dual In-painting Model for Unsupervised Gaze Correction and Animation in the Wild](https://arxiv.org/abs/2008.03834).

## Usage

To use the GCM, you need to first download the pre-trained model from [here](https://drive.google.com/drive/folders/1mhuOYrjdSmYEcpyG2prtAcdELKVRu750?usp=sharing) and put it under the `pretrained` folder.

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

