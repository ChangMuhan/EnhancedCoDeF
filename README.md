# EnhancedCoDeF
(MUST 2509 Computer Vison course project)


Improvements to [**CoDeF: Content Deformation Fields for Temporally Consistent Video Processing**](https://github.com/ant-research/CoDeF/) (CVPR 2024)

<img src='./demo.gif'></img>

## Requirements

The codebase is tested on

* Ubuntu 20.04
* Python 3.10
* [PyTorch](https://pytorch.org/) 2.0.0
* [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) 2.0.2
* 1 NVIDIA GPU (V100) with CUDA version 12.6.

To use video visualizer, please install `ffmpeg` via

```shell
sudo apt-get install ffmpeg
```

For additional Python libraries, please install with

```shell
pip install -r requirements.txt
```

Our code also depends on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
See [this repository](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)
for Pytorch extension install instructions.


## Data
1. Our model is trained on **single videos**. And the videos are selected from DAVIS 2017 dataset.

We prepared a video in `./videos/tennis-vest/` for training and evaluating.

2. We extract optical flows of video sequences using [RAFT](https://github.com/princeton-vl/RAFT). To get started, please follow the instructions provided [here](https://github.com/princeton-vl/RAFT#demos) to download their pretrained model. Once downloaded, place the model in the `data_preprocessing/RAFT/models` folder (we use 'raft-sintel.pth' here).


## Pretrained checkpoints

You can download checkpoints pre-trained on the provided video via

| Video Name | Config |                           Download                           |
| :-------- | :----: | :----------------------------------------------------------: |
| tennis-vest | configs/base.yaml |  [Google drive link]([https://drive.google.com/file/d/1jbVXusAzxLsW2iIoULv76ZdH0MGtA5Ml/view?usp=drive_link] |


And organize files as follows

```
EnhancedCoDeF
│
└─── ckpts/{VIDEO_NAME}
    │
    └─── segment_0
        │
        └─── ckpt_final.pt
    │
    └─── segment_1
        │
        └─── ckpt_final.pt

    |
    └─── ...
```

## Fast Test
To simply use our model to test the reconstruction of a video, run:
```shell
./test.sh
```
It will automatically generate the RAFT optical flow files and evaluate the model. Finally you will see the reconstructed video in `./results/{VIDEO_NAME}/`.

Please make sure the checkpoints above are placed in `./ckpts/VIDEO_NAME`, and there are optical flow files in `./videos/{VIDEO_NAME}_flow`


## Train A New Model
You can also use your own video to train the model and observe the reconstruction effect. Make sure your own video are organized as follows:
```
EnhancedCoDeF
│
└─── videos/{YOUR_OWN_VIDEO}
    │
    └─── 00000.jpg
    │
    └─── 00001.jpg
    |
    └─── ...
```

Then run:

```shell
./train.sh
```
