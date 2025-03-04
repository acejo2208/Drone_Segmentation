# Drone Segmentation Projects
- This project is built on [mmdetection](https://github.com/open-mmlab/mmdetection).
- The Drone Datasets are trained and evaluated on [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
## Environment Requirements 
- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n drone_islab python=3.7 -y
    conda activate drone_islab
    ```

2. Install PyTorch and torchvision following the [official instructions
](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt pacakge,
    you can use more CUDA versions such as 9.0.

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```

4. Clone the MMDetection repository.

    ```shell
    git clone https://github.com/acejo2208/Drone_Segmentation
    cd Drone_Segmentation
    ```

5. Install build requirements and then install MMDetection.

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) to build an image. Ensure that you are using [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMDetection with conda.

```shell
conda create -n drone_islab python=3.7 -y
conda activate drone_islab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd Drone_Segmentation
pip install -r requirements/build.txt
pip install -v -e .
```

### Developing with multiple MMDetection versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection in the current directory.

To use the default MMDetection installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
## Train and Test Models on Drone Segmentation Dataset
- To change the configuration, please go to [Drone_Segmentation/configs/mask_rcnn_drone/mask_rcnn.py](https://github.com/acejo2208/Drone_Segmentation/blob/main/configs/mask_rcnn_drone/mask_rcnn.py)
- The checkpoints file under [Drone_Segmentation/work_dirs/vis_drone](https://github.com/acejo2208/Drone_Segmentation/tree/main/work_dirs/vis_drone)
##### Train with a single GPU
```shell
python tools/train.py ${CONFIG_FILE}
E.g., for our dataset
python tools/train.py configs/mask_rcnn_drone/mask_rcnn.py
```

##### Train with multiple GPUs
```shell
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
E.g., Our dataset
bash ./tools/dist_train.py configs/mask_rcnn_drone/mask_rcnn.py 4
```
##### Test with a single GPU
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
E.g., Our dataset
python tools/test.py configs/mask_rcnn_drone/mask_rcnn.py work_dirs/vis_drone/mask_rcnn.pth --eval bbox segm
```
1. Visualize the results. Press and key for the next image
```shell
python tools/test.py configs/mask_rcnn_drone/mask_rcnn.py work_dirs/vis_drone/mask_rcnn.pth --show
```
2. Test and save the painted images for future visualization
```shell
python tools/test.py configs/mask_rcnn_drone/mask_rcnn.py work_dirs/vis_drone/mask_rcnn.pth --show_dir visulization
```
3. Test and evaluate the mAP
```shell
python tools/test.py configs/mask_rcnn_drone/mask_rcnn.py work_dirs/vis_drone/mask_rcnn.pth --eval bbox segm
```

## Acknowledgement
Many thanks to the open source codes, i.e., [mmdetection](https://github.com/open-mmlab/mmdetection) and [labelme](https://github.com/wkentaro/labelme).

