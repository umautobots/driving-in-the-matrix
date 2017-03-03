# Driving in the Matrix

Steps to reproduce training results for the paper 
[Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?](https://arxiv.org/abs/1610.01983)
conducted at [UM & Ford Center for Autonomous Vehicles (FCAV)](https://fcav.engin.umich.edu).

![Teaser?](bbox-and-segments.png)

Specifically, we will train [MXNet RCNN](https://github.com/dmlc/mxnet/tree/master/example/rcnn) on our 
[10k dataset](https://fcav.engin.umich.edu/sim-dataset) 
and evaluate on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php).

## System requirements

To run training, you need [CUDA 8](https://developer.nvidia.com/cuda-toolkit), [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
and a linux machine with at least one Nvidia GPU installed. Our training was conducted using 4 Titan-X GPUs.

We provide pre-trained models if you only wish to save time on training (training time per epoch for us was roughly 
10k: 40 minutes,
50k: 3.3 hours,
200k: 12.5 hours).
 

## Download the dataset

Create a directory and download the archive files for 10k images, annotations and image sets from [our website](https://fcav.engin.umich.edu/sim-dataset/).
Assuming you have downloaded these to a directory named `ditm-repro` (driving in the matrix repro):

```
$ ls -1 ditm-repro
repro_10k_annotations.tgz
repro_10k_images.tgz
repro_image_sets.tgz
```

Extract them.

```
$ pushd ditm-repro
$ tar zxvf repro_10k_images.tgz
$ tar zxvf repro_10k_annotations.tgz
$ tar zxvf repro_image_sets.tgz
$ popd
$ ls -1 ditm-repro/VOC2012
Annotations
ImageSets
JPEGImages
```


## Train on GTA

To make training as reproducible (across our own machines, and now for you!) as possible, we ran training within 
a docker container [as detailed here](https://github.com/umautobots/nn-dockerfiles/tree/master/mxnet-rcnn).

If you are familiar with MXNet and its RCNN example and already have it installed, you will likely feel comfortable 
adapting these examples to run outside of docker.

### Build the MXNet RCNN Container

```
$ git clone https://github.com/umautobots/nn-dockerfiles.git
$ pushd nn-dockerfiles
$ docker build -t mxnet-rcnn mxnet-rcnn
$ popd
```

This will take several minutes.

```
$ docker images | grep mxnet
mxnet-rcnn                 latest               bb488173ad1e        25 seconds ago      5.54 GB
```

### Kick off training

### (or download pretrained models)

## Evaluate on KITTI

### Download the KITTI object detection dataset

### Convert it to VOC format

### Evaluate GTA10k trained network on KITTI

### Convert VOC evaluations to KITTI format

### Run KITTI's benchmark on results

## Citation
If you find this useful in your research please cite:

> M. Johnson-Roberson, C. Barto, R. Mehta, S. N. Sridhar, K. Rosaen and R. Vasudevan, “Driving in the matrix: Can virtual worlds replace human-generated annotations for real world tasks?,” in IEEE International Conference on Robotics and Automation, pp. 1–8, 2017.
    
    @inproceedings{Johnson-Roberson:2017aa,
        Author = {M. Johnson-Roberson and Charles Barto and Rounak Mehta and Sharath Nittur Sridhar and Karl Rosaen and Ram Vasudevan},
        Booktitle = {{IEEE} International Conference on Robotics and Automation},
        Date-Added = {2017-01-17 14:22:19 +0000},
        Date-Modified = {2017-01-17 14:23:03 +0000},
        Keywords = {conf},
        Note = {accepted},
        Title = {Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?},
        Year = {2017}}
