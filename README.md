# Driving in the Matrix

Steps to reproduce training results for the paper 
[Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?](https://arxiv.org/abs/1610.01983)
conducted at [UM & Ford Center for Autonomous Vehicles (FCAV)](https://fcav.engin.umich.edu).

The code to capture the dataset can be found [in our GTAVisionExport repository](https://github.com/umautobots/GTAVisionExport).

![Teaser?](bbox-and-segments.png)

Specifically, we will train [MXNet RCNN](https://github.com/dmlc/mxnet/tree/master/example/rcnn) on our 
[10k dataset](https://fcav.engin.umich.edu/sim-dataset) 
and evaluate on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php).

## System requirements

To run training, you need [CUDA 8](https://developer.nvidia.com/cuda-toolkit), [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
and a linux machine with at least one Nvidia GPU installed. Our training was conducted using 4 Titan-X GPUs.

Training time per epoch for us was roughly 
10k: 40 minutes,
50k: 3.3 hours,
200k: 12.5 hours. We plan on providing the trained parameters from the best performing epoch for 200k soon.
 

## Download the dataset

Create a directory and download the archive files for 10k images, annotations and image sets from [our website](https://fcav.engin.umich.edu/sim-dataset/).
Assuming you have downloaded these to a directory named `ditm-data` (driving in the matrix data):

```
$ ls -1 ditm-data
repro_10k_annotations.tgz
repro_10k_images.tgz
repro_image_sets.tgz
```

Extract them.

```
$ pushd ditm-data
$ tar zxvf repro_10k_images.tgz
$ tar zxvf repro_10k_annotations.tgz
$ tar zxvf repro_image_sets.tgz
$ popd
$ ls -1 ditm-data/VOC2012
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

### Download pre-trained VGG16 network

```
$ mkdir -p pretrained-networks
$ cd pretrained-networks && wget http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params && cd -
```

### Kick off training

```
$ mkdir -p training-runs/mxnet-rcnn-gta10k
$ nvidia-docker run --rm --name run-mxnet-rcnn-end2end \
  `#container volume mapping` \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k:/media/output \
  -v `pwd`/pretrained-networks:/media/pretrained \
  -v `pwd`/ditm-data:/root/mxnet/example/rcnn/data/VOCdevkit \
  -it mxnet-rcnn \
  `# python script` \
  python train_end2end.py \
  --image_set 2012_trainval10k \
  --root_path /media/output \
  --pretrained /media/pretrained/vgg16 \
  --prefix /media/output/e2e \
  --gpus 0 \
  2>&1 | tee training-runs/mxnet-rcnn-gta10k/e2e-training-logs.txt
  
...
INFO:root:Epoch[0] Batch [20]	Speed: 6.41 samples/sec	Train-RPNAcc=0.784970,	RPNLogLoss=0.575420,	RPNL1Loss=2.604233,	RCNNAcc=0.866071,	RCNNLogLoss=0.650824,	RCNNL1Loss=0.908024,	
INFO:root:Epoch[0] Batch [40]	Speed: 7.10 samples/sec	Train-RPNAcc=0.807546,	RPNLogLoss=0.539875,	RPNL1Loss=2.544102,	RCNNAcc=0.895579,	RCNNLogLoss=0.461218,	RCNNL1Loss=1.019715,	
INFO:root:Epoch[0] Batch [60]	Speed: 6.76 samples/sec	Train-RPNAcc=0.822298,	RPNLogLoss=0.508551,	RPNL1Loss=2.510861,	RCNNAcc=0.894723,	RCNNLogLoss=0.406725,	RCNNL1Loss=1.005053,	
...
```

As the epochs complete, the trained parameters will be available inside `training-runs/mxnet-rcnn-gta10k`.

## Training on other segments

To train on 50k or 200k, first download and extract `repro_200k_images.tgz` and `repro_200k_annotations.tgz` and then
run a similar command as above but with `image_set` set to `2012_trainval50k` or `2012_trainval200k`.

## Evaluate on KITTI

### Download the KITTI object detection dataset

Visit [KITTI's object detection landing page](http://www.cvlibs.net/datasets/kitti/eval_object.php) and follow the links named:

- "Download training labels of object data set"
- "Download left color images of object dataset"

Download these files into a folder named `datasets/kitti/downloads`

```
$ mkdir -p datasets/kitti/downloads
$ cd datasets/kitti/downloads && wget http://www.cvlibs.net/download.php?file=data_object_image_2.zip && cd -
$ cd datasets/kitti/downloads && wget http://www.cvlibs.net/download.php?file=data_object_label_2.zip && cd -
$ ls -lh datasets/kitti/downloads
total 12G
-rw-rw-r-- 1 krosaen krosaen  12G Sep 11 14:59 data_object_image_2.zip
-rw-rw-r-- 1 krosaen krosaen 5.4M Sep 11 15:00 data_object_label_2.zip
```

extract them:

```
$ unzip datasets/kitti/downloads/data_object_label_2.zip -d datasets/kitti
$ unzip datasets/kitti/downloads/data_object_image_2.zip -d datasets/kitti
$ ls datasets/kitti/training/
image_2  label_2
```

create a training set text file expected by the conversion script

```
$ cd datasets/kitti && ls -1 training/image_2 | cut -d. -f1 > train.txt && cd -
$ head datasets/kitti/train.txt
000000
000001
000002
000003
000004
000005
000006
000007
000008
000009

```

### Convert it to VOC format

Install our [Visual Object Dataset Converter tool](https://github.com/umautobots/vod-converter):

```
$ git clone https://github.com/umautobots/vod-converter.git
```

and convert KITTI to PASCAL VOC format:

```
$ python3.6 vod-converter/vod_converter/main.py --select-only-known-labels --from kitti --from-path datasets/kitti --to voc --to-path datasets/kitti-voc
INFO:root:Namespace(filter_images_without_labels=False, from_key='kitti', from_path='datasets/kitti', select_only_known_labels=True, to_key='voc', to_path='datasets/kitti-voc')
Successfully converted from kitti to voc.
$ ls datasets/kitti-voc/VOC2012/
Annotations  ImageSets  JPEGImages
```

### Evaluate GTA10k trained network on KITTI

Now that we have KITTI in VOC format (expected by the MXNet RCNN code), we can evaluate our trained model on KITTI.


Here we evaluate on the 1st epoch.

Note: there is one more hack we need: the MXNet RCNN code is expecting jpeg images, but KITTI is in png, so rather
than having to convert the images, we can make a modest update to the script and map that in via a docker
volume too.

Download this repo to get the hack ready

```
$ git clone https://github.com/umautobots/driving-in-the-matrix.git
```


Prepare some directories to store the training results

```
mkdir -p training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/cache
```

Run `mxnet-rcnn` forward using the network values from the 1st epoch, storing results in the directories created
above.


```
nvidia-docker run --rm \
  `#container volume mapping for dataset` \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k:/media/output \
  -v `pwd`/datasets/kitti-voc:/root/mxnet/example/rcnn/data/VOCdevkit \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch:/root/mxnet/example/rcnn/data/VOCdevkit/results \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/cache:/media/ngv/output/cache \
  `#container volume mapping for dataset reader for png` \
  -v `pwd`/driving-in-the-matrix/mxnet-rcnn-hacks/pascal_voc_png.py:/root/mxnet/example/rcnn/rcnn/dataset/pascal_voc.py:ro \
  -it mxnet-rcnn \
  `# python script` \
  python test.py \
  --image_set 2012_trainval \
  --root_path /media/output \
  --network vgg \
  --prefix /media/output/e2e \
  --epoch 1 \
  --gpu 0 \
  2>&1 | tee -a `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/testing-logs.txt
...
num_images 7481
wrote gt roidb to /media/output/cache/voc_2012_trainval_gt_roidb.pkl
testing 0/7481 data 2.8556s net 2.2184s post 0.0030s
testing 1/7481 data 0.0364s net 0.0505s post 0.0027s
testing 2/7481 data 0.0479s net 0.0442s post 0.0016s
testing 3/7481 data 0.0336s net 0.0443s post 0.0015s
testing 4/7481 data 0.0253s net 0.0428s post 0.0013s
testing 5/7481 data 0.0169s net 0.0429s post 0.0006s
testing 6/7481 data 0.0162s net 0.0430s post 0.0012s
testing 7/7481 data 0.0168s net 0.0436s post 0.0010s
testing 8/7481 data 0.0166s net 0.0444s post 0.0011s
testing 9/7481 data 0.0163s net 0.0441s post 0.0013s
...
testing 7479/7481 data 0.0301s net 0.0451s post 0.0021s
testing 7480/7481 data 0.0220s net 0.0451s post 0.0016s
...
Writing car VOC results file
AP for car = 0.4535
...
```

Note that at this point we have VOC AP scores.

### Convert VOC evaluations to KITTI format

Download our script for converting:

```
$ git clone https://github.com/umautobots/voc-predictions-to-kitti.git
```

Make a directory to store output:

```
$ mkdir training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels
```

Run it to convert (apologies for awkward manner in which it was run, I got a little **too** into docker for a while there :))

```
$ docker run -it --rm --name run-voc-predictions-to-kitti \
  -v `pwd`/voc-predictions-to-kitti:/usr/src/myapp:ro \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/VOC2012/Main:/input/VOC:ro \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels:/output/KITTI \
  -w /usr/src/myapp \
  python:3.6 python doit.py
cat
sofa
sheep
boat
bus
motorbike
cow
dog
horse
car
pottedplant
tvmonitor
person
aeroplane
diningtable
bicycle
bird
train
bottle
chair
```

Notice the results are now available in KITTI style:

```
$ ls -1 training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels/ | head
000000.txt
000001.txt
000002.txt
000003.txt
000004.txt
000005.txt
000006.txt
000007.txt
000008.txt
000009.txt
$ $ head training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels/000000.txt
motorbike -1 -1 -1 244.50 232.90 321.00 283.20 -1 -1 -1 -1 -1 -1 -1 0.297
motorbike -1 -1 -1 238.70 207.40 400.20 288.90 -1 -1 -1 -1 -1 -1 -1 0.025
motorbike -1 -1 -1 235.80 233.70 276.40 258.40 -1 -1 -1 -1 -1 -1 -1 0.006
motorbike -1 -1 -1 268.40 220.60 332.60 255.90 -1 -1 -1 -1 -1 -1 -1 0.002
motorbike -1 -1 -1 223.20 241.10 250.30 258.70 -1 -1 -1 -1 -1 -1 -1 0.002
motorbike -1 -1 -1 291.80 242.70 343.30 273.90 -1 -1 -1 -1 -1 -1 -1 0.001
Car -1 -1 -1 246.90 193.50 442.60 262.40 -1 -1 -1 -1 -1 -1 -1 0.063
Car -1 -1 -1 317.70 218.80 417.40 246.70 -1 -1 -1 -1 -1 -1 -1 0.049
Car -1 -1 -1 335.60 208.40 402.10 228.90 -1 -1 -1 -1 -1 -1 -1 0.034
Car -1 -1 -1 255.70 224.70 337.90 280.70 -1 -1 -1 -1 -1 -1 -1 0.028
```

### Run KITTI's benchmark on results

Build the container for evaluating kitti:

```
$ pushd nn-dockerfiles
$ docker build -t kitti-evaluate kitti-evaluate
$ popd
$ docker images | grep kitti-evaluate
kitti-evaluate              latest                 0d34717f0e7f        20 seconds ago      977MB
```

Evaluate 70% overlap:

```
$ mkdir -p training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels/evaluate70
$ docker run --rm \
  -v `pwd`/datasets/kitti:/root/data/ground_truth:ro \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels:/root/results/my-kitti-labels/data:ro \
  -v `pwd`/training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels/evaluate70:/root/results/my-kitti-labels \
  -it kitti-evaluate ./evaluate_object70 my-kitti-labels
Thank you for participating in our evaluation!
Loading detections...
  done.
Mean AP Easy 0.435010
Mean AP Moderate 0.292753
Mean AP Hard 0.241420
PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
==> 1 page written on `car_detection.pdf'.
PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
==> 1 page written on `car_orientation.pdf'.
PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
==> 1 page written on `pedestrian_detection.pdf'.
PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
==> 1 page written on `pedestrian_orientation.pdf'.
PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
==> 1 page written on `cyclist_detection.pdf'.
PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
==> 1 page written on `cyclist_orientation.pdf'.
Your evaluation results are available at:
http://www.cvlibs.net/datasets/kitti/user_submit_check_login.php?benchmark=object&user=(null)&result=my-kitti-labels
```

and you will then have the files available:

```
$ ls training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels/evaluate70
data  stats_car_detection.txt  stats_car_orientation.txt    stats_cyclist_orientation.txt   stats_pedestrian_orientation.txt
plot  stats_car_mAP.txt        stats_cyclist_detection.txt  stats_pedestrian_detection.txt
```

where you can see the 70% mAP:

```
$ cat training-runs/mxnet-rcnn-gta10k/evaluate-on-kitti-1-epoch/kitti-labels/evaluate70/stats_car_mAP.txt
0.435010
0.292753
0.241420
```

You can evaluate the 50% overlap numbers with a similar command per [instructions on wiki](https://github.com/umautobots/nn-dockerfiles/tree/master/kitti-evaluate).



## Citation
If you find this useful in your research please cite:

> M. Johnson-Roberson, C. Barto, R. Mehta, S. N. Sridhar, K. Rosaen and R. Vasudevan, ?Driving in the matrix: Can virtual worlds replace human-generated annotations for real world tasks?,? in IEEE International Conference on Robotics and Automation, pp. 1?8, 2017.
    
    @inproceedings{Johnson-Roberson:2017aa,
        Author = {M. Johnson-Roberson and Charles Barto and Rounak Mehta and Sharath Nittur Sridhar and Karl Rosaen and Ram Vasudevan},
        Booktitle = {{IEEE} International Conference on Robotics and Automation},
        Date-Added = {2017-01-17 14:22:19 +0000},
        Date-Modified = {2017-02-23 14:37:23 +0000},
        Keywords = {conf},
        Pages = {1--8},
        Title = {Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?},
        Year = {2017}}
