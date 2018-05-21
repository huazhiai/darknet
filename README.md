# YOLO3训练自己的单类model
* YOLOv3：https://pjreddie.com/darknet/yolo/
* 说明书：https://pjreddie.com/media/files/papers/YOLOv3.pdf
* 今天Yolo Object Detector的作者发布了YOLOv3，看起来性能更好但速度有所牺牲。
* 大致使用方法见其他人对YOLOv2的blog，这里说一下训练单一class时的区别。
* 一般我们训练的class都是1类，所以在cfg/yolov3.cfg里：
*     把所有“classes=80”替换为1 (别忘了.data里也要改);
*     把所有“filters=255”替换为18;
* 原因：
* 在YOLOv2的【Region】上面那层的filters=num*(cls+1+4)，所以按原来的算法filters=5*(1+1+4)=30
* 但在YOLOv3里【region】替换成了3个【yolo】层，因其用了三个scale来预测bounding box，每个scale上预测三个box
* 尽管每个【yolo】里num=9，但它们上一层filters=num*(cls+1+4)里num实际是3，原因见上一行
* 例：默认yolov3.cfg里是Coco的80类，classes=80
* 所以每个【yolo】层上面的filter数=3*(80+1+4)=255
# YOLOv3: 训练自己的数据 
https://blog.csdn.net/lilai619/article/details/79695109
# my训练过程
* cd D:\darknet\scripts
* python voc_label.py
* type 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt >> train_voc.txt
* cd D:\darknet\build\darknet\x64
* my_tiny-yolo-voc.cfg::max_batches = 80200
* darknet.exe detector train data/my_voc.data my_tiny-yolo-voc.cfg my_tiny-yolo-voc_init.weights
* darknet.exe detector test  data/my_voc.data my_tiny-yolo-voc.cfg backup/my_tiny-yolo-voc_40200.weights data/dog.jpg
# Yolo-v3 and Yolo-v2 for Windows and Linux
### (neural network for object detection)

[![CircleCI](https://circleci.com/gh/AlexeyAB/darknet.svg?style=svg)](https://circleci.com/gh/AlexeyAB/darknet)

1. [How to use](#how-to-use)
2. [How to compile on Linux](#how-to-compile-on-linux)
3. [How to compile on Windows](#how-to-compile-on-windows)
4. [How to train (Pascal VOC Data)](#how-to-train-pascal-voc-data)
5. [How to train (to detect your custom objects)](#how-to-train-to-detect-your-custom-objects)
6. [When should I stop training](#when-should-i-stop-training)
7. [How to calculate mAP on PascalVOC 2007](#how-to-calculate-map-on-pascalvoc-2007)
8. [How to improve object detection](#how-to-improve-object-detection)
9. [How to mark bounded boxes of objects and create annotation files](#how-to-mark-bounded-boxes-of-objects-and-create-annotation-files)
10. [Using Yolo9000](#using-yolo9000)
11. [How to use Yolo as DLL](#how-to-use-yolo-as-dll)



|  ![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png) | &nbsp; ![map_fps](https://hsto.org/webt/pw/zd/0j/pwzd0jb9g7znt_dbsyw9qzbnvti.jpeg) mAP (AP50) https://pjreddie.com/media/files/papers/YOLOv3.pdf |
|---|---|

* Yolo v3 source chart for the RetinaNet on MS COCO got from Table 1 (e): https://arxiv.org/pdf/1708.02002.pdf
* Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
* Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg


# "You Only Look Once: Unified, Real-Time Object Detection (versions 2 & 3)"
A Yolo cross-platform Windows and Linux version (for object detection). Contributtors: https://github.com/pjreddie/darknet/graphs/contributors

This repository is forked from Linux-version: https://github.com/pjreddie/darknet

More details: http://pjreddie.com/darknet/yolo/

This repository supports:

* both Windows and Linux
* both OpenCV 2.x.x and OpenCV <= 3.4.0 (3.4.1 and higher isn't supported)
* both cuDNN v5-v7
* CUDA >= 7.5
* also create SO-library on Linux and DLL-library on Windows

##### Requires: 
* **Linux GCC>=4.9 or Windows MS Visual Studio 2015 (v140)**: https://go.microsoft.com/fwlink/?LinkId=532606&clcid=0x409  (or offline [ISO image](https://go.microsoft.com/fwlink/?LinkId=615448&clcid=0x409))
* **CUDA 9.1**: https://developer.nvidia.com/cuda-downloads
* **OpenCV 3.4.0**: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/opencv-3.4.0-vc14_vc15.exe/download
* **or OpenCV 2.4.13**: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.13/opencv-2.4.13.2-vc14.exe/download
  - OpenCV allows to show image or video detection in the window and store result to file that specified in command line `-out_filename res.avi`
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

##### Pre-trained models for different cfg-files can be downloaded from (smaller -> faster & lower quality):
* `yolov3.cfg` (236 MB COCO **Yolo v3**) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov3.weights
* `yolov3-tiny.cfg` (34 MB COCO **Yolo v3 tiny**) - requires 1 GB GPU-RAM:  https://pjreddie.com/media/files/yolov3-tiny.weights
* `yolov2.cfg` (194 MB COCO Yolo v2) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov2.weights
* `yolo-voc.cfg` (194 MB VOC Yolo v2) - requires 4 GB GPU-RAM: http://pjreddie.com/media/files/
