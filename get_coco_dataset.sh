#! /bin/bash

## coco dataset 2014

# Clone COCO API
git clone https://github.com/cocodataset/cocoapi
cd cocoapi

# Download Images
mkdir images
cd images
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
# wget http://images.cocodataset.org/zips/test2014.zip
# wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
