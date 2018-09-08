# Data preparation for training on ImageNet (ILSVRC2012)

### 1. Download data

You will need:
1. `ILSVRC2012_img_train.tar`
2. `ILSVRC2012_img_val.tar`
3. `ILSVRC2012_devkit_t12.tar.gz`
4. `ILSVRC2012_bbox_train_v2.tar.gz`

You can get them [here](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads) and [here](http://academictorrents.com/).  
It is ~145 GB in total.  
There are 1281167 train images and 50000 validation images.

### 2. Explore and preprocess
Run `explore_imagenet.ipynb` to see some images and bounding boxes.  
Also it removes bad boxes and creates metadata files.

### 3. Create `.tfrecords`
```
python create_tfrecords.py \
    --metadata_file=training_metadata.csv \
    --output=/mnt/datasets/imagenet/train_shards/ \
    --labels=integer_encoding.json \
    --boxes=boxes.npy \
    --num_shards=1000
```
It takes ~8 hours. Number of skipped images is 23.

```
python create_tfrecords.py \
    --metadata_file=validation_metadata.csv \
    --output=/mnt/datasets/imagenet/val_shards/ \
    --labels=integer_encoding.json \
    --num_shards=100
```
It takes ~12 minutes. Number of skipped images is 1.

### Final data size
There will be 1281144 train images and 49999 validation images.
