# Data preparation

## Downloads
Training on ImageNet (ILSVRC2012).
You will need (http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads):
1. ILSVRC2012_img_train.tar
2. ILSVRC2012_img_val.tar
3. ILSVRC2012_devkit_t12.tar.gz
4. ILSVRC2012_bbox_train_v2.tar.gz


1281167


(~ 8 hours)
python create_tfrecords.py \
    --metadata_file=training_metadata.csv \
    --output=/mnt/datasets/imagenet/train_shards/ \
    --labels=integer_encoding.json \
    --boxes=boxes.npy \
    --num_shards=1000
    
Number of skipped images: 23

1281167 - 23 = 1281144

(~12 min)
python create_tfrecords.py \
    --metadata_file=validation_metadata.csv \
    --output=/mnt/datasets/imagenet/val_shards/ \
    --labels=integer_encoding.json \
    --num_shards=100
    
Number of skipped images: 1
50000