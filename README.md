# ShuffleNet v2
This is an implementation of [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
](https://arxiv.org/abs/1807.11164).

| model | accuracy | top 5 accuracy |
| --- | --- | --- |
| 0.5x | 0.608 | 0.822 |
| 1.0x | 0.689 | 0.885 |

You can download trained checkpoints from [here](https://drive.google.com/drive/folders/1KGIdE8SmR-Af9zheuQ68EhD0ck4h4riH?usp=sharing).

## How to use the pretrained models
You only need two things:
1. File `architecture.py`. It contains a definition of the graph.
2. Checkpoint. You can load it into the graph using `tf.train.Saver` or `tf.train.init_from_checkpoint`.

For an example of using the pretrained model see: `inference_with_trained_model.ipynb`.

## Requirements
1. for using the pretrained models: `tensorflow 1.10`
2. for dataset preparation: `pandas, Pillow, tqdm, opencv, ...`

## How to train
1. Prepare ImageNet. See `data/README.md`.
2. Set the right parameters in the beginning of `train.py` file.
2. Run `python train.py`.

## Credit
The training code is heavily inspired by:  
1. https://github.com/tensorflow/models/tree/master/official/resnet
2. https://cloud.google.com/tpu/docs/inception-v3-advanced

## Other implementations
1. [miaow1988/ShuffleNet_V2_pytorch_caffe](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe)
2. [tensorpack/examples/ImageNetModels](https://github.com/tensorpack/tensorpack/tree/master/examples/ImageNetModels#shufflenet)
