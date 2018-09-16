# ShuffleNet v2
This is an implementation of [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
](https://arxiv.org/abs/1807.11164).

| model | accuracy | top 5 accuracy |
| --- | --- | --- |
| 0.5x | 0.607 | 0.822 |
| 1.0x | 0.688 | 0.886 |

You can download ImageNet trained checkpoints from [here](https://drive.google.com/drive/folders/1KGIdE8SmR-Af9zheuQ68EhD0ck4h4riH?usp=sharing).

## How to use the pretrained models
You only need two things:
1. File `architecture.py`. It contains a definition of the graph.
2. Checkpoint. You can load it into the graph using `tf.train.Saver` or `tf.train.init_from_checkpoint`.

For an example of using the pretrained model see: `inference_with_trained_model.ipynb`.

## Speed benchmarks

| model | accuracy | images/second |
| --- | --- | --- |
| ShuffleNet v2 0.5x | 0.607 | 3192 |
| ShuffleNet v2 1.0x | 0.689 | 2349 |
| ShuffleNet v2 1.5x | - | 1855 |
| ShuffleNet v2 2.0x | - | 1570 |
| MobileNet v1 0.5x | 0.633 | 3317 |
| MobileNet v1 0.75x | 0.684 | 2187 |
| MobileNet v1 1.0x | 0.709 | 1685 |
| MobileNet v2 0.35x | 0.603 | 2722 |
| MobileNet v2 0.75x | 0.698 | 1527 |
| MobileNet v2 1.0x | 0.718 | 1292 |

All measurements were done using batches of size 8, images of size 224x224, and NVIDIA GTX 1080 Ti.  
See `benchmark_speed.ipynb` for the code.

MobileNet v1 results are taken from [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md). MobileNet v2 results are taken from [here](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

## Notes
1. Using moving averages of weights doesn't increase accuracy for some reason.

## Requirements
1. for using the pretrained models: `tensorflow 1.10`
2. for dataset preparation: `pandas, Pillow, tqdm, opencv, ...`

## How to train
1. Prepare ImageNet. See `data/README.md`.
2. Set the right parameters in the beginning of `train.py` file.
3. Run `python train.py`.
4. Run `tensorboard` to see the loss curves. Examples of loss curves are in `images/`.
5. Use `evaluation.ipynb` for the final evaluation on ImageNet.

## Credit
The training code is heavily inspired by:  
1. https://github.com/tensorflow/models/tree/master/official/resnet
2. https://cloud.google.com/tpu/docs/inception-v3-advanced

## Other implementations
1. [miaow1988/ShuffleNet_V2_pytorch_caffe](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe)
2. [tensorpack/examples/ImageNetModels](https://github.com/tensorpack/tensorpack/tree/master/examples/ImageNetModels#shufflenet)
