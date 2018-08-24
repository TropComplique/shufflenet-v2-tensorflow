# ShuffleNet v2
This is an implementation of [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
](https://arxiv.org/abs/1807.11164).

| model | accuracy | top 5 accuracy |
| --- | --- | --- |
| 0.5x | 0.608 | 0.822 |
| 1.0x | 0.689 | 0.885 |

You can download trained checkpoints from [here](https://drive.google.com/drive/folders/1KGIdE8SmR-Af9zheuQ68EhD0ck4h4riH?usp=sharing).

## Credit
The training code is heavily inspired by:  
1. https://github.com/tensorflow/models/tree/master/official/resnet
2. https://cloud.google.com/tpu/docs/inception-v3-advanced

## Other implementations
1. [miaow1988/ShuffleNet_V2_pytorch_caffe](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe)
2. [tensorpack/examples/ImageNetModels](https://github.com/tensorpack/tensorpack/tree/master/examples/ImageNetModels#shufflenet)
