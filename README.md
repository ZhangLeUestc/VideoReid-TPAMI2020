# Ordered or Orderless: A Revisit for Video based Person Re-Identification

This repository contains the Pytorch implementation of the Ensemble_CRF methods in the following paper:

[Ordered or Orderless: A Revisit for Video based Person Re-Identification](https://arxiv.org/abs/1912.11236)  

[*Le Zhang*](https://zhangleuestc.github.io/), Joey Tianyi Zhou, Ming-Ming Cheng, Yun Liu, Jia-Wang Bian, Zeng Zeng, Chunhua Shen

IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.

 <div align=center><img src="https://github.com/ZhangLeUestc/VideoReid-TPAMI2020/blob/master/image.jpg"  /></div>

Our experiment is mainly based on the following paper: 

Chen, Dapeng, et al. "Group consistent similarity learning via deep crf for person re-identification." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

## Requirements
- [python 3.6](We recommend to use Anaconda, since many python libs like numpy and sklearn are needed in our code.)
- [PyTorch and torchvision](https://pytorch.org/) (we run the code under version 0.4.0, maybe versions >=1.0 also work.)  
- [metric-learn 0.3.0](https://pypi.org/project/metric-learn/0.3.0/)

## Dataset Downloads
Please Download the [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) Dataset firstly.

## Training Example
CUDA_VISIBLE_DEVICES=2,3,4 python main_mars.py -b 132 -d mars --epoch 100  --instances_num 6 --cnnlr 0.5e-2  --logs-dir logs/mars/

## Testing Example

CUDA_VISIBLE_DEVICES=2,3,4 python test_mars.py 

## Pretrained models:

We also provide the pretrained models in [Google Drive](https://zhangleuestc.github.io/)

## Notes

1. Although the KemenyYoung method used in the paper is theotically tidy, it is somehow time consuing. Motivated by the [TSN](https://github.com/yjxiong/temporal-segment-networks) framework, we pool the visual features from a set of sparsely sampled frames and then re-identify based on the Euclidean distance amongest all the gallery videos. This version performs on par with the KemenyYoung methods but is much faster. However, if you are still interested in the implementation of the KemenyYoung method, please feel free to contact me.

2. As Mars is the largest video re-id datasets, we sparsely sample several frames from all tracklet during training to speed up the training process. 

3. One may modify the sampling strategy used in the training and testing part to improve our results. 

## Citations
Please cite the following papers if you use this repository in your research work:
```sh

@article{Zhang2020OrderlessReID,
    author  = {Le Zhang and
               Zenglin Shi and
               Joey Tianyi Zhou and
               Ming-Ming Cheng and
               Yun Liu and
               Jia-Wang Bian and Zeng Zeng and Chunhua Shen},
    title   = {Ordered or Orderless: A Revisit for Video based Person Re-Identification},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year    = {2020},
    eprint  = {1912.11236},
    url     = {},
    venue   = {TPAMI},
}
 

```

```

Contact **Le Zhang**(zhangleuestc@gmail.com) for questions, comments and reporting bugs.



