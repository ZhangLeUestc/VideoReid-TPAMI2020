# Ordered or Orderless: A Revisit for Video based Person Re-Identification

This repository contains the Pytorch implementation of the Ensemble_CRF methods in the following paper:

[Ordered or Orderless: A Revisit for Video based Person Re-Identification](https://arxiv.org/abs/1912.11236)  
[*Le Zhang*](https://zhangleuestc.github.io/), Joey Tianyi Zhou, Ming-Ming Cheng, Yun Liu, Jia-Wang Bian, Zeng Zeng, Chunhua Shen
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.

Our experiment is mainly based on the following paper: 
Chen, Dapeng, et al. "Group consistent similarity learning via deep crf for person re-identification." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

## Requirements
- [python 3.6](We recommend to use Anaconda, since many python libs like numpy and sklearn are needed in our code.)
We use the MTCNN to first detect and align the faces. We used two customized layers which may not be included in the official caffe.
- [PyTorch and torchvision](https://pytorch.org/) (we run the code under version 0.4.0, maybe versions >=1.0 also work.)  
-[metric-learn 0.3.0](https://pypi.org/project/metric-learn/0.3.0/)

## Dataset Downloads
Please Download the [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) Dataset firstly.

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

Contact **Le Zhang** [:envelope:](mailto:zhangleuestc@gmail.com) for questions, comments and reporting bugs.



