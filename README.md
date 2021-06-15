# MUST-GAN

### [Code](https://github.com/TianxiangMa/MUST-GAN) | [paper](https://arxiv.org/pdf/2011.09084.pdf)

The Pytorch implementation of our CVPR2021 paper "MUST-GAN: Multi-level Statistics Transfer for Self-driven Person Image Generation".
 
[Tianxiang Ma](https://tianxiangma.github.io/), Bo Peng, Wei Wang, Jing Dong,

CRIPAC,NLPR,CASIA & University of Chinese Academy of Sciences.


## Requirement
* python3
* pytorch 1.1.0
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate
* visdom


## Getting Started

### Installation

- Clone this repo:
```bash
git clone https://github.com/TianxiangMa/MUST-GAN.git
cd MUST-GAN
```

### Data Preperation
We train and test our model on [Deepfashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) dataset. Especially, we utilize [High-Res Images](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00?resourcekey=0-fsjVShvqXP2517KnwaZ0zw) in the In-shop Clothes Retrieval Benchmark.



Download this dataset and unzip (You will need to ask for password.) it, then put the folder **img_highres** under the `./datasets` directory. Download [train/test](https://drive.google.com/drive/folders/15xOoAJaVSMq09ln4NVcoeT_thJJ5vPio?usp=sharing) split list, which are used by a lot of methods, and put them under `./datasets` directory.
- Run the following code to split train/test dataset.
```bash
python tool/generate_fashion_datasets.py
```

Download [source-target paired images](https://drive.google.com/drive/folders/1DLnjVsts1xNPbPGHPCth97--SPgVAdyy?usp=sharing) list, as same as the list used by many previous work.
Becouse our method can self-supervised training, we **do not** need the **fashion-resize-pairs-train.csv**, you can download [**train_images_lst.csv**](https://drive.google.com/drive/folders/1DLnjVsts1xNPbPGHPCth97--SPgVAdyy?usp=sharing) for training.

Download [train/test keypoints annotation](https://drive.google.com/drive/folders/1cIxnfS7loVhj8cbv8dELMI68AZuJDW-J?usp=sharing) files and [semantic segmentation](https://drive.google.com/drive/folders/1c_rJtaAVY6cUAvFGoBicNzETPwlR_q8d?usp=sharing) files.

Put all the above files into the `./datastes` folder.

- Run the following code to generate pose map and pose connection map.
```bash
python tool/generate_pose_map.py
python tool/generate_pose_connection_map.py
```

### Testing
Download our [pretrained model](https://drive.google.com/drive/folders/1NQM3LxvD0RPgrNdwL474keByKOtPezFh?usp=sharing), and put it into `./check_points/MUST-GAN/` folder.

Run the following code, and set the parameters along your need.
```bash
bash scripts/test.sh
```

### Training
Run the following code, and set the parameters along your need.
```bash
bash scripts/train.sh
```

## Citation
If you use this code for your research, please cite our paper:

```
@article{ma2020must,
  title={MUST-GAN: Multi-level Statistics Transfer for Self-driven Person Image Generation},
  author={Ma, Tianxiang and Peng, Bo and Wang, Wei and Dong, Jing},
  journal={arXiv preprint arXiv:2011.09084},
  year={2020}
}
```


## Acknowledgments
Our code is based on [PATN](https://github.com/tengteng95/Pose-Transfer) and [ADGAN](https://github.com/menyifang/ADGAN), thanks for their great work.

