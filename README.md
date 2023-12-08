# Text Matching Improves Sequential Recommendation by Reducing Popularity Biases

Source code for our CIKM 2023 paper :
[Text Matching Improves Sequential Recommendation by Reducing Popularity Biases](https://arxiv.org/pdf/2308.14029.pdf)

Click the links below to view our papers, checkpoints and datasets

<a href='https://arxiv.org/abs/2308.14029'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a><a href='https://huggingface.co/OpenMatch/TASTE-beauty'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-beauty-blue'></a><a href='https://huggingface.co/OpenMatch/TASTE-sports'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-sports-blue'></a><a href='https://huggingface.co/OpenMatch/TASTE-toys'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-toys-blue'></a><a href='https://huggingface.co/OpenMatch/TASTE-yelp'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yelp-blue'></a><a href='https://drive.google.com/drive/folders/1U_tkCJq80kdefV9z_FdWDCMWvtUrm0or?usp=sharing'><img src='https://img.shields.io/badge/Google Drive-Dataset-yellow'></a> 

 If you find this work useful, please cite our paper  and give us a shining star 🌟

```
@article{liu2023text,
  title={Text Matching Improves Sequential Recommendation by Reducing Popularity Biases},
  author={Liu, Zhenghao and Mei, Sen and Xiong, Chenyan and Li, Xiaohua and Yu, Shi and Liu, Zhiyuan and Gu, Yu and Yu, Ge},
  journal={arXiv preprint arXiv:2308.14029},
  year={2023}
}
```

## 

## Requirements

**1. Install the following packages using Pip or Conda under this environment**

```
python>=3.8
transformers==4.22.2
numpy==1.23.5
datasets==2.11.0
faiss-cpu
scikit-learn>=1.1.2
pandas
tensorboard
```

We provide the version file `reproduce/environment.yml` of all our used packages, if you have any problems configuring the environment, please refer to this document.

**2. Install openmatch. To download OpenMatch as a library and obtain openmatch-thunlp-0.0.1.**


```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e.
```

We do not include all the requirements in the package. You may need to manually install `torch`

**3. Prepare the pretrained T5 weights**

TASTE is built on [t5-base](https://huggingface.co/t5-base/tree/main) model，

You can download the weights from this link and place them in `./pretrained_model/`
## Reproduce DSCI565 project
```bash
git clone https://github.com/yyg00/multi-modal-recommendation.git
cd multi-modal-recommendation
bash reproduce/dataprocess/gen_all_items.sh
bash reproduce/dataprocess/build_train.sh
bash reproduce/train/yelp/train_yelp.sh
bash reproduce/valid/yelp/eval_yelp.sh
bash reproduce/test/yelp/test_yelp.sh
```

## Acknowledgement

+ [OpenMatch](https://github.com/OpenMatch/OpenMatch) This repository is built upon OpenMatch! An all-in-one toolkit for information retrieval!
+ [FID](https://github.com/facebookresearch/FiD) The model architecture of TASTE refers to FID!
+ [RecBole](https://github.com/RUCAIBox/RecBole) Data processing and evaluation code based on RecBole! RecBole is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified, comprehensive and efficient framework for research purpose!

## Contact

If you have questions, suggestions, and bug reports, please email:
```
meisen@stumail.neu.edu.cn
```
