# Lafite
Code for paper [LAFITE: Towards Language-Free Training for Text-to-Image Generation](https://arxiv.org/abs/2111.13792) (CVPR 2022)

Update more details later.

## Requirements

The implementation is based on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [CLIP](https://github.com/openai/CLIP), the required packages can be found in the links.


## Preparing Datasets
Example:
```
python dataset_tool.py --source=./path_to_some_dataset/ --dest=./datasets/some_dataset.zip --width=256 --height=256 --transform=center-crop
```
the files at ./path_to_some_dataset/ should be like:

./path_to_some_dataset/

&ensp;&ensp;&boxvr;&nbsp; 1.png

&ensp;&ensp;&boxvr;&nbsp; 1.txt

&ensp;&ensp;&boxvr;&nbsp; 2.png

&ensp;&ensp;&boxvr;&nbsp; 2.txt

&ensp;&ensp;&boxvr;&nbsp; ...

We provide links to several commonly used datasets that we have already processed (with CLIP-ViT/B-32):

[MS-COCO Training Set](https://drive.google.com/file/d/1b82BCh65XxwR-TiA8zu__wwiEHLCgrw2/view?usp=sharing) 

[MS-COCO Validation Set](https://drive.google.com/file/d/1qBy5rPfo1go4d-PjF_Gu0kESCZ9Nt1Ta/view?usp=sharing)

[LN-COCO Training Set](https://drive.google.com/file/d/177Q_TGEXmIf_bk8j3bE_yAhr_3YrhLQY/view?usp=sharing)

[LN-COCO Testing Set](https://drive.google.com/file/d/12o2q2K7Ia6GTeqKL-g4x52t1Dv9lRrpK/view?usp=sharing)

[Multi-modal CelebA-HQ Training Set](https://drive.google.com/file/d/1TVpvwfi40Quk1oG1xvc8K2EQfb0koWN5/view?usp=sharing)

[Multi-modal CelebA-HQ Testing Set](https://drive.google.com/file/d/1FbsRLyqcQiwsyYENEvtP01-w9l1Hzpvl/view?usp=sharing)

[CUB Training Set](https://drive.google.com/file/d/1Hc3JZnHiDLpM6L2DuFuMTFTBXLgRB5DL/view?usp=sharing)

[CUB Testing Set](https://drive.google.com/file/d/1tzJQnwtAd7bhs0bLAzNGwCeC-DItUoKJ/view?usp=sharing)

## Training

These hyper-parameters are used for **MS-COCO**. Please tune **itd**, **itc** and **gamma** on different datasets, they might be **sensitive** to datasets.

Examples:

Training with ground-truth pairs
```
python train.py --gpus=4 --outdir=./outputs/ --temp=0.5 --itd=5 --itc=10 --gamma=10 --mirror=1 --data=./datasets/COCO2014_train_CLIP_ViTB32.zip --test_data=./datasets/COCO2014_val_CLIP_ViTB32.zip --mixing_prob=0.0
```

Training with language-free methods (pseudo image-text feature pairs)
```
python train.py --gpus=4 --outdir=./outputs/ --temp=0.5 --itd=10 --itc=10 --gamma=10 --mirror=1 --data=./datasets/COCO2014_train_CLIP_ViTB32.zip --test_data=./datasets/COCO2014_val_CLIP_ViTB32.zip --mixing_prob=1.0
```

## Pre-trained Models
Here we provide several pre-trained models (on google drive). 

[Model trained on MS-COCO, Language-free (Lafite-G), CLIP-ViT/B-32](https://drive.google.com/file/d/1eNkuZyleGJ3A3WXTCIGYXaPwJ6NH9LRA/view?usp=sharing)

[Model trained on MS-COCO, Language-free (Lafite-NN), CLIP-ViT/B-32](https://drive.google.com/file/d/1WQnlCM4pQZrw3u9ZeqjeUNqHuYfiDEU3/view?usp=sharing)

[Model trained on MS-COCO with Ground-truth Image-text Pairs, CLIP-ViT/B-32](https://drive.google.com/file/d/1tMD6MWydRDMaaM7iTOKsUK-Wv2YNDRRt/view?usp=sharing)

[Model trained on MS-COCO with Ground-truth Image-text Pairs, CLIP-ViT/B-16](https://drive.google.com/file/d/1sYlYmPE6MKwp_2NxquxdGyWV8htVfCGX/view?usp=sharing)

[Model Pre-trained On Google CC3M](https://drive.google.com/file/d/17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq/view?usp=sharing)

## Testing
Calculating metrics:

```
python calc_metrics.py --network=./some_pre-trained_models.pkl --metrics=fid50k_full,is50k --data=./training_data.zip --test_data=./testing_data.zip
```

To generate images with pre-trained models, you can use ./generate.ipynb. Also, you can try this [Colab notebook](https://colab.research.google.com/github/pollinations/hive/blob/main/interesting_notebooks/LAFITE_generate.ipynb) by @[voodoohop](https://github.com/voodoohop), in which the model pre-trained on CC3M is used.

To calculate SOA scores for MS-COCO, you can use ./generate_for_soa.py and [Semantic Object Accuracy for Generative Text-to-Image Synthesis](https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis)

## Citation
```
@article{zhou2021lafite,
  title={LAFITE: Towards Language-Free Training for Text-to-Image Generation},
  author={Zhou, Yufan and Zhang, Ruiyi and Chen, Changyou and Li, Chunyuan and Tensmeyer, Chris and Yu, Tong and Gu, Jiuxiang and Xu, Jinhui and Sun, Tong},
  journal={arXiv preprint arXiv:2111.13792},
  year={2021}
}
```

##
Please contact yufanzho@buffalo.edu if you have any question.
