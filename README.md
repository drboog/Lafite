# Lafite
Code for paper [LAFITE: Towards Language-Free Training for Text-to-Image Generation](https://arxiv.org/abs/2111.13792) (CVPR 2022)


## Requirements

The implementation is based on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [CLIP](https://github.com/openai/CLIP), the required packages can be found in the links.


## Preparing datasets
Example:
```
python dataset_tool.py --source=./path_to_some_dataset/ --dest=./datasets/some_dataset.zip --width=256 --height=256 --transform=center-crop
```
the files at ./path_to_some_dataset/ should be like:
&ensp;&ensp;&boxvr;&nbsp; ./path_to_some_dataset/1.png

&ensp;&ensp;&boxvr;&nbsp; ./path_to_some_dataset/1.txt

&ensp;&ensp;&boxvr;&nbsp; ./path_to_some_dataset/2.png

&ensp;&ensp;&boxvr;&nbsp; ./path_to_some_dataset/2.txt

&ensp;&ensp;&boxvr;&nbsp; ...


## Training


## Pre-trained Models
Here we provide several pre-trained models (on google drive).

[Model trained on MS-COCO, Language-free (Lafite-G), CLIP-ViT/B-32](https://drive.google.com/file/d/1eNkuZyleGJ3A3WXTCIGYXaPwJ6NH9LRA/view?usp=sharing)

[Model trained on MS-COCO, Language-free (Lafite-NN), CLIP-ViT/B-32](https://drive.google.com/file/d/1WQnlCM4pQZrw3u9ZeqjeUNqHuYfiDEU3/view?usp=sharing)

[Model trained on MS-COCO with Ground-truth Image-text Pairs, CLIP-ViT/B-16](https://drive.google.com/file/d/1tMD6MWydRDMaaM7iTOKsUK-Wv2YNDRRt/view?usp=sharing)

[Model trained on MS-COCO with Ground-truth Image-text Pairs, CLIP-ViT/B-16](https://drive.google.com/file/d/17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq/view?usp=sharing)

[Model Pre-trained On Google CC3M](https://drive.google.com/file/d/17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq/view?usp=sharing)

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
