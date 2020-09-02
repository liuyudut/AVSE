## A Novel Baseline for Zero-shot Learning via Adversarial Visual-Semantic Embedding (BMVC 2020)
- Exploit a simple and effective baseline model for zero-shot learning.
- Perform embedding-to-image generation which visually exhibits the embeddings.
- Obtain consistent and promising improvements over previous baseline models.

![architecture](https://github.com/Liuy8/MUC/blob/master/MUC_overview.png)

## Dependencies

- PyTorch 
- Numpy
- sklearn
- scipy

## Experiment on CUB dataset

- Run ```python3 avse_main.py --dataset CUB --manualSeed 3483 --nclass_all 200 --lr 0.0001 --classifier_lr 0.001 --gamma 0.1 --nepoch 100 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --save_name cub``` 

## Experiment on SUN dataset

- Run ```python3 avse_main.py --dataset SUN --manualSeed 4115 --nclass_all 717 --lr 0.0002 --classifier_lr 0.0005 --gamma 0.1 --nepoch 100 --nz 102 --attSize 102 --resSize 2048 --syn_num 300 --save_name sun``` 

## Experiment on AWA2 dataset

- Run ```python3 avse_main.py --dataset AWA2 --manualSeed 9182 --nclass_all 50 --lr 0.00001 --classifier_lr 0.001 --gamma 0.1 --nepoch 50 --nz 85 --attSize 85 --resSize 2048 --syn_num 300 --save_name awa2``` 

## Experiment on Flower dataset

- Run ```python3 avse_main.py --dataset FLO --manualSeed 806 --nclass_all 102 --lr 0.0001 --classifier_lr 0.001 --gamma 0.1 --nepoch 100 --nz 1024 --attSize 1024 --resSize 2048 --syn_num 300 --save_name flower``` 

## Notes
- Some codes are based on the codebase of the [repository](https://github.com/hshustc/CVPR19_Incremental_Learning).
- More instructions will be provided later.

# Citation
Please cite the following paper if it is helpful for your research:
```
@InProceedings{AVSE_BMVC2020,
author = {Liu, Yu and Tuytelaars, Tinne}
title = {A Novel Baseline for Zero-shot Learning via Adversarial Visual-Semantic Embedding},
booktitle = {British Machine Vision Conference (BMVC)},
year = {2020}
}
```
