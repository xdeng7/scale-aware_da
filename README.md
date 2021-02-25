
# Scale Aware Adaptation for Land-Cover Classificationin Remote Sensing Imagery (in PyTorch)

This repository provides the ResNet-101-based segmentation model trained on ISPRS dataset from the paper `Scale Aware Adaptation for Land-Cover Classificationin Remote Sensing Imagery` (the provided weights achieve **47.66**% mean IoU on the validation set for Potsdam to Vaihingen)



** Train the model
```bash
$ python train_scaleDA_isprs.py --folder DATA-ROOT
```

** Test the model
```bash
$ python test_isprs.py --resume CHECKPOINT 
```

Please cite our paper:
```
@inproceedings{deng2021scale,
  title={Scale Aware Adaptation for Land-Cover Classification in Remote Sensing Imagery},
  author={Deng, Xueqing and Zhu, Yi and Tian, Yuxin and Newsam, Shawn},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2160--2169},
  year={2021}
}
```
