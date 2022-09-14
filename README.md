# VideoClassification

视频分类分布式训练代码

## 视频数据准备
- 需要 opencv cv2.VideoCapture 能够读取的视频文件
- 结构如下
```
----|videodataset:
     ----|train:
          ----|class_a:
               ----|xxx.avi, yyy.mp4 ...
          ----|class_b:
               ____|nnn.avi, mmm.mp4 ...
     ----|val:
          ----|class_a:
               ----|vvv.avi, zzz.mp4 ...
          ----|class_b:
               ____|www.avi, uuu.mp4 ...
```
- 需要修改configs下训练所对应的xxx_config.yaml文件

## 训练方法
```
python main.py --config="configs/dist_slowfast_train_config.yaml"
```