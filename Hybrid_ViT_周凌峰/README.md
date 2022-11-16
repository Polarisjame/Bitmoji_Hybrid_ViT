# Hybrid_ViT

My Hybrid_ViT model is re produced based on the paper [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).

---

# Hybrid ViT Model

该模型由ResNet34以及ViT组成

## ResNet34

模型参考[ResNet](./utils/ResidualNet.py)
原论文使用的是ResNet50，这里为了减少参数量用了ResNet34，也将最后两层残差层合并，在PatchEmbedding中用1*1的卷积核得到EmbeddingSize的通道。

## ViT

结构同原论文，最后只提取CLS Token的特征作为分类向量

---

## Requirements

My code works with the following environment.
* `python=3.7`
* `pytorch=1.12.1+cu116`
* tqdm
* numpy
* pandas

## Dataset

using [`Bitmojidata`](https://drive.google.com/file/d/1atMwmdOJe_fqG8Tyg5eqxZ-iDyPxDJOR/view?usp=sharing), Put all files under `./data/Bitmojidata`

## Training and Testing

You can run `python train.py` to train a model in cmd line and `python train.py -h` to get help.

You can usr `test(args, model, data, device)` in train.py to test the model

Here are some important parameters:

* `--batch_size`
* `--re_zero`: Use Re_zero in ResNet if True
* `--learning_rate`
* `--epochs`

## Results

model_epoch.pth and loss/acc figure is saved under `./checkpoint`,model is saved every 5 epochs.

