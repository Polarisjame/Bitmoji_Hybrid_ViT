import os
import torchvision.transforms as T
from PIL import Image
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader


class Bitmoji(data.Dataset):

    def __init__(self, root, args, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        train_imgs = root + r'/trainimages'
        test_imgs = root + r'/testimages'
        if test:
            imgs = [os.path.join(test_imgs, img) for img in os.listdir(test_imgs)]
        else:
            imgs = [os.path.join(train_imgs, img) for img in os.listdir(train_imgs)]

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int((1-args.valid_ratio) * imgs_num)]
            # self.imgs = imgs[:100]
        else:
            self.imgs = imgs[int(args.valid_ratio * imgs_num):]

        train_file = root + '/train.csv'
        df = pd.read_csv(train_file)
        self.values = df.values

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(90),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        # print(img_path)
        if self.test:
            label = 0
        else:
            index = int(img_path.split('\\')[-1][0:-4])
            value = self.values[index][1]
            # print(value)
            label = 0 if value == -1 else 1
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


class VITSet():
    def __init__(self, args):
        super().__init__()
        # self.save_hyperparameters(args)
        self.valset = None
        self.trainset = None
        self.batch_size = args.batch_size
        self.valid_ratio = args.valid_ratio
        self.num_workers = args.num_workers
        self.n_test = None
        self.args = args

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = Bitmoji(root=r'./data/Bitmojidata', train=True, args=self.args)
            self.valset = Bitmoji(root=r'./data/Bitmojidata', train=False, args=self.args)
            self.testset = Bitmoji(root=r'./data/Bitmojidata', test=True, args=self.args)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # for data in dataloader:
        #     input, label = data
        #     print(label.data)
        return dataloader

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
