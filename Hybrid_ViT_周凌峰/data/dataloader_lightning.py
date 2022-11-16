from torch.utils.data import DataLoader
from data.dataload import DogCat, Bitmoji


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

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = Bitmoji(root=r'./data/Bitmojidata', train=True)
            self.valset = Bitmoji(root=r'./data/Bitmojidata', train=False)
            self.testset = Bitmoji(root=r'./data/Bitmojidata', test=True)
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
