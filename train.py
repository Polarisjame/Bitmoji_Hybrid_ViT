from time import sleep

import pandas as pd
import torch
# from torch.nn import CrossEntropyLoss
# from torch.optim import Adam, lr_scheduler

import numpy as np
import random
# from CatDog.models.VIT import VIT
import argparse
# from train import train
# from CatDog.data.dataload import load_data
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler, SGD
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from models.VIT import VIT
from utils.ResidualNet import Res34, CNN
from data import VITSet

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size

def evaluate(model, data, lossF, device):
    val_targ = []
    val_pred = []
    val_loss = 0
    with torch.no_grad():
        for (data_x, data_y) in data:
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            logits = model(data_x)
            loss = lossF(logits, data_y)
            val_loss += loss.item()
            _, indices = torch.max(logits.to('cpu'), dim=1)
            val_pred.extend(indices)
            val_targ.extend(data_y.to('cpu'))
        val_loss /= len(data)
        # print(val_pred,val_targ)
        val_f1 = f1_score(val_targ, val_pred)
        val_recall = recall_score(val_targ, val_pred)
        val_precision = precision_score(val_targ, val_pred)
        # val_acc = binary_accuracy(val_targ, val_pred)
    return val_loss, val_f1, val_recall, val_precision


def train(args, model, data, device):
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    lossF = CrossEntropyLoss()
    # state_dict = torch.load(r'./checkpoint/vit_model_epoch46.pth')
    # model.load_state_dict(state_dict)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_f1_list = []
    # optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_decacy)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        with tqdm(total=len(train_loader)) as t:
            for (data_x, data_y) in train_loader:
                # print(data_y)
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                t.set_description('Epoch {%d}' % epoch)
                cur_lr = optimizer.param_groups[0]['lr']
                model.train()
                optimizer.zero_grad()
                out = model(data_x)
                loss = lossF(out, data_y)
                with torch.no_grad():
                    _, indices = torch.max(out.to('cpu'), dim=1)
                    # print(indices)
                    correct = torch.sum(indices == data_y.to('cpu'))
                    train_acc = correct.item() * 1.0 / len(data_y)
                train_loss_list.append(loss.item())
                train_acc_list.append(train_acc)
                t.set_postfix(train_loss="{:.5f}".format(loss.item()), lr="{:.3e}".format(cur_lr), train_Acc=train_acc)
                loss.backward()
                optimizer.step()
                sleep(0.1)
                t.update(1)
                scheduler.step()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            model.eval()
            val_loss, F1, R, P = evaluate(model, val_loader, lossF, device)
            val_loss_list.append(val_loss)
            val_f1_list.append(F1)
            val_acc_list.append(P)
            print(
                '\nEpoch:{:d},loss:{:.4f},Precission:{:.4f},Recall:{:.4f},F1:{:.4f}'.format(epoch, val_loss, P,
                                                                                            R, F1))
            # if epoch % 5 == 0 or epoch >= 43:
            #     torch.save(model.state_dict(), "./checkpoint/res_model_epoch{:d}.pth".format(epoch))
            file_save_name = "./checkpoint/cnn_model_{:d}".format(epoch)
            train_list = list(range(len(train_loss_list)))
            plt.subplot(211)
            plt.plot(train_list, train_loss_list)
            plt.title('train_loss')
            plt.subplot(212)
            plt.plot(train_list, train_acc_list)
            plt.savefig(file_save_name + ".jpg")
    val_list = list(range(len(val_loss_list)))
    plt.subplot(221)
    plt.plot(val_list, val_loss_list)
    plt.title('val_loss')
    plt.subplot(222)
    plt.plot(val_list, val_acc_list)
    plt.title('val_acc')
    plt.subplot(223)
    plt.plot(val_list, val_f1_list)
    plt.title('val_f1')
    plt.plot(train_list, train_acc_list)
    plt.savefig(file_save_name + "_val.jpg")
            # if epoch % 5 == 0:
            #     torch.save(model.state_dict(), "./checkpoint/res_model_epoch{:d}.pth".format(epoch))


# def test(args, model, datas, device):
#     state_dict = torch.load(r'./checkpoint/vit_model_epoch44.pth')
#     model.load_state_dict(state_dict)
#     test_loader = datas.test_dataloader()
#
#     with tqdm(total=len(test_loader)) as t:
#         model.to(device)
#         model.eval()
#         result = []
#         for (data_x, _) in test_loader:
#             # print(data_y)
#             data_x = data_x.to(device)
#             out = model(data_x)
#             _, indices = torch.max(out.to('cpu'), dim=1)
#             result.extend(indices.data.cpu().numpy())
#             t.update(1)
#     # print(result)
#     result = [np.array(1) if a == 1 else -1 for a in result]
#     ind = list(range(3000, 4084))
#     ind = [str(a)+'.jpg' for a in ind]
#     data = {'image_id': ind, 'is_male': result}
#     df = pd.DataFrame(data)
#     df.to_csv(r'./out_44.csv', index=False)


def main(args):
    data = VITSet(args)
    data.setup()
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    if args.use_only_res34:
        args.use_hybrid = False
        model = Res34(args, 3, 2).to(device)
    else:
        model = VIT(args).to(device)
    # print(model)
    print(getModelSize(model))
    # model = CNN(args, 3, 2).to(device)
    train(args, model, data, device)
    # test(args, model, data, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIT")
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='input batch size for training (default: 32)')
    parser.add_argument('-vr', '--valid_ratio', type=float, default=0.05, help='divide valid:train sets by train_ratio')
    parser.add_argument('-cuda', '--use_cuda', type=bool, default=True, help='Use cuda or not')
    parser.add_argument('-k', '--topk', type=int, default=5, help='save the topk model')
    parser.add_argument('-nu', '--num_workers', type=int, default=0, help='thread number')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='thread number')
    parser.add_argument('--re_zero', type=bool, default=True)
    parser.add_argument('--l2_decacy', type=float, default=1e-4)
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='input training epoch for training (default: 50)')
    parser.add_argument('--use_hybrid', type=bool, default=True,
                        help='set True if use Hybrid_Model else use only ViT Model(default: True)')
    parser.add_argument('--use_only_res34', type=bool, default=False,
                        help='set True if use ResNet34 else use  ViT Model(default: False)')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--resnet_out_channels', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='input learning rate for training (default: 1e-4)')
    args = parser.parse_args()
    main(args)
