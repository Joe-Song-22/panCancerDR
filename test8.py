import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset
from model_test import *
import torch.nn as nn
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
device = torch.device("cuda:1")
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from model_upgrade import *
# 使用SMOTE进行采样
smote = SMOTE(random_state=42)



# # 使用RandomUnderSampler进行过采样
# ros = RandomOverSampler(random_state=42)
# source_x_resampled, source_y_resampled = ros.fit_resample(source_x, source_y)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


source_data1 = pd.read_csv('/data/sr/updated_output7cetuximab.csv')
target_data1 = pd.read_csv('/data/sr/Target_expr_resp_z.Cetuximab_tp4k.csv')
source_data2 = pd.read_csv('/data/sr/updated_output2.csv')
target_data2 = pd.read_csv('/data/sr/Target_expr_resp_z.PLX4720_451Lu_tp4k.csv')
#Target_data = pd.read_csv('/data/sr/Target_expr_resp_z.Etoposide_tp4k.tsv', sep='\t')
#source2_data = pd.read_excel('/data/sr/Etoposide_tp4k_2.xlsx')

#print(source3_data)

x_1 = source_data1[source_data1['label'] == 1].iloc[:, 5:].values
x_2 = source_data1[source_data1['label'] == 2].iloc[:, 5:].values
x_3 = source_data1[source_data1['label'] == 3].iloc[:, 5:].values
x_4 = source_data1[source_data1['label'] == 4].iloc[:, 5:].values
x_5 = source_data1[source_data1['label'] == 5].iloc[:, 5:].values
x_6 = source_data1[source_data1['label'] == 6].iloc[:, 5:].values
x_7 = source_data1[source_data1['label'] == 7].iloc[:, 5:].values
x_8 = source_data1[source_data1['label'] == 8].iloc[:, 5:].values
x_9 = source_data1[source_data1['label'] == 9].iloc[:, 5:].values
x_10 = source_data1[source_data1['label'] == 10].iloc[:, 5:].values
x_11 = source_data1[source_data1['label'] == 11].iloc[:, 5:].values
x_12 = source_data1[source_data1['label'] == 12].iloc[:, 5:].values
x_13 = source_data1[source_data1['label'] == 13].iloc[:, 5:].values
x_14 = source_data1[source_data1['label'] == 14].iloc[:, 5:].values
x_15 = source_data1[source_data1['label'] == 15].iloc[:, 5:].values
x_16 = source_data1[source_data1['label'] == 16].iloc[:, 5:].values
x_17 = source_data1[source_data1['label'] == 17].iloc[:, 5:].values
x_18 = source_data1[source_data1['label'] == 18].iloc[:, 5:].values
x_19 = source_data1[source_data1['label'] == 19].iloc[:, 5:].values
x_20 = source_data1[source_data1['label'] == 20].iloc[:, 5:].values
x_21 = source_data1[source_data1['label'] == 21].iloc[:, 5:].values
x_22 = source_data1[source_data1['label'] == 22].iloc[:, 5:].values
x_23 = source_data1[source_data1['label'] == 23].iloc[:, 5:].values
t_1 = target_data1[target_data1['label'] == 23].iloc[:, 3:].values
r_1 = source_data1[source_data1['label'] == 1].iloc[:, 1].values
r_2 = source_data1[source_data1['label'] == 2].iloc[:, 1].values
r_3 = source_data1[source_data1['label'] == 3].iloc[:, 1].values
r_4 = source_data1[source_data1['label'] == 4].iloc[:, 1].values
r_5 = source_data1[source_data1['label'] == 5].iloc[:, 1].values
r_6 = source_data1[source_data1['label'] == 6].iloc[:, 1].values
r_7 = source_data1[source_data1['label'] == 7].iloc[:, 1].values
r_8 = source_data1[source_data1['label'] == 8].iloc[:, 1].values
r_9 = source_data1[source_data1['label'] == 9].iloc[:, 1].values
r_10 = source_data1[source_data1['label'] == 10].iloc[:, 1].values
r_11 = source_data1[source_data1['label'] == 11].iloc[:, 1].values
r_12 = source_data1[source_data1['label'] == 12].iloc[:, 1].values
r_13 = source_data1[source_data1['label'] == 13].iloc[:, 1].values
r_14 = source_data1[source_data1['label'] == 14].iloc[:, 1].values
r_15 = source_data1[source_data1['label'] == 15].iloc[:, 1].values
r_16 = source_data1[source_data1['label'] == 16].iloc[:, 1].values
r_17 = source_data1[source_data1['label'] == 17].iloc[:, 1].values
r_18 = source_data1[source_data1['label'] == 18].iloc[:, 1].values
r_19 = source_data1[source_data1['label'] == 19].iloc[:, 1].values
r_20 = source_data1[source_data1['label'] == 20].iloc[:, 1].values
r_21 = source_data1[source_data1['label'] == 21].iloc[:, 1].values
r_22 = source_data1[source_data1['label'] == 22].iloc[:, 1].values
r_23 = source_data1[source_data1['label'] == 23].iloc[:, 1].values
t_r = target_data1[target_data1['label'] == 23].iloc[:, 1].values

x2_1 = source_data2[source_data2['label'] == 1].iloc[:, 5].values
x2_2 = source_data2[source_data2['label'] == 2].iloc[:, 5].values
x2_3 = source_data2[source_data2['label'] == 3].iloc[:, 5].values
x2_4 = source_data2[source_data2['label'] == 4].iloc[:, 5].values
x2_5 = source_data2[source_data2['label'] == 5].iloc[:, 5].values
x2_6 = source_data2[source_data2['label'] == 6].iloc[:, 5].values
x2_7 = source_data2[source_data2['label'] == 7].iloc[:, 5].values
x2_8 = source_data2[source_data2['label'] == 8].iloc[:, 5].values
x2_9 = source_data2[source_data2['label'] == 9].iloc[:, 5].values
x2_10 = source_data2[source_data2['label'] == 10].iloc[:, 5].values
t2_1 = target_data2[target_data2['label'] == 11].iloc[:, 3].values
r2_1 = source_data2[source_data2['label'] == 1].iloc[:, 1].values
r2_2 = source_data2[source_data2['label'] == 2].iloc[:, 1].values
r2_3 = source_data2[source_data2['label'] == 3].iloc[:, 1].values
r2_4 = source_data2[source_data2['label'] == 4].iloc[:, 1].values
r2_5 = source_data2[source_data2['label'] == 5].iloc[:, 1].values
r2_6 = source_data2[source_data2['label'] == 6].iloc[:, 1].values
r2_7 = source_data2[source_data2['label'] == 7].iloc[:, 1].values
r2_8 = source_data2[source_data2['label'] == 8].iloc[:, 1].values
r2_9 = source_data2[source_data2['label'] == 9].iloc[:, 1].values
r2_10 = source_data2[source_data2['label'] == 10].iloc[:, 1].values
t2_r = target_data2[target_data2['label'] == 11].iloc[:, 3].values


#x_4 = np.concatenate((x_3, x_4), axis = 0)
#r_4 = np.concatenate((r_3, r_4), axis = 0)
# x_1, r_1 = smote.fit_resample(x_1, r_1)
#x_2, r_2 = smote.fit_resample(x_2, r_2)
# x_3, r_3 = smote.fit_resample(x_3, r_3)
# x_4, r_4 = smote.fit_resample(x_4, r_4)
# x_5, r_5 = smote.fit_resample(x_5, r_5)
# x_6, r_6 = smote.fit_resample(x_6, r_6)
# x_7, r_7 = smote.fit_resample(x_7, r_7)
# x_8, r_8 = smote.fit_resample(x_8, r_8)
# print(x_8, r_8)
# x_9, r_9 = smote.fit_resample(x_9, r_9)
# x_10, r_10 = smote.fit_resample(x_10, r_10)
#x_12 = np.concatenate((x_11, x_12, x_13, x_14, x_15), axis=0)
#r_12 = np.concatenate((r_11, r_12, r_13, r_14, r_15), axis=0)
#x_12, r_12 = smote.fit_resample(x_12, r_12)
#x_23, r_23 = smote.fit_resample(x_23, r_23)
#x_11, r_11 = smote.fit_resample(x_11, r_11)
#x_12, r_12 = smote.fit_resample(x_12, r_12)
#x_13, r_13 = smote.fit_resample(x_13, r_13)
# x_14, r_14 = smote.fit_resample(x_14, r_14)
# x_15, r_15 = smote.fit_resample(x_15, r_15)
# x_16, r_16 = smote.fit_resample(x_16, r_16)
# x_17, r_17 = smote.fit_resample(x_17, r_17)
# x_18, r_18 = smote.fit_resample(x_18, r_18)
# x_19, r_19 = smote.fit_resample(x_19, r_19)
# x_20, r_20 = smote.fit_resample(x_20, r_20)
# x_21, r_21 = smote.fit_resample(x_21, r_21)
# x_22, r_22 = smote.fit_resample(x_22, r_22)
# t_1, t_r = smote.fit_resample(t_1, t_r)
#x1_label = torch.cat((torch.ones((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1))), dim=1)
#x1_1label = torch.cat((torch.ones((x1_1.shape[0],1)),torch.zeros((x1_1.shape[0],1))),dim=1)
#x2_0 = Target_data[Target_data['response'] == 0].iloc[:, 2:].values
#x2_0label = torch.cat((torch.zeros((x2_0.shape[0],1)),torch.ones((x2_0.shape[0],1))),dim=1)
#x2_1 = Target_data[Target_data['response'] == 1].iloc[:, 2:].values
#x2_1label = torch.cat((torch.zeros((x2_1.shape[0],1)),torch.ones((x2_1.shape[0],1))),dim=1)
# x3_0 = source3_data[source3_data['response'] == 0].iloc[:, 2:].values
# x3_0label = torch.cat((torch.zeros((x3_0.shape[0],1)),torch.zeros((x3_0.shape[0],1)),torch.ones((x3_0.shape[0],1))),dim=1)
# x3_1 = source3_data[source3_data['response'] == 1].iloc[:, 2:].values
# x3_1label = torch.cat((torch.zeros((x3_1.shape[0],1)),torch.zeros((x3_1.shape[0],1)),torch.ones((x3_1.shape[0],1))),dim=1)
result_list = []
x = {}
x_ = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_9, t_1]
# 循环从 x1 到 x11
for i in range(8):
    # 初始化一个全零张量
    zeros_tensor = torch.zeros((x_[i].shape[0], 9))
    # 在第 i 列插入全一张量
    zeros_tensor[:, i] = 1
    # 将生成的张量加入到结果列表中
    x_[i] = zeros_tensor
zeros_tensor = torch.zeros((x_[8].shape[0], 9))
zeros_tensor[:, 8] = 1
x_[8] = zeros_tensor
t_ = [x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8, x2_9, x2_10, t2_1]
for i in range(10):
    zeros_tensor = torch.zeros((t_[i].shape[0], 11))
    zeros_tensor[:, i] = 1
    t_[i] = zeros_tensor
zeros_tensor = torch.zeros((t_[10].shape[0], 11))
zeros_tensor[:, 10] = 1
t_[10] = zeros_tensor
# print(x['x3_label'])
# 将列表中的张量连接起来，沿着列的方向
# print(x['x1_label'])


s1 = np.concatenate((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_9), axis=0)
s1_2 = np.concatenate((x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8, x2_9, x2_10), axis=0)
l1 = torch.cat((x_[0], x_[1], x_[2], x_[3], x_[4], x_[5], x_[6], x_[7]), dim=0)
l1_2 = torch.cat((t_[0], t_[1], t_[2], t_[3], t_[4], t_[5], t_[6], t_[7], t_[8], t_[9]), dim=0)
s2 = np.concatenate((t_1,), axis=0)
s2_2 = np.concatenate((t2_1,), axis=0)
l2 = torch.cat((x_[8],), dim=0)
l2_2 = torch.cat((t_[10],), dim=0)
# t1 = np.concatenate((x2_0,), axis=0)
# t2 = np.concatenate((x2_1,), axis=0)
# tl1 = torch.cat((x2_0label,), dim=0)
# tl2 = torch.cat((x2_1label,), dim=0)
# pca = PCA(n_components=800)  # 选择要降到的目标维度
# pca_s1 = pca.fit_transform(s1)
# pca_s2 = pca.fit_transform(s2)
bs = 128
# y1_0 = source_data[source_data['response'] == 0].iloc[:, 1:2].values
# y1_1 = source_data[source_data['response'] == 1].iloc[:, 1:2].values
# y2_0 = Target_data[Target_data['response'] == 0].iloc[:, 1:2].values
# y2_1 = Target_data[Target_data['response'] == 1].iloc[:, 1:2].values
#y3_0 = source3_data[source3_data['response'] == 0].iloc[:, 1:2].values
#y3_1 = source3_data[source3_data['response'] == 1].iloc[:, 1:2].values
r1 = np.concatenate((r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_9), axis=0)
r1_2 = np.concatenate((r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, r2_7, r2_8, r2_9, r2_10), axis=0)
r2 = np.concatenate((t_r,), axis=0)
r2_2 = np.concatenate((t2_r,), axis=0)
# tr0 = np.concatenate((y2_0,), axis=0)
# tr1 = np.concatenate((y2_1,), axis=0)
# s_11, r_11 = smote.fit_resample(s1, r1)
s1 = torch.Tensor(s1)
s1_2 = torch.Tensor(s1_2)
s2 = torch.Tensor(s2)
s2_2 = torch.Tensor(s2_2)
# s_11 = torch.Tensor(s_11)
# r_11 = torch.Tensor(r_11)
# t1 = torch.Tensor(t1)
# t2 = torch.Tensor(t2)
r1 = torch.Tensor(r1)
r1_2 = torch.Tensor(r1_2)
r2 = torch.Tensor(r2)
r2_2 = torch.Tensor(r2_2)
# tr0 = torch.Tensor(tr0)
# tr1 = torch.Tensor(tr1)
#s1_train,l1_train,r0_train = train_test_split(s1,l1,r0,random_state=66)
#s2_train,l2_train,r1_train = train_test_split(s2,l2,r1,random_state=66)


s_train = TensorDataset(s1, l1, r1)
s2_train = TensorDataset(s1_2, l1_2, r1_2)
# s_t = TensorDataset(s_11, r_11)
s_loader = DataLoader(s_train, batch_size=bs, shuffle=False, drop_last=True)
s2_loader = DataLoader(s2_train, batch_size=bs, shuffle=False, drop_last=True)
# s_loader2 = DataLoader(s_t, batch_size= bs,shuffle=True,drop_last=True)

s_test = TensorDataset(s2, l2, r2)
s2_test = TensorDataset(s2_2, l2_2, r2_2)
s_test = DataLoader(s_test, batch_size=bs, shuffle=False)
s2_test = DataLoader(s2_test, batch_size=bs, shuffle=False)
# s2_train = TensorDataset(s2, l2, r1)
# s2_loader = DataLoader(s2_train, batch_size=bs, shuffle=True, drop_last=True)
# s2_test = TensorDataset(t2, tl2, tr1)
# s2_test = DataLoader(s2_test,batch_size=bs,shuffle=True)
model_p = model_plus(s2.shape[1], )

CEloss = nn.BCEWithLogitsLoss()
adversarial_loss = nn.CrossEntropyLoss()
model = Model_upgrade(s1.shape[1], 2048, 603).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
#discriminator_optimizer = torch.optim.Adam(Model.parameters(), lr=0.0002, betas=(0.5, 0.999))




losses = []
tlosses = []
auc_scores = []
tauc_scores = []
epochs = 2000
for epo in range(epochs):

    pred_ = []
    label_ = []
    # for i, data in enumerate(zip(s_loader2,cycle(s_loader))):
    for i, data in enumerate(s_loader):
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.discriminator.parameters():
            param.requires_grad = True
        for param in model.TripletLoss.parameters():
            param.requires_grad = True
        for param in model.trans.parameters():
            param.requires_grad = False

        s1_ = data[0].to(device)
        l1_ = data[1].to(device)
        r1_ = data[2].to(device)
        # s2_ = data[0][0].to(device)
        # r2_ = data[0][1].to(device)
        c2, d3, trip = model(s1_, r1_,)

        c2 = c2.flatten()
        loss2 = CEloss(c2, r1_)
        loss3 = adversarial_loss(d3, l1_)
        total_loss = loss2+trip+loss3
        r1_ = r1_
        label_0 = r1_.cpu().numpy()
        c2_0 = c2
        c2_0 = c2_0.detach().cpu().numpy()
        label_.extend(label_0)
        pred_.extend(c2_0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    loss_ = total_loss.item()
    # print(label_)
    # print(pred_)
    auc = roc_auc_score(label_, pred_)
    auc_scores.append(auc)
    losses.append(loss_)
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.discriminator.parameters():
        param.requires_grad = False
    for param in model.TripletLoss.parameters():
        param.requires_grad = False
    for param in model.trans.parameters():
        param.requires_grad = True

        tlabel_ = torch.Tensor().to(device)
        tpred_ = torch.Tensor().to(device)

        for i, data_t in enumerate(s_test):
            t1_ = data_t[0].to(device)
            tl_ = data_t[1].to(device)
            tr_ = data_t[2].to(device)
            tc2, td3, t_trip = model(t1_, tr_)
            tc2 = tc2.flatten()
            loss_t = CEloss(tc2, tr_)
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            tlabel_ = torch.cat((tlabel_, tr_), dim=0)

            tpred_ = torch.cat((tpred_, tc2), dim=0)

        # print(tlabel_)
        # print(tpred_)
    tlabel_ = tlabel_.cpu().numpy()
    tpred_ = tpred_.cpu().detach().numpy()
    tauc = roc_auc_score(tlabel_, tpred_)
    print(tauc)
    tauc_scores.append(tauc)
x = range(1, 2001)
#生成auc曲线
plt.plot(x, auc_scores, marker='.', linestyle='-', color='b')
plt.plot(x, tauc_scores, marker='.', linestyle='-', color='r')
plt.xlim(0, max(x))
plt.ylim(0, 1)
# 添加标题和轴标签
plt.title('AUC Scores over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

#生成loss曲线
plt.plot(x, losses, marker='.', linestyle='-', color='b')

# 添加标题和轴标签
plt.title('losses over Epochs')
plt.xlabel('Epoch')
plt.ylabel('loss')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()