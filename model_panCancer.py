import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv


# class AsymmetricAdaptiveClustering(nn.Module):
#     def __init__(self, n_clusters, data_dim, alpha=0.5, device='cuda:0'):
#         """
#         初始化 AsymmetricAdaptiveClustering
#         :param n_clusters: 聚类数量
#         :param data_dim: 特征维度
#         :param alpha: 自适应强度（0-1之间）
#         :param device: 计算设备
#         """
#         super(AsymmetricAdaptiveClustering, self).__init__()
#         self.n_clusters = n_clusters
#         self.data_dim = data_dim
#         self.alpha = alpha
#         self.device = device
#
#         # 初始化簇中心
#         self.cluster_centers = nn.Parameter(
#             torch.rand(n_clusters, data_dim, device=device)
#         )
#
#     def compute_distances(self, data):
#         """
#         计算数据到每个簇中心的距离
#         :param data: 输入数据 (batch_size, data_dim)
#         :return: 距离矩阵 (batch_size, n_clusters)
#         """
#         distances = torch.cdist(data, self.cluster_centers, p=2)  # 欧式距离
#         return distances
#
#     def update_centers(self, data, labels=None):
#         """
#         更新簇中心并计算损失
#         :param data: 输入数据 (batch_size, data_dim)
#         :param labels: 可选的监督标签 (batch_size,)
#         :return: 聚类损失
#         """
#         # 计算距离矩阵
#         distances = self.compute_distances(data)  # (batch_size, n_clusters)
#
#         # 计算软分配概率
#         soft_assignments = F.softmax(-self.alpha * distances, dim=1)  # (batch_size, n_clusters)
#
#         # 更新簇中心
#         weighted_sum = torch.matmul(soft_assignments.T, data)  # (n_clusters, data_dim)
#         cluster_weights = soft_assignments.sum(dim=0).unsqueeze(1)  # (n_clusters, 1)
#         cluster_weights = torch.clamp(cluster_weights, min=1e-6)  # 防止除零
#         new_centers = weighted_sum / cluster_weights  # (n_clusters, data_dim)
#
#         # 进行中心更新
#         self.cluster_centers.data = (
#                 (1 - self.alpha) * self.cluster_centers.data + self.alpha * new_centers
#         )
#
#         # 计算损失（平方距离）
#         loss = torch.sum(soft_assignments * distances) / data.size(0)
#         return loss
class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    # got the dot product between all embeddings
    cor_mat = torch.matmul(x, x.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    norm_mat = cor_mat.diag()

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = F.relu(distances)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0")

    #device = torch.device("cpu")
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0")

    #device = torch.device("cpu")
    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            #nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            # nn.Dropout(dropout_prob),
            #nn.Linear(hidden_size, output_size)
            nn.Linear(hidden_size, hidden_size),
            #nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        x1 = self.fc1(x)
        return x1

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        input = ctx.saved_tensors
        iter_num = getattr(ctx, 'iter_num', 0)
        alpha = 10.0
        low = 0.0
        high = 1.0
        max_iter = 100

        coeff = float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
                         - (high - low) + low)
        return -coeff * gradOutput

class GRL(torch.nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, input):
        return GRLFunction.apply(input)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(740, 128)
        #self.fc1.weight.data.normal_(0, 0.01)
        #self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 21)
        #self.fc2.weight.data.normal_(0, 0.3)
        #self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64, 1)


    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        # x = torch.where(x >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        return x


# def get_positional_encoding(d_model):
#     # 初始化位置编码矩阵
#     positional_encoding = np.zeros((d_model, d_model))
#
#     # 计算每个维度对应的周期
#
#     position = np.arange(1, d_model // 2 + 1) * np.pi
#
#     div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
#
#     # 对于偶数维度应用余弦函数
#     positional_encoding[:, 0::2] = np.cos(position*div_term)
#     # 对于奇数维度应用正弦函数
#     positional_encoding[:, 1::2] = np.sin(position*div_term)
#
#     return positional_encoding
def get_positional_encoding(d_model):
    # 初始化位置编码矩阵
    positional_encoding = np.zeros((d_model, d_model))

    # 计算每个维度对应的周期
    x = np.linspace(0, 2 * np.pi, d_model)  # 从0到2π的d_model个采样点
    for i in range(d_model):
        frequency = (i + 1)  # 周期与行号相关
        if i % 2 == 0:  # 偶数行使用余弦
            positional_encoding[i, :] = np.cos(frequency * x)
        else:  # 奇数行使用正弦
            positional_encoding[i, :] = np.sin(frequency * x)

    return positional_encoding

class Model(nn.Module):
    def __init__(self, feature_size, feature_hidden_size, prediction_input_size,):
        super(Model, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_size, feature_hidden_size, prediction_input_size)
        positional_encoding = get_positional_encoding(prediction_input_size)
        self.positional_encoding = torch.tensor(positional_encoding, dtype=torch.float32).unsqueeze(0)
        self.classifier = Classifier(prediction_input_size)
        self.discriminator = Discriminator()
        self.TripletLoss = HardTripletLoss()
        # self.aac = AsymmetricAdaptiveClustering(
        #     n_clusters=2,
        #     data_dim=prediction_input_size,
        #     alpha=0.5,
        #     )
    def forward(self, x1, r1, ):
        features1 = self.feature_extractor(x1)
        # features2 = self.feature_extractor(x2)
        # features0 = self.feature_extractor(x2)
        # features = torch.cat((features0,features1),dim=0)
        #r0 = torch.zeros((32,1))
        #r1 = torch.ones((32,1))
        # r = (torch.cat((r0,r1),dim=0)).to("cuda:1")
        # positional_encoding = self.positional_encoding.to(features1.device)
        positional_encoding = self.positional_encoding.squeeze(0).to(features1.device)
        positional_encoding = positional_encoding.T
        features1 = torch.matmul(features1, positional_encoding)
        mean = torch.mean(features1, dim=0)

        # 计算每个特征的标准差
        std = torch.std(features1, dim=0)

        # 对特征进行标准化
        features1 = (features1 - mean) / std

        # print(features1)
        normalized_features1 = features1
        # normalized_features2 = features2
        # normalized_features = (features / l2_norm)
        #df = pd.DataFrame(normalized_features.detach().numpy())
        #csv_file_path = '/data/sr/tensor_data2.csv'
        #df.to_csv(csv_file_path, mode='a', index=False, header=False)
        class_2 = self.classifier(normalized_features1)
        dis_3 = self.discriminator(normalized_features1)
        #print(dis_3)

        trip = self.TripletLoss(normalized_features1, r1)
        # aac_loss = self.aac.update_centers(normalized_features1, r1)

        return class_2, dis_3, trip, normalized_features1
