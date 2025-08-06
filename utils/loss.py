import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from pyitlib import discrete_random_variable as drv
import math


# 也吃显存
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, x):
        """
        :param x: [B, V, D] — B: batch, V/C: variables, D: features
        :return: scalar InfoNCE loss
        """
        B, V, D = x.shape
        x = F.normalize(x, dim=-1)  # 归一化特征向量

        loss_total = 0.0
        count = 0

        for i in range(V):
            anchor = x[:, i, :]              # [B, D]
            positives = []
            negatives = []

            for j in range(V):
                if i == j:
                    continue
                positives.append(x[:, j, :])  # 正样本（其他变量作为正）
            positives = torch.stack(positives, dim=1)  # [B, V-1, D]

            # anchor: [B, 1, D], positives: [B, V-1, D] → sim: [B, V-1]
            sim = torch.matmul(positives, anchor.unsqueeze(2)).squeeze(2) / self.temperature

            # 使用 softmax 处理相似度作为分类目标
            labels = torch.zeros(B, dtype=torch.long, device=x.device)  # 每一行的第一个为正样本
            loss = F.cross_entropy(sim, labels)
            loss_total += loss
            count += 1

        return loss_total / count



class MINE_MutualInformation(nn.Module):
    def __init__(self, dim, hidden_dim=64, reduction='mean'):
        """
        :param dim: 每个变量的特征维度 D
        :param hidden_dim: MLP 隐藏层大小
        :param reduction: 'mean' or 'sum' over variable pairs
        """
        super().__init__()
        self.reduction = reduction
        self.T = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        :param x: [B, C, D]  —— B=batch size, C=变量个数, D=特征维度
        :return: mutual information loss (scalar, to minimize)
        """
        x = x.to(self.T[0].weight.device)
        x = x.permute(0, 2, 1)          # [B, D, C] → [B, C, D]
        B, C, D = x.shape
        total_mi = 0.0
        count = 0

        for i in range(C):
            for j in range(i + 1, C):
                xi = x[:, i, :]  # [B, D]
                xj = x[:, j, :]  # [B, D]

                # 联合分布
                joint = torch.cat([xi, xj], dim=1)  # [B, 2D]
                joint_term = self.T(joint)  # [B, 1]

                # 边缘分布（打乱 xj）
                xj_shuffle = xj[torch.randperm(B)]
                marginal = torch.cat([xi, xj_shuffle], dim=1)
                marginal_term = self.T(marginal)  # [B, 1]

                # MI 下界估计
                mi = joint_term.mean() - torch.log(torch.exp(marginal_term).mean() + 1e-8)
                total_mi += mi
                count += 1

        total_mi = total_mi / count if self.reduction == 'mean' else total_mi
        return total_mi  # 负的互信息（用于最小化）


class VectorizedMINE(nn.Module):
    def __init__(self, dim, hidden_dim=64, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.T = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.register_buffer('triu_indices', None)

    def forward(self, x):
        x = x.to(self.T[0].weight.device)
        x = x.permute(0, 2, 1)          # [B, D, C] → [B, C, D]
        B, C, D = x.shape
        # 生成上三角索引（仅首次运行时计算）
        if self.triu_indices is None or self.triu_indices.size(1) != C*(C-1)//2:
            i, j = torch.triu_indices(C, C, offset=1, device=x.device)
            self.triu_indices = torch.stack([i, j])  # [2, K]
        
        # 批量提取变量对 [B, K, D]
        xi = x[:, self.triu_indices[0], :]  # [B, K, D]
        xj = x[:, self.triu_indices[1], :]  # [B, K, D]
        
        # 联合分布判别
        joint = torch.cat([xi, xj], dim=-1).view(-1, 2*D)  # [B*K, 2D]
        joint_term = self.T(joint).view(B, -1)  # [B, K]
        
        # 边缘分布构造（独立打乱每个变量对）
        perm = torch.randperm(B, device=x.device).expand(len(self.triu_indices[0]), -1).T  # [B, K]
        xj_shuffle = xj.gather(0, perm.unsqueeze(-1).expand(-1, -1, D))  # [B, K, D]
        marginal = torch.cat([xi, xj_shuffle], dim=-1).view(-1, 2*D)
        marginal_term = self.T(marginal).view(B, -1)  # [B, K]
        
        # 互信息计算（数值稳定）
        mi_per_pair = joint_term - marginal_term.logsumexp(dim=0) + math.log(B)
        total_mi = mi_per_pair.sum(dim=1).mean()  # 先按变量对求和，再按批次平均
        
        if self.reduction == 'mean':
            return -total_mi / (C*(C-1)//2)  # 变量对平均
        else:
            return -total_mi



# 使用sklearn实现的互信息
def MutualInfoCalculator(x):
    B, L, C = x.shape
    # 调整维度顺序并重塑，每个变量序列是一个整体
    x_np = x.permute(0, 2, 1).reshape(B, C * L).detach().cpu().numpy()

    mi_matrix = np.zeros((C, C))
    for i in range(C):
        # 提取第i个变量的完整序列并展开
        var_i = x_np[:, i * L:(i + 1) * L].flatten()
        for j in range(C):
            # 提取第j个变量的完整序列并展开
            var_j = x_np[:, j * L:(j + 1) * L].flatten()
            # 计算第i个和第j个变量序列之间的互信息
            mi_matrix[i, j] = mutual_info_regression(var_i.reshape(-1, 1), var_j)[0]

    # 去除对角线，返回非对角元素均值
    mean_mi = (np.sum(mi_matrix) - np.trace(mi_matrix)) / (C * (C - 1))
    return torch.tensor(mean_mi, dtype=torch.float32, device=x.device)




def MutualInfoCalculator2(x):
    B, L, C = x.shape
    # 把数据移到 GPU 上
    x = x.to(torch.float32)
    mi_matrix = torch.zeros((C, C), device=x.device)
    num_bins = 10  # 离散化的区间数量，可按需调整

    for i in range(C):
        for j in range(C):
            # 提取第 i 个和第 j 个变量的完整序列
            var_i = x[:, :, i].flatten()
            var_j = x[:, :, j].flatten()

            # 手动计算二维直方图
            min_i, max_i = var_i.min(), var_i.max()
            min_j, max_j = var_j.min(), var_j.max()
            bins_i = torch.linspace(min_i, max_i, num_bins + 1, device=x.device)
            bins_j = torch.linspace(min_j, max_j, num_bins + 1, device=x.device)
            hist_ij = torch.zeros((num_bins, num_bins), device=x.device)
            for k in range(B * L):
                bin_i = torch.bucketize(var_i[k], bins_i, right=False) - 1
                bin_j = torch.bucketize(var_j[k], bins_j, right=False) - 1
                if 0 <= bin_i < num_bins and 0 <= bin_j < num_bins:
                    hist_ij[bin_i, bin_j] += 1

            # 计算一维直方图
            hist_i = torch.histc(var_i, bins=num_bins, min=float(min_i), max=float(max_i))
            hist_j = torch.histc(var_j, bins=num_bins, min=float(min_j), max=float(max_j))

            # 计算联合概率和边缘概率
            p_ij = hist_ij / (B * L)
            p_i = hist_i / (B * L)
            p_j = hist_j / (B * L)

            # 避免除零错误
            p_ij = p_ij.masked_fill(p_ij == 0, 1e-10)
            p_i = p_i.masked_fill(p_i == 0, 1e-10)
            p_j = p_j.masked_fill(p_j == 0, 1e-10)

            # 计算互信息
            mi = (p_ij * torch.log(p_ij / (p_i.unsqueeze(1) * p_j.unsqueeze(0)))).sum()
            mi_matrix[i, j] = mi

    # 去除对角线，返回非对角元素均值
    mask = ~torch.eye(C, dtype=torch.bool, device=x.device)
    mean_mi = mi_matrix[mask].mean()
    return mean_mi



def MutualInfoCalculator3(x, method='hist', bins=10):
    """
    完整时间序列互信息计算（输入形状[B, L, C]）
    参数说明：
    method - 'knn'(连续变量)/'hist'(离散化)
    bins - 离散化分箱数
    """
    # 检查输入张量的维度
    assert x.dim() == 3, f"Input tensor should be 3D, but got {x.dim()}D."
    B, L, C = x.shape
    mi_matrix = torch.zeros((C, C), device=x.device)

    try:
        # 时序特征构造（含滑动窗口）
        x_padded = torch.nn.functional.pad(x, (0, 0, 1, 0, 0, 0), mode='replicate')
    except NotImplementedError:
        print("Replicate padding is not supported. Using constant padding instead.")
        x_padded = torch.nn.functional.pad(x, (0, 0, 1, 0, 0, 0), mode='constant', value=0)

    x_structured = torch.stack([x_padded[:, :-1, :], x_padded[:, 1:, :]], dim=-1).reshape(B * (L - 1), C, 2)

    for i in range(C):
        for j in range(i + 1, C):
            # 提取成对时间序列
            seq_i = x_structured[:, i, :]  # [B*(L-1), 2]
            seq_j = x_structured[:, j, :]  # [B*(L-1), 2]

            if method == 'knn':
                if drv is None:
                    raise ImportError("The 'dit' library is required for the 'knn' method.")
                # 使用k近邻法计算连续变量互信息
                seq_i_np = seq_i.cpu().numpy()
                seq_j_np = seq_j.cpu().numpy()
                mi = drv.mutual_information_xy(
                    seq_i_np, seq_j_np,
                    cartesian_product=True,
                    k=5  # 近邻数
                )
                mi = torch.tensor(mi, device=x.device)
            else:
                # 离散化处理
                min_i, max_i = seq_i.min(), seq_i.max()
                min_j, max_j = seq_j.min(), seq_j.max()
                bins_i = torch.linspace(min_i, max_i, bins + 1, device=x.device)
                bins_j = torch.linspace(min_j, max_j, bins + 1, device=x.device)
                disc_i = torch.bucketize(seq_i, bins_i)
                disc_j = torch.bucketize(seq_j, bins_j)
                disc_i_np = disc_i.cpu().numpy().flatten()
                disc_j_np = disc_j.cpu().numpy().flatten()
                mi = mutual_info_score(disc_i_np, disc_j_np)
                mi = torch.tensor(mi, device=x.device)

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix



