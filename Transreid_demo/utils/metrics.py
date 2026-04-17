import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, paths, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)  # 排序获取索引
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_cmc_ori = []
    all_AP = []
    all_paths = []  # 存储每个 query 对应的 paths
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        all_cmc_ori.append(orig_cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        # 获取排序后的 paths
        valid_paths = [paths[i] for i, k in zip(order, keep) if k][:max_rank]
        all_paths.append(valid_paths)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc_copy = all_cmc_ori.copy()
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, all_cmc_copy, all_paths

def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, paths = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths.extend(paths)

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)

        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        
        q_paths = self.paths[:self.num_query]
        g_paths = self.paths[self.num_query:]
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            from utils.DMON import dmon
            from torch.nn import functional as F

            k1 = 2
            k2 = 2
            gf_ = dmon(gf.clone(), k1, k2)
            qf_ = dmon(qf.clone(), k1, k2)
            
            qf_ = F.normalize(qf_, dim=1)
            gf_ = F.normalize(gf_, dim=1)    
               
            from utils.ARO import aro
            distmat = aro(qf_, gf_, 20,20)
            

            
        cmc, mAP, all_cmc_1, all_paths1 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, g_paths)
     
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


from torch.nn import functional as F

def ensure_compatible_shape(qf, gf):
    """
    确保 qf 和 gf 在特征维度上兼容。若不兼容，则调整到相同的特征维度。
    """
    if qf.size(1) != gf.size(1):
        min_dim = min(qf.size(1), gf.size(1))
        qf = qf[:, :min_dim]  # 截取到相同的最小特征维度
        gf = gf[:, :min_dim]
    return qf, gf

def compute_dis(self):  # called after each epoch
    # 合并特征张量
    feats = torch.cat(self.feats, dim=0)

    # 归一化特征
    if self.feat_norm:
        print("The test feature is normalized")
        feats = F.normalize(feats, dim=1, p=2)  # 按通道归一化

    # 划分查询和库特征
    qf = feats[:self.num_query]
    q_pids = np.asarray(self.pids[:self.num_query])
    q_camids = np.asarray(self.camids[:self.num_query])

    gf = feats[self.num_query:]
    g_pids = np.asarray(self.pids[self.num_query:])
    g_camids = np.asarray(self.camids[self.num_query:])

    # 判断是否使用 re-ranking
    if self.reranking:
        print('=> Enter reranking')
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    else:
        print('=> Computing DistMat with euclidean_distance')

        # 使用 pose_enhance 和 pose_enhance_gallery 进行特征增强
        qf_, gf_ = pose_enhance(qf, gf, weight_q=2, weight_g=2)
        qf_, gf_ = pose_enhance_gallery(qf_, gf_, g_pids, weight_g=4)

        # 最终归一化
        qf = F.normalize(qf_, dim=1)
        gf = F.normalize(gf_, dim=1)

        # 计算欧式距离矩阵
        distmat = euclidean_distance(qf, gf)

    # 计算 CMC 和 mAP
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP, distmat, self.pids, self.camids, qf, gf

   
       



