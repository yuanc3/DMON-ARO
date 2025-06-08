import torch

def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist

def dmon(feat: torch.tensor, k1=2, sigma=0.5, gamma=0.75, decay_factors=[1, 0.5, 0.25, 0.125], chunk_size=128):
    feat = feat.clone()
    num_samples = feat.size(0)

    dist = torch.zeros((num_samples, num_samples), device='cpu')

    for i in range(0, num_samples, chunk_size):
        end_i = min(i + chunk_size, num_samples)
        feat_i = feat[i:end_i].to('cuda')
        for j in range(0, num_samples, chunk_size):
            end_j = min(j + chunk_size, num_samples)
            feat_j = feat[j:end_j].to('cuda')
            euc_dist = torch.cdist(feat_i, feat_j, p=2)
            dist[i:end_i, j:end_j] = euc_dist.to('cpu')
            del euc_dist, feat_j
            torch.cuda.empty_cache()
        del feat_i
        torch.cuda.empty_cache()
        
    eye = torch.eye(num_samples, device=dist.device)
    dist[eye == 1] = float('inf')

    _, rank = dist.topk(k1, largest=False)

    multi_hop_neighbors = [rank]
    for hop in range(2):  
        next_hop_neighbors = []
        for i in range(rank.size(0)):
            hop_neighbors = []
            for j in multi_hop_neighbors[-1][i]:
                hop_neighbors.extend(rank[j][:k2].tolist())
            next_hop_neighbors.append(list(set(hop_neighbors)))
        multi_hop_neighbors.append(next_hop_neighbors)

    weights_list = []
    for hop, neighbors in enumerate(multi_hop_neighbors):
        hop_mask = torch.zeros_like(dist, dtype=torch.bool)
        for i, hop_neighbors in enumerate(neighbors):
            hop_neighbors = list(set(hop_neighbors))
            hop_mask[i, hop_neighbors] = True

        current_sigma = sigma * (1.5 ** hop)
        weights = torch.exp(- (dist ** 2) / (2 * current_sigma ** 2))
        weights = weights * hop_mask
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8 
        weights = weights / weights_sum
        weights_list.append(decay_factors[hop] * weights)

    weighted_feat_sum = sum([torch.mm(weights, feat) for weights in weights_list])
 
    feat = gamma* feat + (1-gamma) * weighted_feat_sum
    return feat
