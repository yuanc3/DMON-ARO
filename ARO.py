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

    return dist

def getNewDismat(dismat, k, mean: bool = False):
    val, rank = dismat.topk(k, largest=False)
    dismat[dismat > val[:, -1].unsqueeze(1)] = 1
    return dismat

def aro(qf: torch.tensor, gf: torch.tensor, k1, k2):
    qf = qf.to('cuda')
    gf = gf.to('cuda')
 
    dist_qg = pairwise_distance(qf, gf)
    dist_gg = pairwise_distance(gf, gf)

    dist_qg = torch.nn.functional.normalize(dist_qg)
    dist_gg = torch.nn.functional.normalize(dist_gg)

    qg2 = torch.concat([getNewDismat(dist_qg.clone(), k1, largest=False)], dim=1)
    gg2 = torch.concat([getNewDismat(dist_gg.clone(), k2, largest=False)], dim=1)
    qg2 = torch.nn.functional.normalize(qg2)
    gg2 = torch.nn.functional.normalize(gg2)
   
    di = qg2 @ gg2.T

    return (dist_qg-di).to('cpu')

