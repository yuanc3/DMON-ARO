import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torchvision.transforms import InterpolationMode, RandomCrop, Resize

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, paths) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            # camids = camids.to(device)
                            camids = torch.zeros_like(camids, dtype=torch.long).to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view,paths) in enumerate(val_loader):
      
                    with torch.no_grad():
                        img = img.to(device)
                        # camids = camids.to(device)
                        camids = torch.zeros_like(camids, dtype=torch.long).to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

# 水平翻转
def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

# 随机裁剪和调整大小
def random_crop(img, crop_size):
    # 创建随机裁剪变换
    transform = RandomCrop(crop_size)
    # 先裁剪，再调整大小
    cropped_img = transform(img)
    resized_img = Resize((256, 128), interpolation=InterpolationMode.BILINEAR)(cropped_img)
    return resized_img


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):

            img = img.to(device)

            camids = camids.to(device)

            target_view = target_view.to(device)


            with torch.no_grad():
                feat = model(img, cam_label=camids, view_label=target_view)

                # feat += model(fliplr(img), cam_label=camids2, view_label=target_view)
            
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


# def do_inference(cfg, model, val_loader, num_query, batch_size=64):
#     device = "cuda"
#     logger = logging.getLogger("transreid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
#     evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()
#     img_path_list = []

#     all_feats = []  # 用于存储所有的特征
#     all_pids = []   # 用于存储所有的PID
#     all_camids = [] # 用于存储所有的camid
#     all_imgpaths = [] # 用于存储所有的图片路径

#     for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
#         # 输出调试信息，检查 pid 和 camid 的类型
#         print(f"Processing batch {n_iter+1}/{len(val_loader)}")
#         print(f"pid type: {type(pid)}, camid type: {type(camid)}")

#         img = img.to(device)
#         camids = camids.to(device)
#         target_view = target_view.to(device)

#         # 处理 pid（如果它是元组，提取第一个元素）
#         if isinstance(pid, tuple):
#             pid = pid[0]  # 假设 pid 是一个元组，取第一个元素

#         # 确保 pid 是 Tensor 类型
#         if isinstance(pid, torch.Tensor):
#             all_pids.extend(pid.cpu())  # 如果是 Tensor 类型，使用 cpu() 转移到 CPU 后再扩展
#         else:
#             all_pids.append(pid)  # 如果是 int 类型，直接 append

#         # 处理 camid（如果它是元组，提取第一个元素）
#         if isinstance(camid, tuple):
#             camid = camid[0]  # 假设 camid 是一个元组，取第一个元素

#         # 确保 camid 是 Tensor 类型
#         if isinstance(camid, torch.Tensor):
#             all_camids.extend(camid.cpu())  # 如果是 Tensor 类型，使用 cpu() 转移到 CPU 后再扩展
#         else:
#             all_camids.append(camid)  # 如果是 int 类型，直接 append

#         with torch.no_grad():
#             feat = model(img, cam_label=camids, view_label=target_view)
#             # feat += model(fliplr(img), cam_label=camids2, view_label=target_view) # 如果需要进行翻转

#         # 将当前批次的特征、PID、CAMID 等信息保存下来
#         all_feats.append(feat.cpu())  # 将特征从GPU转到CPU，避免占用过多显存
#         all_imgpaths.extend(imgpath)

#         # 清理显存
#         torch.cuda.empty_cache()

#     # 合并所有批次的特征
#     all_feats = torch.cat(all_feats, dim=0)  # 合并所有的特征
#     all_pids = torch.tensor(all_pids)  # 转换为Tensor
#     all_camids = torch.tensor(all_camids)  # 转换为Tensor

#     # 更新评估器
#     evaluator.update((all_feats, all_pids, all_camids))

#     # 计算评价指标
#     cmc, mAP, _, _, _, _, _ = evaluator.compute()

#     # 输出评估结果
#     logger.info("Validation Results ")
#     logger.info("mAP: {:.1%}".format(mAP))
#     for r in [1, 5, 10]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

#     return cmc[0], cmc[4]







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

import pickle

def do_inference_dis(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    # 存1501个id的特征
    # all_pids = set()
    # for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
    #     all_pids.update(pid)
    # # 去掉-1和0
    # all_pids = list(all_pids)
    # # all_pids.remove(-1)
    # all_pids.remove(0)
    # all_pids = sorted(all_pids)
    # print( len(all_pids))
    # exit()
    
    
    
    
    with open('dist.pkl', 'rb') as f:
        img_path_list = pickle.load(f)
        
    start = len(img_path_list)
    
    
    from tqdm import tqdm
    import numpy as np
    for id in tqdm(range(1+start,1502)):
        feat_all = []
        img_all = None
        camids_all = None
        target_view_all = None
        img_path_all = []
        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                pid = np.array(pid)
    
                img = img[pid == id]
                camids = camids[pid == id]
                target_view = target_view[pid == id]
                imgpath = np.array(imgpath)[pid == id]
                img_path_all.extend(imgpath)
                if img.shape[0] == 0:
                    continue
                if img_all is None:
                    img_all = img
                    camids_all = camids
                    target_view_all = target_view
                    continue
                if img_all.shape[0] < 256:
                    img_all = torch.cat((img_all, img), dim=0)
                    camids_all = torch.cat((camids_all, camids), dim=0)
                    target_view_all = torch.cat((target_view_all, target_view), dim=0)
                    continue
                
                feat = model(img_all, cam_label=camids_all, view_label=target_view_all)
                # feat += model(fliplr(img), cam_label=camids, view_label=target_view)

                feat_all.append(feat)
                img_all = None
                camids_all = None
                target_view_all = None

                
        
        if img_all is not None:
            with torch.no_grad():
                feat = model(img_all, cam_label=camids_all, view_label=target_view_all)
            # feat += model(fliplr(img), cam_label=camids, view_label=target_view)
            feat_all.append(feat)


        feat_all = torch.cat(feat_all, dim=0)
        import torch.nn.functional as F
        F.normalize(feat_all, p=2, dim=1)
        # dist = pairwise_distance(feat_all, feat_all)
        feat_mean = feat_all.mean(0, keepdim=True)
        dist = pairwise_distance(feat_mean, feat_all)

        # 获得dist最小的5%的img_path_list
        dist = dist.squeeze().detach()
        dist = dist.cpu().numpy()
        idx = np.argsort(dist)
        
        length = len(idx)
        # idx = idx[:int(length*0.05)]
        img_path_all = np.array(img_path_all)
        
        paths = img_path_all[idx]
        img_path_list.append(paths)

        with open('dist_market.pkl', 'wb') as f:
            pickle.dump(img_path_list, f)
            
      
        




