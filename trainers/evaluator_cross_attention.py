# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

from trainers.re_ranking import re_ranking as re_ranking_func

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def cross_attention(anchor, output):
    c, h, w = anchor.size()
    anchor = anchor.view(c, h*w)
    output = output.view(c, h*w)
    anchor = anchor.t()
    attention = anchor.mm(output)
    attention = F.softmax(attention, 0)
    output_ = output.mm(attention)
    output1 = torch.mean(output_, -1)
    output2 = torch.mean(output, -1)
    diff = torch.sum(output2-output1)
    output = output1+output2
    return output


class ResNetEvaluator:
    def __init__(self, model):
        self.model = model

    def save_incorrect_pairs(self, distmat, queryloader, galleryloader,
                             g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes = plt.subplots(1, 11, figsize=(12, 8))
            img = queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j + 1].set_title(g_pids[gallery_index])
                axes[j + 1].set_axis_off()
                axes[j + 1].imshow(img)
            fig.savefig(os.path.join(savefig, '%d.png' % q_pids[i]))
            plt.close(fig)

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
                 ranks=[1, 2, 4, 5, 8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        qf, q_pids, q_camids, q_raw_feat = [], [], [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)

            for i in range(feature0.size()[0]):
                q_raw_feat.append(normalize(feature0[i, :, :, :], axis=0))
                temp = cross_attention(normalize(feature0[i, :, :, :], axis=0), normalize(feature0[i, :, :, :], axis=0))
                qf.append(temp)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.stack(qf, 0)
        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        q_lst = []
        cnt = 0
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            for q in range(qf.size()[0]):
                g_lst = []
                for g in range(feature0.size()[0]):
                    cur_gf = cross_attention(normalize(q_raw_feat[q], axis=0), normalize(feature0[g, :, :, :], axis=0))
                    g_lst.append(torch.sum((qf[i, :] - cur_gf) ** 2).clamp(min=1e-12).sqrt())
                cnt+=1
                print(cnt)
                q_lst.append(g_lst)

            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)

            g_pids.extend(pids)
            g_camids.extend(camids)

        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        distmat = torch.tensor(q_lst)

        if savefig:
            print("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader,
                                      g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1])
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices] == q_camids.view([num_q, -1])))
        # keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank + 1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP
