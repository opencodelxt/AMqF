import math

import torch
import torch.nn.functional as F
from torch import nn

from models.common import FeatureExtractor, Decoder


def decov_loss(x):
    x_mean = x - torch.mean(x, dim=1, keepdim=True)
    x_cov = x_mean.mm(x_mean.T)
    loss = torch.norm(x_cov, p='fro') - (torch.diag(x_cov) ** 2 + 1e-6).sum().sqrt()
    return 0.5 * loss


class JSScore(nn.Module):
    def __init__(self):
        super(JSScore, self).__init__()

    def forward(self, ref, dist):
        return 1 - js_div(ref, dist, get_softmax=True)


class CosScore(nn.Module):
    def __init__(self):
        super(CosScore, self).__init__()

    def forward(self, ref, dist):
        return F.cosine_similarity(ref, dist, dim=1).view(-1, 1)


class IQA_Model(nn.Module):
    def __init__(self, num_words=1024):
        super().__init__()
        self.encoder = FeatureExtractor()
        self.light_adapter = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.struct_adapter = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.contrast_adapter = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.decoder = Decoder(out_channels=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vocab = nn.Parameter(torch.Tensor(num_words, 512), requires_grad=True)
        self.cos_score = CosScore()
        nn.init.kaiming_normal_(self.vocab, a=math.sqrt(5))

    def forward(self, dist, ref):
        # 构建词典基
        vocab_norm = F.normalize(self.vocab, p=2, dim=1)
        vocab_norm = vocab_norm.unsqueeze(2).unsqueeze(3)

        # 提取特征
        dist_feat = self.encoder(dist)[-1]
        ref_feat = self.encoder(ref)[-1]

        # 多头特征 (2048 -> 512)
        dist_cls, light_dist_feat, struct_dist_feat, contrast_dist_feat = dist_feat.chunk(4, dim=1)
        ref_cls, light_ref_feat, struct_ref_feat, contrast_ref_feat = ref_feat.chunk(4, dim=1)

        light_dist_feat_new = light_dist_feat + dist_cls
        struct_dist_feat_new = struct_dist_feat + dist_cls
        contrast_dist_feat_new = contrast_dist_feat + dist_cls

        light_ref_feat_new = light_ref_feat + ref_cls
        struct_ref_feat_new = struct_ref_feat + ref_cls
        contrast_ref_feat_new = contrast_ref_feat + ref_cls

        light_ref_feat_new = self.light_adapter(light_ref_feat_new)
        struct_ref_feat_new = self.struct_adapter(struct_ref_feat_new)
        contrast_ref_feat_new = self.contrast_adapter(contrast_ref_feat_new)

        # 计算各质量因子在词典基上的投影
        dist_feats = [light_dist_feat_new, struct_dist_feat_new, contrast_dist_feat_new]
        ref_feats = [light_ref_feat_new, struct_ref_feat_new, contrast_ref_feat_new]
        for i, feat in enumerate(dist_feats):
            feat_norm = F.normalize(feat, p=2, dim=1)
            dist_feats[i] = F.conv2d(feat_norm, weight=vocab_norm)
            dist_feats[i] = self.avgpool(dist_feats[i]).flatten(1)
        for i, feat in enumerate(ref_feats):
            feat_norm = F.normalize(feat, p=2, dim=1)
            ref_feats[i] = F.conv2d(feat_norm, weight=vocab_norm)
            ref_feats[i] = self.avgpool(ref_feats[i]).flatten(1)

        # 计算各质量因子的相似度
        score = sum([self.cos_score(ref_feat, dist_feat) for ref_feat, dist_feat in
                     zip(ref_feats, dist_feats)]) / len(ref_feats)
        # 根据各质量因子重建图像
        if self.training:
            ref_light_img = self.decoder(light_ref_feat_new).mean(dim=1)
            ref_struct_img = self.decoder(struct_ref_feat_new).mean(dim=1)
            ref_contrast_img = self.decoder(contrast_ref_feat_new).mean(dim=1)
            dist_light_img = self.decoder(light_dist_feat_new).mean(dim=1)
            dist_struct_img = self.decoder(struct_dist_feat_new).mean(dim=1)
            dist_contrast_img = self.decoder(contrast_dist_feat_new).mean(dim=1)
            return (score, ref_light_img, ref_struct_img, ref_contrast_img,
                    dist_light_img, dist_struct_img, dist_contrast_img)
        else:
            return score


# if __name__ == '__main__':
#     model = ThreeBoVW()
#     model.cuda()
#
#     ref = torch.randn(4, 3, 224, 224, device='cuda')
#     dist = torch.randn(4, 3, 224, 224, device='cuda')
#     (score, ref_light_img, ref_struct_img, ref_contrast_img, dist_light_img,
#      dist_struct_img, dist_contrast_img) = model(dist, ref)
#     print(score)
#     print(ref_light_img.shape)
