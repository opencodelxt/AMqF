import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class FeatureExtractorVGG(nn.Module):
    def __init__(self):
        super(FeatureExtractorVGG, self).__init__()
        self.model = models.vgg16(pretrained=True)  # 改为VGG16模型
        self.layers = list(self.model.features.children())  # 获取VGG16的卷积层

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs[-3], outputs[-2], outputs[-1]  # 提取最后三层的特征


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.layers = list(self.model.children())[:-2]

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs[-3], outputs[-2], outputs[-1]


def dot_product_distance(feat1, feat2):
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    mat = torch.matmul(feat1, feat2.T) / 0.1
    mat = F.softmax(mat, dim=-1)
    similarity = torch.diag(mat)
    return similarity


def js_div(p_output, q_output, get_softmax=True):
    """
    计算JS散度（Jensen-Shannon Divergence）作为相似度度量。
    参数:
        p_output: 张量，样本P的输出
        q_output: 张量，样本Q的输出
        get_softmax: 布尔值，是否应用softmax
    返回:
        两个分布的平均KL散度
    """
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_output = F.softmax(p_output, dim=1)
        q_output = F.softmax(q_output, dim=1)

    mean_output = (p_output + q_output) / 2
    mean_output = mean_output.clamp(1e-6, 1 - 1e-6)
    p_output = p_output.clamp(1e-6, 1 - 1e-6)
    q_output = q_output.clamp(1e-6, 1 - 1e-6)
    log_mean_output = mean_output.log()

    part1 = KLDivLoss(log_mean_output, p_output).sum(dim=1)
    part2 = KLDivLoss(log_mean_output, q_output).sum(dim=1)
    return (part1 + part2) / 2


def js_distance(X, Y, win=7):
    """
    计算JS散度和L2距离的加权和。
    参数:
        X, Y: 输入特征图张量
        win: 窗口大小
    返回:
        加权后相似度分数
    """
    batch_size, chn_num, _, _ = X.shape
    # 重新调整输入特征图形状
    patch_x = X.shape[2] // win
    patch_y = X.shape[3] // win
    X_patch = X.view([batch_size, chn_num, win, patch_x, win, patch_y])
    Y_patch = Y.view([batch_size, chn_num, win, patch_x, win, patch_y])
    patch_num = patch_x * patch_y
    X_1D = X_patch.permute((0, 1, 3, 5, 2, 4)).contiguous().view([batch_size, -1, chn_num * patch_num])
    Y_1D = Y_patch.permute((0, 1, 3, 5, 2, 4)).contiguous().view([batch_size, -1, chn_num * patch_num])
    X_pdf = X_1D
    Y_pdf = Y_1D
    jsd = js_div(X_pdf, Y_pdf)
    L2 = ((X_1D - Y_1D) ** 2).sum(dim=1)
    w = (1 / (torch.sqrt(torch.exp((- 1 / (jsd + 10)))) * (jsd + 10) ** 2))
    final = jsd + L2 * w
    return final.mean(dim=1)



class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=512, kernel_size=3, out_channels=512, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=512, kernel_size=3, out_channels=512, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=512, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=256, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=out_channels, stride=1, pad=1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = self.activate(self.conv3(x))
        x = self.activate(self.conv4(x))
        x = self.conv5(x)
        return (torch.tanh(x) + 1) / 2


class DecoderVGG(nn.Module):
    def __init__(self, out_channels=1):
        super(DecoderVGG, self).__init__()
        self.conv1 = reflect_conv(in_channels=128, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=128, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = reflect_conv(in_channels=32, kernel_size=3, out_channels=out_channels, stride=1, pad=1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = self.activate(self.conv3(x))
        x = self.activate(self.conv4(x))
        x = self.conv5(x)
        return (torch.tanh(x) + 1) / 2



class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=pad)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def extract_luminance_structure_contrast(img):

    luminance = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]


    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(img.device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(luminance.unsqueeze(1), sobel_x, padding=1)
    grad_y = F.conv2d(luminance.unsqueeze(1), sobel_y, padding=1)
    structure = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(1)

    mean_luminance = F.avg_pool2d(luminance.unsqueeze(1), 3, stride=1, padding=1)
    contrast = torch.sqrt(F.avg_pool2d((luminance.unsqueeze(1) - mean_luminance) ** 2, 3, stride=1, padding=1)).squeeze(
        1)

    return luminance, structure, contrast
