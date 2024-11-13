import logging
import os
import cv2
import numpy as np
import torch
from models.AMqF import IQA_Model
from utils.process_image import ToTensor, five_point_crop
from utils.util import setup_seed, set_logging

class ImageQualityTester:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.load_model()

    def create_model(self):
        self.model = AMqF()
        self.model.cuda()

    def load_model(self):
        checkpoint = torch.load('best_csiq.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

    def evaluate_image(self, ref_image_path, dis_image_path):
        # Load images
        d_img_org = cv2.imread(dis_image_path)
        r_img_org = cv2.imread(ref_image_path)

        if d_img_org is None or r_img_org is None:
            logging.error("Could not read one of the images.")
            return None

        # Preprocess the images
        d_img_org = ToTensor()(d_img_org).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        r_img_org = ToTensor()(r_img_org).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        pred = 0
        for i in range(self.opt.num_avg_val):
            d_img_cropped, r_img_cropped = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
            pred_score = self.model(d_img_cropped, r_img_cropped)
            pred += pred_score

        pred /= self.opt.num_avg_val
        return pred.item()  # Return the predicted quality score

if __name__ == '__main__':
    config = TestOptions().parse()
    setup_seed(config.seed)
    set_logging(config)

    tester = ImageQualityTester(config)
    ref_image_path = 'path/to/reference/image.jpg'  # 替换为实际的参考图像路径 eg: images/1600.png
    dis_image_path = 'path/to/distorted/image.jpg'   # 替换为实际的失真图像路径 eg: images/1600.fnoise.4.png
    score = tester.evaluate_image(ref_image_path, dis_image_path)
    print(f'Predicted Image Quality Score: {score}')