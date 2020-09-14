#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import torch
import numpy as np
from PIL import Image
from skimage import measure

import config
from models import darknet_model
from helper.image import adjust_lines, letterbox_image, draw_lines, draw_boxes, min_area_line, line_to_line, min_area_rect_box


class TABLE_NET:
    def __init__(self, weights=None, cfg=None, size=(512, 512)):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if cfg == None:
            cfg = config.tableNetCFGPath
        self.net = darknet_model.Darknet(cfg).to(self.device)

        if weights == None:
            weights = config.tableNetPath
        if weights.endswith('.weights'):
            self.net.load_darknet_weights(weights)
        else:
            self.net.load_state_dict(torch.load(weights))

        print('loading weights:%s Done' % weights)

        self.size = size
        self.net.eval()

    def draw_boxes(self, im, bboxes, color=(0, 0, 0)):
        return draw_boxes(im, bboxes, color)

    def draw_lines(self, im, bboxes, color=(0, 0, 0), lineW=3):
        return draw_lines(im, bboxes, color, lineW)

    def forward(self, img):

        img_resize, fx, fy, dx, dy = letterbox_image(img, self.size)
        img_resize = np.array(img_resize)
        w, h = self.size

        image = cv2.dnn.blobFromImage(
            img_resize, 1, size=(w, h), swapRB=False)
        image = torch.tensor(np.array(image)/255.0).view(1, 3, h, w)
        image = image.to(self.device)
        output = self.net(image)

        output = torch.exp(output[0])

        out = output[:, dy:, dx:]
        out = out.detach().numpy()

        return out, fx, fy

    def _get_table_rowcols(self, img, prob, row=100, col=100):

        out, fx, fy = self.forward(img)

        rows = out[0]
        cols = out[1]

        ##### rows计算过程#################################
        # rows > prob 是一个二值图： 0 是背景， 1是rows点
        # 测试：cv2.imwrite('test/rows_out.jpg', (rows > prob)*255)
        labels = measure.label(rows > prob, connectivity=2)

        # 1. 联通区域计算，
        # 2. 保留行的宽度 > row的联通区域，bbox = 边界外接框(min_row, min_col, max_row, max_col)
        # 3. 获取 coords区域内像素点坐标
        # 4. 计算rows线的坐标(x1, y1, x2, y2)
        regions = measure.regionprops(labels)
        rows_lines = [min_area_line(line.coords)
                      for line in regions if line.bbox[3]-line.bbox[1] > row]

        ##### cols计算过程#################################

        labels = measure.label(cols > prob, connectivity=2)
        regions = measure.regionprops(labels)
        cols_lines = [min_area_line(line.coords)
                      for line in regions if line.bbox[2]-line.bbox[0] > col]

        # 在resize尺寸的图片上draw_lines
        tmp = np.zeros(self.size[::-1], dtype='uint8')
        tmp = draw_lines(tmp, cols_lines+rows_lines, color=255, lineW=1)

        labels = measure.label(tmp > 0, connectivity=2)
        regions = measure.regionprops(labels)

        for region in regions:
            ymin, xmin, ymax, xmax = region.bbox
            label = region.label  # 区域标记
            # 比较窄区域去掉
            if ymax-ymin < 20 or xmax-xmin < 20:
                labels[labels == label] = 0

        labels = measure.label(labels > 0, connectivity=2)

        # 获取标记图坐标
        indY, indX = np.where(labels > 0)
        xmin, xmax = indX.min(), indX.max()
        ymin, ymax = indY.min(), indY.max()

        # 边界检查，去掉超过边界的rows和cols
        rows_lines = [p for p in rows_lines if xmin <= p[0] <= xmax and xmin <=
                      p[2] <= xmax and ymin <= p[1] <= ymax and ymin <= p[3] <= ymax]
        cols_lines = [p for p in cols_lines if xmin <= p[0] <= xmax and xmin <=
                      p[2] <= xmax and ymin <= p[1] <= ymax and ymin <= p[3] <= ymax]

        # 同比例缩放尺寸
        rows_lines = [[box[0]/fx, box[1]/fy, box[2]/fx, box[3]/fy]
                      for box in rows_lines]
        cols_lines = [[box[0]/fx, box[1]/fy, box[2]/fx, box[3]/fy]
                      for box in cols_lines]
        return rows_lines, cols_lines

    def predict(self, img, prob=0.5, row=100, col=100, alph=50):
        """
        获取单元格
        """
        img = Image.fromarray(img[:, :, (2, 1, 0)])
        w, h = self.size
        rows_lines, cols_lines = self._get_table_rowcols(img, prob, row, col)
        new_rows_lines, new_cols_lines = adjust_lines(
            rows_lines, cols_lines, alph=alph)

        rows_lines = new_rows_lines+rows_lines
        cols_lines = cols_lines+new_cols_lines

        nrow = len(rows_lines)
        ncol = len(cols_lines)

        for i in range(nrow):
            for j in range(ncol):

                rows_lines[i] = line_to_line(rows_lines[i], cols_lines[j], 32)
                cols_lines[j] = line_to_line(cols_lines[j], rows_lines[i], 32)

        tmp = np.zeros((img.size[1], img.size[0]), dtype='uint8')
        tmp = draw_lines(tmp, cols_lines+rows_lines, color=255, lineW=1)

        tabel_labels = measure.label(tmp == 0, connectivity=2)
        regions = measure.regionprops(tabel_labels)
        rboxes = []
        for region in regions:
            if region.bbox_area < h*w-10:
                rbox = min_area_rect_box(region.coords)
                rboxes.append(rbox)

        return rboxes, cols_lines, rows_lines
