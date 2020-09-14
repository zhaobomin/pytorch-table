#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import table_model
import os
import sys
from PIL import Image
import numpy as np
import cv2

if __name__ == '__main__':

    net = table_model.TABLE_NET()

    img = cv2.imread(sys.argv[1])
    h, w, c = img.shape

    rboxes, cols_lines, rows_lines = net.predict(
        img, prob=0.5, row=10, col=10, alph=10)

    box_img = net.draw_boxes(img, rboxes, (0, 255, 0))

    cv2.imwrite('test/pred_box.jpg', box_img)

    lines_img = np.zeros((h, w), dtype='uint8')
    lines_img = net.draw_lines(lines_img, cols_lines+rows_lines,
                               color=255, lineW=1)
    cv2.imwrite('test/pred_seg.png', lines_img)
