#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:49:54 2019
image
@author: chineseocr
"""
from scipy.spatial import distance as dist
from PIL import Image
import numpy as np
import cv2


def draw_boxes(im, bboxes, color=(0, 0, 0)):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w,  = im.shape[:2]
    thick = int((h + w) / 300)
    i = 0
    for box in bboxes:

        x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        cx = np.mean([x1, x2, x3, x4])
        cy = np.mean([y1, y2, y3, y4])
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)),
                 c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)),
                 c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)),
                 c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)),
                 c, 1, lineType=cv2.LINE_AA)
        mess = str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h, c, thick // 2)
        i += 1
    return tmp


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w = im.shape[:2]
    i = 0
    for box in bboxes:
        x1, y1, x2, y2 = box
        tmp = cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)),
                       c, lineW, lineType=cv2.LINE_AA)
        i += 1
    return tmp


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)
    fx = new_w/image_w
    fy = new_h/image_h
    dx = (w-new_w)//2
    dy = (h-new_h)//2

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, (dx, dy))
    return boxed_image, fx, fy, dx, dy


def min_area_line(coords):

    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = sort_box(box)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = solve(box)
    if w < h:
        xmin = (x1+x2)/2
        xmax = (x3+x4)/2
        ymin = (y1+y2)/2
        ymax = (y3+y4)/2

    else:
        xmin = (x1+x4)/2
        xmax = (x2+x3)/2
        ymin = (y1+y4)/2
        ymax = (y2+y3)/2

    return [xmin, ymin, xmax, ymax]


def min_area_rect_box(coords):
    """
    多边形外接矩形
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()
    box = sort_box(box)
    return box


def sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    return x1, y1, x2, y2, x3, y3, x4, y4


def _order_points(pts):
    # 根据x坐标对点进行排序
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 从排序中获取最左侧和最右侧的点
    # x坐标点
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # 现在，根据它们的y坐标对最左边的坐标进行排序，这样我们就可以分别抓住左上角和左下角
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # 现在我们有了左上角坐标，用它作为锚来计算左上角和右上角之间的欧氏距离;
    # 根据毕达哥拉斯定理，距离最大的点将是我们的右下角
    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    # 返回左上角，右上角，右下角和左下角的坐标
    return np.array([tl, tr, br, bl], dtype="float32")


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1+x3+x2+x4)/4.0
    cy = (y1+y3+y4+y2)/4.0
    w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
    h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2
    #x = cx-w/2
    #y = cy-h/2
    sinA = (h*(x1-cx)-w*(y1 - cy))*1.0/(h*h+w*w)*2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


#####################直线处理#####################

def fit_line(p1, p2):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2-y1
    B = x1-x2
    C = x2*y1-x1*y2
    return A, B, C


def sqrt(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def point_to_points(p, points, alpha=10):
    # 点到点之间的距离
    sqList = [sqrt(p, point) for point in points]
    if max(sqList) < alpha:
        return True
    else:
        return False


def point_line_cor(p, A, B, C):
    # 判断点与之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A*x+B*y+C
    return r


def line_to_line(points1, points2, alpha=10):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    A1, B1, C1 = fit_line((x1, y1), (x2, y2))
    A2, B2, C2 = fit_line((ox1, oy1), (ox2, oy2))
    flag1 = point_line_cor([x1, y1], A2, B2, C2)
    flag2 = point_line_cor([x2, y2], A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):

        x = (B1*C2-B2*C1)/(A1*B2-A2*B1)
        y = (A2*C1-A1*C2)/(A1*B2-A2*B1)
        p = (x, y)
        r0 = sqrt(p, (x1, y1))
        r1 = sqrt(p, (x2, y2))

        if min(r0, r1) < alpha:

            if r0 < r1:
                points1 = [p[0], p[1], x2, y2]
            else:
                points1 = [x1, y1, p[0], p[1]]

    return points1


def adjust_lines(RowsLines, ColsLines, alph=50):
    # 调整line
    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):

        x1, y1, x2, y2 = RowsLines[i]
        cx1, cy1 = (x1+x2)/2, (y1+y2)/2
        for j in range(nrow):
            if i != j:
                x3, y3, x4, y4 = RowsLines[j]
                cx2, cy2 = (x3+x4)/2, (y3+y4)/2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x2, y2, x4, y4])

    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        cx1, cy1 = (x1+x2)/2, (y1+y2)/2
        for j in range(ncol):
            if i != j:
                x3, y3, x4, y4 = ColsLines[j]
                cx2, cy2 = (x3+x4)/2, (y3+y4)/2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newColsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newColsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newColsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newColsLines.append([x2, y2, x4, y4])

    return newRowsLines, newColsLines
