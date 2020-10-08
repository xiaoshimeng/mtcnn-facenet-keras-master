import sys
from operator import itemgetter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# .......................#
"""
    将图片按一定比例缩放
"""


# .......................#
def caculateScales(img):
    copy_img = img.copy()

    pr_scale = 1.0
    h, w, _ = copy_img.shape

    if min(h, w) > 500:
        pr_scale = 500.0 / min(h, w)
        h = int(h * pr_scale)
        w = int(w * pr_scale)
    elif max(h, w) < 500:
        pr_scale = 500.0 / max(h, w)
        h = int(h * pr_scale)
        w = int(w * pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1

    return scales


# .......................#
"""
    对Pnet处理后的结果进行精修处理
"""


# .......................#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    stride = 0
    """stride 略等于2"""
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    (x, y) = np.where(cls_prob >= threshold)

    boundingbox = np.array([x, y]).T
    """找到对应原图的位置"""
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    # plt.scatter(bb1[:, 0], bb1[:, 1], linewidths=1)
    # plt.scatter(bb2[:, 0], bb2[:, 1], linewidths=1, c='r')
    # plt.show()
    boundingbox = np.concatenate((bb1, bb2), axis=1)

    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]
    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T

    boundingbox = boundingbox + offset * 12.0 * scale

    rectangles = np.concatenate((boundingbox, score), axis=1)

    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)


# .......................#
"""
    将长方形调整为正方形
"""


# .......................#
def rect2square(rectangeles):
    w = rectangeles[:, 2] - rectangeles[:, 0]
    h = rectangeles[:, 3] - rectangeles[:, 1]
    l = np.maximum(w, h).T
    rectangeles[:, 0] = rectangeles[:, 0] + w * 0.5 - l * 0.5
    rectangeles[:, 1] = rectangeles[:, 1] + h * 0.5 - l * 0.5
    rectangeles[:, 2:4] = rectangeles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangeles


# .......................#
"""
    非极大抑制
"""


# .......................#
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# .......................#
"""
    对Pnet处理后的结果进行处理
"""


# .......................#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)

    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)


# .......................#
"""
    对Rnet的结果进行处理
"""


# .......................#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)

    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)


# .......................#
"""
    人脸对齐
"""


# .......................#
def Alignment_1(img, landemark):
    if landemark.shape[0] == 68:
        x = landemark[32, 0] - landemark[45, 0]
        y = landemark[32, 1] - landemark[45, 1]
    elif landemark.shape[0] == 5:
        x = landemark[0, 0] - landemark[1, 0]
        y = landemark[0, 1] - landemark[1, 1]
    # 眼睛连线相对于水平倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算弧度值
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))
    RotationMatrix = np.array(RotationMatrix)

    new_landmark = []
    for i in range(landemark.shape[0]):
        pts = []
        pts.append(
            RotationMatrix[0, 0] * landemark[i, 0] + RotationMatrix[0, 1] * landemark[i, 1] + RotationMatrix[0, 2])
        pts.append(
            RotationMatrix[1, 0] * landemark[i, 0] + RotationMatrix[1, 1] * landemark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)
    return new_img, new_landmark


# .......................#
"""
    图片预处理
    高斯归一化
"""


# .......................#
def pre_process(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


# .......................#
"""
    L2标准化
"""


# .......................#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


# .......................#
"""
    计算128特征值
"""


# .......................#
def calc_128_vec(model, img):
    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre, [128])
    return pre


# .......................#
"""
    计算人脸距离
"""


# .......................#
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# .......................#
"""
    人脸比对
"""


# .......................#
def compare_faces(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    dis = face_distance(known_face_encoding, face_encoding_to_check)
    return list(dis <= tolerance)
