import cv2
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils

from net.inception import InceptionResNetV1

img = cv2.imread('face_dataset/LGF.jpg')

# 创建mtcnn对象
mtcnn_model = mtcnn()
# 设置门限值
threshold = [0.5, 0.7, 0.9]
#检测人脸
rectangles = mtcnn_model.detcet(img, threshold)

draw = img.copy()
#转化为正方形
rectangles = utils.rect2square(np.array(rectangles))
print(rectangles.shape)

#载入facenet
facenet_model = InceptionResNetV1()
#model.summary()
model_path = 'model/facenet_keras.h5'
facenet_model.load_weights(model_path)

for rectangle in rectangles:
    if rectangle is not None:
        landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (rectangle[3] - rectangle[1]) * 160

        crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        crop_img = cv2.resize(crop_img, (160, 160))
        cv2.imshow('before', crop_img)
        new_img, _ = utils.Alignment_1(crop_img, landmark)
        cv2.imshow('two eyes', new_img)

        new_img = np.expand_dims(new_img, 0)
        feature1 = utils.calc_128_vec(facenet_model, new_img)
        print(feature1)

cv2.waitKey(0)
