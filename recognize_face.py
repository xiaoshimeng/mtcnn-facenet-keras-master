from net.mtcnn import mtcnn
from utils import utils
import os
from net.inception import InceptionResNetV1
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class face_rec():
    def __init__(self):
        #创建mtcnn对象
        #检测图片中的人脸
        self.mtcnn_model = mtcnn()
        #门限函数
        self.threshold = [0.5, 0.8, 0.9]

        #载入facenet
        #将检测到的人脸转为128维向量
        self.facenet_model = InceptionResNetV1()
        #model.summary()
        model_path = 'model/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        #--------------------------------------------#
        #   对数据库的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        # --------------------------------------------#

        face_list = os.listdir('face_dataset')

        self.known_face_encodings = []
        self.known_face_names = []
        for face in face_list:
            name = face.split('.')[0]

            img = cv2.imread('face_dataset/' + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #检测人脸
            rectangles = self.mtcnn_model.detcet(img, self.threshold)

            #转换成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            #facenet要传入一个168 * 168大小的图片
            rectangle = rectangles[0]
            #记录landmark
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                        rectangle[3] - rectangle[1]) * 160
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)
            new_img = np.expand_dims(new_img, 0)
            face_encodings = utils.calc_128_vec(self.facenet_model, new_img)

            self.known_face_encodings.append(face_encodings)
            self.known_face_names.append(name)

    def recognize(self, draw):
        #------------------------------#
        #   人脸识别
        #   先定位，再进行数据库的匹配
        # ------------------------------#
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        #检测人脸
        rectangles = self.mtcnn_model.detcet(draw_rgb, self.threshold)
        print(np.shape(rectangles))
        if len(rectangles) == 0:
            return
        #转换为正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)
        #--------------------------#
        #   对检测到的人脸进行编码
        #--------------------------#
        face_encodings = []
        for rectangle in rectangles:
            #获取landmark再小图中的坐标
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                        rectangle[3] - rectangle[1]) * 160
            #截取图像
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))
            #对齐
            new_img, _ = utils.Alignment_1(crop_img, landmark)
            new_img = np.expand_dims(new_img, 0)
            #利用facenet_model计算128维特征向量
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            #取出一张脸并于数据库中所有人脸进行比对，计算得分
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = 'Unknown'
            #找出人脸最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            #取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        #-----------------------------------------------#
        #   画框
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw

if __name__ == '__main__':
    face = face_rec()
    video = cv2.VideoCapture(0)

    while True:
        ret, draw = video.read()
        face.recognize(draw)
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


