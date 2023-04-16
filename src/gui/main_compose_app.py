#from subprocess import call
#call(["pyside2-uic", "/home/volkan/proje/qt/ob2_basic/form.ui", "-o", "/home/volkan/proje/qt/ob2_basic/window.py"])

# This Python file uses the following encoding: utf-8
import sys
import os
import window
import numpy as np
import cv2
import random
import math

from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QFile, Slot, QSize
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QIcon



class ob2_main(QMainWindow, window.Ui_OB2):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.create_images)
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.shapes = np.array(list(range(1, 21))+list(range(26, 32))+list(range(37, 41))+[62, 63, 70, 71])
        self.no_img = 20
        image1 = QPixmap("{}/{}/create.png".format(self.path, "icon"));
        icon1 = QIcon(image1);
        self.pushButton_2.setIcon(icon1);
        self.pushButton_2.setIconSize(QSize(500, 500))
        image2 = QPixmap("{}/{}/print.png".format(self.path, "icon"));
        icon2 = QIcon(image2);
        self.pushButton.setIcon(icon2);
        self.pushButton.setIconSize(QSize(300, 300))

    def find_alpha_comp (self, alpha, beta):
        return alpha + np.multiply(beta, 1-alpha).astype('float32')

    def find_color_comp (self, ca, cb, aa, ab):
        trans_mat = self.find_alpha_comp(aa, ab)
        nom = np.multiply(ca, aa) + np.multiply(np.multiply(cb, ab), (1 - aa))
        return np.divide(nom, trans_mat, out=np.zeros_like(nom), where=trans_mat!=0)

    @Slot()
    def create_images(self):
        folder = 'FINAL_OB2'
        folder_in = 'INPUT_MAIN'
        folder_neg = 'FINAL_NEG'
        folder_res = 'FINAL_RESULT'
        kernel = np.ones((7, 7),  dtype='float32')
        cl = cv2.createCLAHE(2, (8, 8))

        shape_list = self.shapes
        shape_list = np.sort((np.random.choice(shape_list, self.no_img , False)))

        file_list = []
        for shape in shape_list:
            idx = random.choice(range(1, 21))
            file_list.append('{}_{}'.format(shape, idx))

        white = np.ones((3508, 4961, 4), dtype='float32')
        white[:, :, 3] = 0

        white_in = np.ones((3508, 4961, 4), dtype='float32')
        white_in[:, :, 3] = 0

        white_neg = np.ones((3508, 4961, 4), dtype='float32')
        white_neg[:, :, 3] = 0

        background = np.ones((3508, 4961), dtype='float32')

        for ind, file in enumerate(file_list):
            IMG_SIZE = np.array([1, 1]) * np.random.randint(750, 2000)
            all_255 = np.ones(IMG_SIZE,  dtype='float32')
            pt = (np.random.randint(0, 4960 - IMG_SIZE[1]), np.random.randint(0, 3507 - IMG_SIZE[0]))

            raw_in = cv2.imread("{}/{}/input{}.png".format(self.path, folder_in, shape_list[ind]))
            raw_neg = cv2.imread("{}/{}/res{}.png".format(self.path, folder_neg, file))
            raw = cv2.imread("{}/{}/res{}.png".format(self.path, folder, file))
            img_in = cv2.cvtColor(raw_in, cv2.COLOR_BGR2GRAY)
            img_in = cv2.resize(img_in, (IMG_SIZE[0], IMG_SIZE[1]), cv2.INTER_CUBIC)
            img_in = (img_in / 255.0)
            img_neg = cv2.cvtColor(raw_neg, cv2.COLOR_BGR2GRAY)
            img_neg = cv2.resize(img_neg, (IMG_SIZE[0], IMG_SIZE[1]), cv2.INTER_CUBIC)
            img_neg = (img_neg / 255.0)
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            img = cl.apply(img)
            img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]), cv2.INTER_CUBIC)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = (img/255.0)

            bright = (np.random.rand() + 1.5) # Random saturation
            neg_in = (1 - img_in)
            neg_in = np.minimum(neg_in, all_255).astype('float32')
            neg_neg = (1 - img_neg)*bright
            neg_neg = np.minimum(neg_neg, all_255).astype('float32')
            neg = (1 - img)*bright
            neg = np.minimum(neg, all_255).astype('float32')

            mask_in = np.array(neg_in)
            mask_in[mask_in != 0] = 1
            clipped_in = white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], :]
            mask_neg = np.array(neg_neg)
            mask_neg[mask_neg != 0] = 1
            clipped_neg = white_neg[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], :]
            mask = np.array(neg)
            mask[mask != 0] = 1
            clipped = white[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], :]

            chroma = np.random.rand(3)
            ch_r_in = self.find_color_comp(mask_in * chroma[0], clipped_in[:, :, 2], neg_in, clipped_in[:, :, 3])
            ch_g_in = self.find_color_comp(mask_in * chroma[1], clipped_in[:, :, 1], neg_in, clipped_in[:, :, 3])
            ch_b_in = self.find_color_comp(mask_in * chroma[2], clipped_in[:, :, 0], neg_in, clipped_in[:, :, 3])
            ch_a_in = self.find_alpha_comp(neg_in, clipped_in[:, :, 3])

            ch_r_neg = self.find_color_comp(mask_neg * chroma[0], clipped_neg[:, :, 2], neg_neg, clipped_neg[:, :, 3])
            ch_g_neg = self.find_color_comp(mask_neg * chroma[1], clipped_neg[:, :, 1], neg_neg, clipped_neg[:, :, 3])
            ch_b_neg = self.find_color_comp(mask_neg * chroma[2], clipped_neg[:, :, 0], neg_neg, clipped_neg[:, :, 3])
            ch_a_neg = self.find_alpha_comp(neg_neg, clipped_neg[:, :, 3])

            ch_r = self.find_color_comp(mask* chroma[0], clipped[:, :, 2], neg, clipped[:, :, 3])
            ch_g = self.find_color_comp(mask* chroma[1], clipped[:, :, 1], neg, clipped[:, :, 3])
            ch_b = self.find_color_comp(mask* chroma[2], clipped[:, :, 0], neg, clipped[:, :, 3])
            ch_a = self.find_alpha_comp(neg, clipped[:, :, 3])

            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 0] = ch_b_in
            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 1] = ch_g_in
            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 2] = ch_r_in
            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 3] = ch_a_in

            white_neg[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 0] = ch_b_neg
            white_neg[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 1] = ch_g_neg
            white_neg[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 2] = ch_r_neg
            white_neg[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 3] = ch_a_neg

            white[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 0] = ch_b
            white[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 1] = ch_g
            white[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 2] = ch_r
            white[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 3] = ch_a

        chroma = np.random.rand(3)
        white_in[:, :, 2] = self.find_color_comp(white_in[:, :, 2], background * chroma[0], white_in[:, :, 3], background)
        white_in[:, :, 1] = self.find_color_comp(white_in[:, :, 1], background * chroma[1], white_in[:, :, 3], background)
        white_in[:, :, 0] = self.find_color_comp(white_in[:, :, 0], background * chroma[2], white_in[:, :, 3], background)
        white_in[:, :, 3] = background

        white_neg[:, :, 2] = self.find_color_comp(white_neg[:, :, 2], background * chroma[0], white_neg[:, :, 3], background)
        white_neg[:, :, 1] = self.find_color_comp(white_neg[:, :, 1], background * chroma[1], white_neg[:, :, 3], background)
        white_neg[:, :, 0] = self.find_color_comp(white_neg[:, :, 0], background * chroma[2], white_neg[:, :, 3], background)
        white_neg[:, :, 3] = background

        white[:, :, 2] = self.find_color_comp(white[:, :, 2], background * chroma[0], white[:, :, 3], background)
        white[:, :, 1] = self.find_color_comp(white[:, :, 1], background * chroma[1], white[:, :, 3], background)
        white[:, :, 0] = self.find_color_comp(white[:, :, 0], background * chroma[2], white[:, :, 3], background)
        white[:, :, 3] = background

        white = 255 * white
        white_in = 255 * white_in
        white_neg = 255 * white_neg

        out_no = math.floor((len(os.listdir("{}/{}".format(self.path, folder_res))) / 3)+1)
        cv2.imwrite("{}/{}/final_out{}.png".format(self.path, folder_res, out_no), white)
        cv2.imwrite("{}/{}/final_neg{}.png".format(self.path, folder_res, out_no), white_neg)
        cv2.imwrite("{}/{}/final_in{}.png".format(self.path, folder_res, out_no), white_in)

if __name__ == "__main__":
    app = QApplication([])
    widget = ob2_main()
    widget.show()
    sys.exit(app.exec_())