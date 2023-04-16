#from subprocess import call
#call(["pyside2-uic", "/home/volkan/proje/qt/ob2_basic/form.ui", "-o", "/home/volkan/proje/qt/ob2_basic/window.py"])

# This Python file uses the following encoding: utf-8
import sys
import os
import window
import numpy as np
import cv2

import win32api
import win32print

from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QFile, Slot, QSize
from PySide2.QtGui import QPixmap, QIcon



class ob2_main(QMainWindow, window.Ui_OB2):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.create_images)
        self.pushButton.clicked.connect(self.print_images)
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.shapes = range(1, 100)
        self.no_img = 20
        image1 = QPixmap("{}/{}/create.png".format(self.path, "icon"))
        icon1 = QIcon(image1)
        self.pushButton_2.setIcon(icon1)
        self.pushButton_2.setIconSize(QSize(500, 500))
        image2 = QPixmap("{}/{}/print.png".format(self.path, "icon"))
        icon2 = QIcon(image2)
        self.pushButton.setIcon(icon2)
        self.pushButton.setIconSize(QSize(300, 300))
        self.file2print = ""
        print("INIT done.")

    def find_alpha_comp (self, alpha, beta):
        return alpha + np.multiply(beta, 1-alpha).astype('float32')

    def find_color_comp (self, ca, cb, aa, ab):
        trans_mat = self.find_alpha_comp(aa, ab)
        nom = np.multiply(ca, aa) + np.multiply(np.multiply(cb, ab), (1 - aa))
        return np.divide(nom, trans_mat, out=np.zeros_like(nom), where=trans_mat!=0)

    @Slot()
    def create_images(self):
        print("Create click.")
        folder_in = 'INPUT_MAIN'
        folder_res = 'FINAL_RESULT'

        shape_list = self.shapes
        shape_list = np.sort((np.random.choice(shape_list, self.no_img, False)))

        white_in = np.ones((3508, 4961, 4), dtype='float32')
        white_in[:, :, 3] = 0

        background = np.ones((3508, 4961), dtype='float32')

        for ind, shp in enumerate(shape_list):
            IMG_SIZE = np.array([1, 1]) * np.random.randint(750, 2000)
            all_255 = np.ones(IMG_SIZE,  dtype='float32')
            pt = (np.random.randint(0, 4960 - IMG_SIZE[1]), np.random.randint(0, 3507 - IMG_SIZE[0]))

            raw_in = cv2.imread("{}\\{}\\input{}.png".format(self.path, folder_in, shp))
            img_in = cv2.cvtColor(raw_in, cv2.COLOR_BGR2GRAY)
            img_in = cv2.resize(img_in, (IMG_SIZE[0], IMG_SIZE[1]), cv2.INTER_CUBIC)
            img_in = (img_in / 255.0)

            neg_in = (1 - img_in)
            neg_in = np.minimum(neg_in, all_255).astype('float32')

            mask_in = np.array(neg_in)
            mask_in[mask_in != 0] = 1
            clipped_in = white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], :]

            chroma = np.random.rand(3)
            ch_r_in = self.find_color_comp(mask_in * chroma[0], clipped_in[:, :, 2], neg_in, clipped_in[:, :, 3])
            ch_g_in = self.find_color_comp(mask_in * chroma[1], clipped_in[:, :, 1], neg_in, clipped_in[:, :, 3])
            ch_b_in = self.find_color_comp(mask_in * chroma[2], clipped_in[:, :, 0], neg_in, clipped_in[:, :, 3])
            ch_a_in = self.find_alpha_comp(neg_in, clipped_in[:, :, 3])

            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 0] = ch_b_in
            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 1] = ch_g_in
            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 2] = ch_r_in
            white_in[pt[1]:pt[1] + IMG_SIZE[1], pt[0]:pt[0] + IMG_SIZE[0], 3] = ch_a_in

        chroma = np.random.rand(3)
        white_in[:, :, 2] = self.find_color_comp(white_in[:, :, 2], background * chroma[0], white_in[:, :, 3], background)
        white_in[:, :, 1] = self.find_color_comp(white_in[:, :, 1], background * chroma[1], white_in[:, :, 3], background)
        white_in[:, :, 0] = self.find_color_comp(white_in[:, :, 0], background * chroma[2], white_in[:, :, 3], background)
        white_in[:, :, 3] = background

        white_in = 255 * white_in
        print("CREATE saving.")
        out_no = len(os.listdir("{}\\{}".format(self.path, folder_res))) + 1
        self.file2print = "{}\\{}\\final_in{}.png".format(self.path, folder_res, out_no)
        cv2.imwrite(self.file2print, white_in)
        print("CREATE done.")

    @Slot()
    def print_images(self):
        print("PRINT click.")
        print(self.file2print)
        print('/d:"%s"' % win32print.GetDefaultPrinter())

        win32api.ShellExecute(
            0,
            "print",
            self.file2print,
            #
            # If this is None, the default printer will
            # be used anyway.
            #
            '/d:"%s"' % win32print.GetDefaultPrinter(),
            ".",
            0
        )

        print("PRINT done.")


if __name__ == "__main__":
    app = QApplication([])
    widget = ob2_main()
    widget.show()
    sys.exit(app.exec_())
