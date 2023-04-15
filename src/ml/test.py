import cv2
import numpy as np
import glob
import os
import shutil

input_list = [  1,    8,   18,   29,   44,   59,   71,   84,   99,  113,
              121,  133,  144,  154,  164,  175,  186,  198,  211,  224,
              235,  245,  256,  262,  272,  281,  293,  301,  309,  319,  329,
              337,  345,  347,  361,  371,  386,  396,  411,  421,  430,
              439,  450,  458,  477,  485,  502,  517,  524,  535,  554,
              574,  592,  601,  616,  631,  641,  652,  666,  679,  693,
              703,  713,  728,  738,  751,  767,  782,  796,  811,  821,
              832,  841,  849,  861,  871,  880,  891,  901,  912,  922,
              933,  944,  957,  966,  980,  991, 1007, 1021, 1031, 1043,
             1053, 1061, 1071, 1081, 1091, 1101, 1111, 1121, 1131]

IN_HEIGHT = 240
IN_WIDTH = 320
BUFFER_SIZE = 1000
BATCH_SIZE = 5
FOLDER = './crop{}/*'.format(str(IN_HEIGHT))
imList = glob.glob(FOLDER)

for file in imList:
    str = os.path.split(file)
    index = str[-1][7:-4]  # Take file number
    if int(index) in input_list:
        shutil.copyfile(file, "./INPUT_MAIN/input{}.png".format(input_list.index(int(index))+1))


