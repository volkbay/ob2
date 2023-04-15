import cv2
import glob
import os

imList = glob.glob('./crop/*')

for file in imList:
    str = os.path.split(file)
    index = str[-1][7:-4] # Take file number
    if True:
        raw = cv2.imread(file)
        img = cv2.resize(raw, (320, 240), cv2.INTER_CUBIC)
        cv2.imwrite("./crop240/resized{}.png".format(index), img)