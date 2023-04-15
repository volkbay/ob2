import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np

def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    return final_image.astype(np.uint8)

imList = glob.glob('./raw/*')

for file in imList:
    str = os.path.split(file)
    index = str[-1][9:-4] # Take file number
    if index == "1":
        raw = read_transparent_png(file)

        _, bw = cv2.threshold(raw,254,255,cv2.THRESH_BINARY_INV)

        nz  = np.nonzero(bw)
        box = [np.min(nz[0]), np.max(nz[0]), np.min(nz[1]), np.max(nz[1])]
        print(box)
        img = raw[box[0]:box[1],box[2]:box[3]]
        """contours = findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)"""
        """
        plt.imshow(img, cmap='gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()"""
        cv2.imwrite("./crop/cropped{}.png".format(index), img)
        break