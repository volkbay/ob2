import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import random

def show_img(img):
    plt.imshow(img)
    plt.show()

def find_alpha_comp (alpha, beta):
    return alpha + np.multiply(beta, 1-alpha).astype('float32')

def find_color_comp (ca, cb, aa, ab):
    trans_mat = find_alpha_comp(aa, ab)
    nom = np.multiply(ca, aa) + np.multiply(np.multiply(cb, ab), (1 - aa))
    return np.divide(nom, trans_mat, out=np.zeros_like(nom), where=trans_mat!=0)

folder = 'FINAL_OB2'
folder_in = 'INPUT_MAIN'
folder_neg = 'FINAL_NEG'
folder_res = 'FINAL_RESULT'
imList = glob.glob('./{}/res*'.format(folder))
kernel = np.ones((7, 7),  dtype='float32')
cl = cv2.createCLAHE(2,(8, 8))

for out_no in range(1, 11):
    shape_list = list(range(1, 21))+list(range(26, 32))+list(range(37, 41))+[62, 63, 70, 71]
    shape_list = np.sort((np.random.choice(shape_list, 10, False)))
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

    for ind, file in enumerate(file_list):
        IMG_SIZE = np.array([1, 1]) * np.random.randint(750, 2000)
        all_255 = np.ones(IMG_SIZE,  dtype='float32')
        pt = (np.random.randint(0, 4960 - IMG_SIZE[1]), np.random.randint(0, 3507 - IMG_SIZE[0]))

        raw_in = cv2.imread("./{}/input{}.png".format(folder_in, shape_list[ind]))
        raw_neg = cv2.imread("./{}/res{}.png".format(folder_neg, file))
        raw = cv2.imread("./{}/res{}.png".format(folder, file))
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
        ch_r_in = find_color_comp(mask_in * chroma[0], clipped_in[:, :, 2], neg_in, clipped_in[:, :, 3])
        ch_g_in = find_color_comp(mask_in * chroma[1], clipped_in[:, :, 1], neg_in, clipped_in[:, :, 3])
        ch_b_in = find_color_comp(mask_in * chroma[2], clipped_in[:, :, 0], neg_in, clipped_in[:, :, 3])
        ch_a_in = find_alpha_comp(neg_in, clipped_in[:, :, 3])

        ch_r_neg = find_color_comp(mask_neg * chroma[0], clipped_neg[:, :, 2], neg_neg, clipped_neg[:, :, 3])
        ch_g_neg = find_color_comp(mask_neg * chroma[1], clipped_neg[:, :, 1], neg_neg, clipped_neg[:, :, 3])
        ch_b_neg = find_color_comp(mask_neg * chroma[2], clipped_neg[:, :, 0], neg_neg, clipped_neg[:, :, 3])
        ch_a_neg = find_alpha_comp(neg_neg, clipped_neg[:, :, 3])

        ch_r = find_color_comp(mask* chroma[0], clipped[:, :, 2], neg, clipped[:, :, 3])
        ch_g = find_color_comp(mask* chroma[1], clipped[:, :, 1], neg, clipped[:, :, 3])
        ch_b = find_color_comp(mask* chroma[2], clipped[:, :, 0], neg, clipped[:, :, 3])
        ch_a = find_alpha_comp(neg, clipped[:, :, 3])

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

    white = 255 * white
    white_in = 255 * white_in
    white_neg = 255 * white_neg

    cv2.imwrite("./{}/final_out{}.png".format(folder_res, out_no), white)
    cv2.imwrite("./{}/final_neg{}.png".format(folder_res, out_no), white_neg)
    cv2.imwrite("./{}/final_in{}.png".format(folder_res, out_no), white_in)