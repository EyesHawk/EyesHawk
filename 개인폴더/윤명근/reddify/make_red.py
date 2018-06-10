import cv2 as cv
import os
import numpy as np

# make red image at hsv color space [hue shifting : 342~360, 0~18]

if __name__ == '__main__':
    dir_path = 'C:/Users/YOON/Desktop/noblood/'
    file_list = os.listdir(dir_path)
    save_path = 'C:/Users/YOON/Desktop/noblood/output/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in file_list:
        print('read >> ' + dir_path + file)
        img = cv.imread(dir_path + file)
        if img is None:
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_split = cv.split(img)

        hue = img_split[0]
        over_90_hue_mask = np.uint8(90 < hue)
        over_90_hue = cv.bitwise_and(hue, hue, mask=over_90_hue_mask)
        under_90_hue_mask = np.uint8(hue <= 90)
        under_90_hue = cv.bitwise_and(hue, hue, mask=under_90_hue_mask)

        _, over_max, _, _ = cv.minMaxLoc(over_90_hue)
        _, under_max, _, _ = cv.minMaxLoc(under_90_hue)
        if over_max == 0:
            over_max = 1
        if under_max == 0:
            under_max = 1
        reddfiy_over_90_hue = over_90_hue * (18 / over_max)
        reddfiy_under_90_hue = under_90_hue * (18 / under_max) + 342

        reddify_hue = np.uint8(reddfiy_over_90_hue + reddfiy_under_90_hue)

        img_split[0] = reddify_hue
        reddify_img = cv.merge(img_split)

        cv.imwrite(save_path + file, reddify_img)
        print('save << ' + save_path + file + '\n')
    print('success')
