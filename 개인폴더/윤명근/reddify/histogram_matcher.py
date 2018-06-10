import cv2 as cv
import os
import numpy as np
import sys


hist_threshold = 0.0001
# os.path.dirname(__file__)
hist_src_file_name = 'C:/Users/YOON/Desktop/noblood/hist/blood_hand.jpg'
dir_path = 'C:/Users/YOON/Desktop/noblood/'  # 'C:/Users/YOON/Pictures/'
save_path = 'C:/Users/YOON/Desktop/noblood/output/'


def calc_hist_cdf_3ch(image: np.ndarray) -> list:

    hist_cdf_3ch = [np.array([[0]*256], dtype=np.float64)]*3

    for i in range(3):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        min_val, max_val, _, _ = cv.minMaxLoc(hist)
        if max_val == 0:
            max_val = 1
        cv.normalize(hist, hist, min_val / max_val, 1.0, cv.NORM_MINMAX)

        hist_cdf_3ch[i] = hist.copy()
        for pixel in range(1, 256):
            hist_cdf_3ch[i][pixel] += hist_cdf_3ch[i][pixel-1]

        min_val, max_val, _, _ = cv.minMaxLoc(hist_cdf_3ch[i])
        if max_val == 0:
            max_val = 1
        cv.normalize(hist_cdf_3ch[i], hist_cdf_3ch[i], min_val / max_val, 1.0, cv.NORM_MINMAX)

    return hist_cdf_3ch


def hist_match_3ch(target_image: np.ndarray, source_image: np.ndarray) -> np.ndarray:
    source_hist_cdf_3ch = calc_hist_cdf_3ch(source_image)
    matched_target_3ch = [0]*3

    target_hist_cdf_3ch = calc_hist_cdf_3ch(target_image)

    target_image_3ch = cv.split(target_image)

    for ch in range(3):
        lookup_table = np.array([0] * 256, dtype=np.uint8)
        before_idx = 0
        for p in range(256):
            source_cdf_idx = target_hist_cdf_3ch[ch][p]
            for q in range(before_idx, 256):
                target_cdf_idx = source_hist_cdf_3ch[ch][q]
                if (abs(target_cdf_idx - source_cdf_idx) < hist_threshold) | (source_cdf_idx < target_cdf_idx):
                    lookup_table[p] = q
                    before_idx = q
                    break
        matched_target_3ch[ch] = cv.LUT(target_image_3ch[ch], lookup_table)

    matched_target = cv.merge(matched_target_3ch)

    return matched_target


if __name__ == '__main__':
    file_list = os.listdir(dir_path)

    src_img = cv.imread(hist_src_file_name)
    if src_img is None:
        raise FileNotFoundError('hist src file \'' + hist_src_file_name + '\' not found')
    cv.imshow('src', src_img)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for file in file_list:
        print('read >> ' + dir_path + file)
        img = cv.imread(dir_path + file)
        if img is None:
            continue

        matched_img = hist_match_3ch(img, src_img)

        cv.imwrite(save_path + file, matched_img)
        print('save << ' + save_path + file + '\n')
    print('success')
