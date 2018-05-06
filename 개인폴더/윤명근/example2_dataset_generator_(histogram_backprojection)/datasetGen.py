import os
import sys
from cv2 import *
import numpy as np


# dataset generate
#   20180429-210400 폴더 안에
#   image000, label000 ~~~ 데이터를 넣음.
#   label 에는 한 줄에 하나씩 rect 에 대한 정보. (왼쪽상단, 우측하단 point2개)
#
# gui
#   이전으로 가기 (image 에 대한 정보를 덮어씌울 수 있어야 함)
#   data 판단하기
#   종료하기 / 자동종료
#       이어하기?
#       임의 지정해서 데이터 만들기? (rect만 해주면 되니깐)
#       신뢰도 설정? (신뢰도 넘어가버리는 애들에 대해서 다시 검증)
#
def multipleHistogramBackProjection(dir_name):
    read_file_count = 0
    file_list = os.listdir(dir_name)
    quantize_level = 64
    ret = [[0] * quantize_level] * quantize_level  # np.ndarray 로 해야 += 연산자 제대로 될 듯
    for file in file_list:
        if file.endswith('png') | file.endswith('jpg'):
            file_path = dir_name + file
            img = imread(file_path, IMREAD_COLOR)
            if len(np.shape(img)) != 3:
                continue
            hsv_img = cvtColor(img, COLOR_BGR2HSV)
            ret += quantizedHsHistogram(hsv_img)  # 문제있는 부분
            read_file_count += 1
        # if file.endswith('avi') | file.endswith('mp4'):
        #     cap = VideoCapture(file)
        #     can_readable, frame = cap.read()
        #     while can_readable:
        #         hsv_img = cvtColor(frame, COLOR_BGR2HSV)
        #         histImg(hsv_img)
        #         read_file_count += 1
        #         can_readable, frame = cap.read()
    ret /= read_file_count
    return ret, read_file_count


def quantizedHsHistogram(hsv_img, quantize_level=64):
    # https://docs.opencv.org/3.3.1/dc/df6/tutorial_py_histogram_backprojection.html
    # np.ndarray()
    # hsv 에서 hs 만 가져옴.
    ret = [[0] * quantize_level] * quantize_level  # np.ndarray 로 해야 나누기 될 듯
    splitHSV = split(hsv_img)  # split 함수
    for y in hsv_img.rows:
        for x in hsv_img.cols:
            qH = quantize(splitHSV.at(y, x)[0])
            qS = quantize(splitHSV.at(y, x)[1])
            # qV = quantize(splitHSV.at(y, x)[2])
            ret[qH][qS] += 1
    ret /= (hsv_img.rows * hsv_img.cols)
    return ret


def quantize(value, quantize_level=64):
    return value / 255.0 * (quantize_level - 1)


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    # sys.path.insert(cur_dir + "/../")
    from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QInputDialog, QApplication


    quantize_level = 64
    hist_trustability = 0.2  # 1이 최대, 신뢰도 threshold.
    squareAreaSize = 10
    squareAreaShiftSize = int(squareAreaSize / 2)
    squareAreaThreshold = 0.5
    # 픽셀의 히스토그램 신뢰도가 0.2차이가 나는 애들이 50% 일 때 해당 square를 검출함.
    # square 는 areaSize 의 반씩 시프트함.


    hs_histogram, trustGroupCount = multipleHistogramBackProjection('./trustGroup/', quantize_level)
    experimentalGroup = os.listdir('./experimentalGroup/')
    print('generate dataset with %d experimentalGroup, %d trustGroup, trustGroup Size is (?,?)'
          % (len(experimentalGroup), trustGroupCount))


    for file in experimentalGroup:
        img = experimentalGroup[file]
        padding = int(squareAreaSize / 2)
        for y in range(padding, img.rows-padding, squareAreaShiftSize):
            for x in range(padding, img.cols-padding, squareAreaShiftSize):
                hs_val = img.at(y,x)
                hQ_val = quantize(hs_val['h'], quantize_level)
                sQ_val = quantize(hs_val['s'], quantize_level)
                c = 0
                for yy in range(y-squareAreaSize, y+squareAreaSize):
                    for xx in range(x - squareAreaSize, x + squareAreaSize):
                        if hist_trustability <= hs_histogram[hQ_val][sQ_val]:
                            c += 1
                areaTrustability = c / (squareAreaSize**2)
                if squareAreaThreshold <= areaTrustability:
                    print(y & "," & x & " : value("&areaTrustability&")")
                    # y-squareAreaSize ~ y+squareAreaSize-1
                    # x-squareAreaSize ~ x+squareAreaSize-1
                    # rects are detected back projection area

    # app = QApplication([])
    # dialog = QInputDialog()
    # dialog.show()
    # app.exec_()
    # print(dialog.textValue())
