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

def histogramBackProjection(dir_name):
    read_file_count = 0
    file_list = os.listdir(dir_name)
    for file in file_list:
        if file.endswith('png') | file.endswith('jpg'):
            file_path = dir_name + file
            img = imread(file_path, IMREAD_COLOR)
            if len(np.shape(img)) != 3:
                continue
            hsv_img = cvtColor(img, COLOR_BGR2HSV)
            hist_img(hsv_img)
            read_file_count += 1
        # if file.endswith('avi') | file.endswith('mp4'):
        #     cap = VideoCapture(file)
        #     can_readable, frame = cap.read()
        #     while can_readable:
        #         hsv_img = cvtColor(frame, COLOR_BGR2HSV)
        #         histImg(hsv_img)
        #         read_file_count += 1
        #         can_readable, frame = cap.read()
    return read_file_count


def hsv_quantize_img(hsv_img, quantize_level = 64):
    # https://docs.opencv.org/3.3.1/dc/df6/tutorial_py_histogram_backprojection.html
    print(np.shape(hsv_img))
    np.ndarray()
    hsv_img


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    # sys.path.insert(cur_dir + "/../")
    from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QInputDialog, QApplication


    trustGroupCount = histogramBackProjection('./trustGroup/')
    expFileGroup = os.listdir('./experimentalGroup/')
    print('generate dataset with %d experimentalGroup, %d trustGroup, trustGroup Size is (?,?)'
          % (len(expFileGroup), trustGroupCount))

    # app = QApplication([])
    # dialog = QInputDialog()
    # dialog.show()
    # app.exec_()
    # print(dialog.textValue())
