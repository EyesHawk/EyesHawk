import os
import sys
# cur_dir = os.path.dirname(__file__)
# sys.path.insert(0, cur_dir+'/../')
# from DatasetGen.gui.window import MyWindow
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from ..module.image_processing import ImageProcessing as ip
import numpy as np
import cv2 as cv


class Interaction:
    def __init__(self, window, quantize_level: int, trust_dir_name: str, experimental_dir_name: str,
                 output_dir_name: str):
        self.window = window
        self.quantize_level = quantize_level
        self.trust_dir_name = trust_dir_name
        self.experimental_dir_name = experimental_dir_name
        self.output_dir_name = output_dir_name

        self.__hist__ = np.array([[] * quantize_level] * quantize_level)
        self.__trust_file_count__ = 0
        self.__exp_file_order__ = 0
        self.__exp_file_list__ = []
        self.__dirty__ = False  # QTextEdit 이 비어있으면 False
        self.__detection_list__ = [(0,) * 4] * len(self.__exp_file_list__)  # 이렇게 구성될 예정이며 당연히 [] 으로 초기화 됨.
        self.__detection_order__ = 0
        self.__detection_user_info_list__ = [0] * len(self.__exp_file_list__)  # x 버튼은 -1, o 버튼은 1, 나머지는 0으로 사용할 예정
        self.__current_image__ = np.array([[] * 0] * 0)
        self.__rect_drawn_image__ = np.array([[] * 0] * 0)

    def initialize(self):
        # connect 를 모두 한 다음에 호출해주세요.
        self.window.label.resize(400, 400)
        self.window.label.move(180, 25)
        self.__hist__, self.__trust_file_count__ = ip.multipleHistogramBackProjection(self.trust_dir_name,
                                                                                      self.quantize_level)
        self.__exp_file_list__ = os.listdir(self.experimental_dir_name)
        self.__load_images__(self.__exp_file_order__)

    def click_btn_o(self):
        if len(self.__detection_user_info_list__) <= self.__detection_order__:
            print('검출 데이터를 모두 계산하였습니다. save 후 다음 이미지를 눌러주세요.')
            return
        else:
            self.__dirty__ = True
            self.__detection_user_info_list__[self.__detection_order__] = 1
            self.window.textEdit.append('1,' +
                ','.join(str(tuple_data) for tuple_data in self.__detection_list__[self.__detection_order__]))
            self.__increment_detection_order__()

    def click_btn_x(self):
        if len(self.__detection_user_info_list__) <= self.__detection_order__:
            print('검출 데이터를 모두 계산하였습니다. save 후 다음 이미지를 눌러주세요.')
        else:
            self.__dirty__ = True
            self.__detection_user_info_list__[self.__detection_order__] = -1
            self.window.textEdit.append('0,' +
                                        ','.join(str(tuple_data) for tuple_data in
                                                 self.__detection_list__[self.__detection_order__]))
            self.__increment_detection_order__()

    def click_btn_refresh(self):
        self.__dirty__ = False
        self.window.textEdit.setText('')
        self.__refresh_detection_order__()

    def click_btn_save(self):
        file_name = self.output_dir_name + str(self.__exp_file_order__) + '.txt'
        f = open(file_name, 'w')
        f.write(self.window.textEdit.toPlainText())
        f.close()
        self.__dirty__ = False

    def click_prev_image(self):
        if self.__dirty__:
            print('save 또는 refresh 이후 prev 또는 next 이미지로 이동할 수 있습니다.')
            return
        if self.__exp_file_order__ <= 0:
            print('첫번째 이미지입니다.')
            return
        self.window.textEdit.setText('')
        self.__exp_file_order__ -= 1
        self.__load_images__(self.__exp_file_order__)

    def click_next_image(self):
        if self.__dirty__:
            print('save 또는 refresh 이후 prev 또는 next 이미지로 이동할 수 있습니다.')
            return
        if len(self.__exp_file_list__) - 1 <= self.__exp_file_order__:
            print('마지막 이미지입니다.')
            return
        self.window.textEdit.setText('')
        self.__exp_file_order__ += 1
        self.__load_images__(self.__exp_file_order__)

    def __load_images__(self, order: int):
        if self.__dirty__:
            if self.window.textEdit.toPlainText() != "":
                print('refresh 또는 save 를 눌러주세요.')
        else:
            file_path = self.experimental_dir_name + self.__exp_file_list__[order]
            self.__current_image__ = cv.imread(file_path, cv.IMREAD_COLOR)
            # self.__current_image__ = QPixmap(self.experimental_dir_name + self.__exp_file_list__[order])
            if self.__current_image__.size == 0:
                self.window.label.setText('이미지 로드 실패 - 불러올 수 없는 이미지 :\n' + self.__exp_file_list__[order])
            else:
                # todo : 넘 느림...
                self.__detection_list__ = ip.detect_from_hs_hist(self.__hist__, self.__current_image__,
                                                                 quantize_level=64, hist_trustability=0.005,
                                                                 square_area_size=20,
                                                                 square_area_shift_size=10, square_area_threshold=0.03
                                                                 )
                self.__detection_user_info_list__ = [0] * len(self.__detection_list__)
                self.__refresh_detection_order__()
                # self.window.label.resize(400, 400)
                # self.window.label.move(180, 25)

    def __increment_detection_order__(self):
        self.__detection_order__ += 1
        self.__set_detection_order__()
        pix_map = ip.ndarray_to_pixmap(self.__rect_drawn_image__)
        if self.__rect_drawn_image__.size != 0:
            cv.imshow("rect_drawn_image", self.__rect_drawn_image__)
        self.window.label.setPixmap(pix_map)

    def __refresh_detection_order__(self):
        self.__detection_user_info_list__ = [0] * len(self.__detection_list__)
        self.__detection_order__ = 0
        self.__set_detection_order__()
        pix_map = ip.ndarray_to_pixmap(self.__rect_drawn_image__)
        if self.__rect_drawn_image__.size != 0:
            cv.imshow("rect_drawn_image", self.__rect_drawn_image__)
        self.window.label.setPixmap(pix_map)

    def __set_detection_order__(self):
        if len(self.__detection_list__) == 0:
            return
        if len(self.__detection_list__) <= self.__detection_order__:
            last_idx = self.__detection_order__ - 1
            user_info = self.__detection_user_info_list__[last_idx]
            if user_info == 1:
                ip.drawRect(self.__rect_drawn_image__, self.__detection_list__[last_idx], 0, 255, 0)
            elif user_info == -1:
                ip.drawRect(self.__rect_drawn_image__, self.__detection_list__[last_idx], 255, 0, 0)
            ip.drawText(self.__rect_drawn_image__, 'done, save and change image.', (0, 400))
            return

        self.__rect_drawn_image__ = self.__current_image__.copy()
        for idx in range(0, len(self.__detection_user_info_list__)):
            user_info = self.__detection_user_info_list__[idx]
            if user_info == 1:
                # btn_o 를 누른 검출정보
                ip.drawRect(self.__rect_drawn_image__, self.__detection_list__[idx], 0, 255, 0)
            elif user_info == -1:
                # btn_x 를 누른 검출정보
                ip.drawRect(self.__rect_drawn_image__, self.__detection_list__[idx], 255, 0, 0)
            else:
                # 아직 버튼을 누르지 않은 검출정보
                ip.drawRect(self.__rect_drawn_image__, self.__detection_list__[idx], 255, 255, 0)
                for remain_idx in range(idx + 1, len(self.__detection_user_info_list__)):
                    ip.drawRect(self.__rect_drawn_image__, self.__detection_list__[remain_idx], 127, 127, 127)
                break

    def open_file(self):
        f_name = QFileDialog.getOpenFileName(self.window, 'Open file', '/home')

        if f_name[0]:
            f = open(f_name[0], 'r')
        else:
            return

        with f:
            file_data = f.read()
        self.window.textEdit.setText(file_data)

# def showDialog(window):
#     text, ok = QInputDialog.getText(window, 'Input Dialog',
#                                     'Enter your name:')
#
#     if ok:
#         window.le.setText(str(text))
