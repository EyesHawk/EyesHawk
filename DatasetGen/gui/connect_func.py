import os
# import sys
# cur_dir = os.path.dirname(__file__)
# sys.path.insert(0, cur_dir+'/../')
# from DatasetGen.gui.window import MyWindow
from PyQt5.QtWidgets import QFileDialog  # , QMainWindow
from ..module.image_processing import ImageProcessing as ip
import numpy as np
import cv2 as cv
from ..module.image_processing import Tester

save_num = 1064


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
        self.__hist__, self.__trust_file_count__ = ip.multiple_histogram_back_projection(self.trust_dir_name,
                                                                                         self.quantize_level)
        self.__exp_file_list__ = os.listdir(self.experimental_dir_name)
        self.__load_images__(self.__exp_file_order__)

    def click_btn_o(self):
        if len(self.__detection_user_info_list__) <= self.__detection_order__:
            Tester.print('검출 데이터를 모두 계산하였습니다. save 후 다음 이미지를 눌러주세요.')
            return
        else:
            self.__dirty__ = True
            self.__detection_user_info_list__[self.__detection_order__] = 1
            self.window.textEdit.append('1,' +
                                        ','.join(str(tuple_data) for tuple_data in
                                                 self.__detection_list__[self.__detection_order__]))
            self.__increment_detection_order__()

    def click_btn_x(self):
        if len(self.__detection_user_info_list__) <= self.__detection_order__:
            Tester.print('검출 데이터를 모두 계산하였습니다. save 후 다음 이미지를 눌러주세요.')
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
            Tester.print('save 또는 refresh 이후 prev 또는 next 이미지로 이동할 수 있습니다.')
            return
        if self.__exp_file_order__ <= 0:
            Tester.print('첫번째 이미지입니다.')
            return
        self.window.textEdit.setText('')
        self.__exp_file_order__ -= 1
        self.__load_images__(self.__exp_file_order__)

    def click_next_image(self):
        if self.__dirty__:
            Tester.print('save 또는 refresh 이후 prev 또는 next 이미지로 이동할 수 있습니다.')
            return
        if len(self.__exp_file_list__) - 1 <= self.__exp_file_order__:
            Tester.print('마지막 이미지입니다.')
            return
        self.window.textEdit.setText('')
        self.__exp_file_order__ += 1
        self.__load_images__(self.__exp_file_order__)

    def down_sizing_and_save(self):
        global save_num
        save_num += 1
        if (self.__current_image__.shape[0] > 32) & (self.__current_image__.shape[1] > 32):
            # resized_image = np.array([[0] * 32] * 32, dtype=self.__current_image__.dtype)
            y_mean = 0
            x_mean = 0
            for detect in self.__detection_list__:
                x_mean += (detect[2] + detect[0]) / 2
                y_mean += (detect[3] + detect[1]) / 2
            if len(self.__detection_list__) != 0:
                y_mean /= len(self.__detection_list__)
                x_mean /= len(self.__detection_list__)

            shape = self.__current_image__.shape
            start_y = 0
            start_x = 0
            if shape[0] < shape[1]:  # y < x
                rows = shape[0]
                cols = shape[0]
                x_range = (shape[0] / 2, shape[0] / 2 + (shape[1] - shape[0]))
                print('x_range : ' + str(x_range))
                print('x_mean : ' + str(x_mean))

                if x_mean < x_range[0]:  # 왼쪽에 붙이기
                    pass
                elif x_range[1] < x_mean:  # 오른쪽에 붙이기
                    start_x += x_range[1] - x_range[0] - 1
                    cols += x_range[1] - x_range[0] - 1
                else:  # 그대로
                    start_x += x_mean - x_range[0] - 1
                    cols += x_mean - x_range[0] - 1

                if start_x < 0:
                    start_x = 0

            else:  # y > x
                rows = shape[1]
                cols = shape[1]
                y_range = (shape[1] / 2, shape[1] / 2 + (shape[0] - shape[1]))
                print('y_range : ' + str(y_range))
                print('y_mean : ' + str(y_mean))

                if y_mean < y_range[0]:
                    pass
                elif y_range[1] < y_mean:
                    start_y += y_range[1] - y_range[0] - 1
                    rows += y_range[1] - y_range[0] - 1
                else:
                    start_y += y_mean - y_range[0] - 1
                    rows += y_mean - y_range[0] - 1
                if start_y < 0:
                    start_y = 0

            sub_image = self.__current_image__[int(start_y):int(rows), int(start_x):int(cols)]

            image_32 = cv.resize(sub_image, (32, 32))
            cv.imshow('32', image_32)
            save_dir = self.output_dir_name + 'test32image/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            Tester.print(save_dir + str(save_num) + '.jpg')
            cv.imwrite(save_dir + str(save_num) + '.jpg', img=image_32)
        else:
            print('waring : 현재 이미지 크기가 32를 넘지 않음. ' + str(self.__current_image__.shape))

    def __load_images__(self, order: int):
        if self.__dirty__:
            if self.window.textEdit.toPlainText() != "":
                Tester.print('refresh 또는 save 를 눌러주세요.')
        else:
            file_path = self.experimental_dir_name + self.__exp_file_list__[order]
            self.__current_image__ = cv.imread(file_path, cv.IMREAD_COLOR)
            # self.__current_image__ = QPixmap(self.experimental_dir_name + self.__exp_file_list__[order])

            if self.__current_image__ is None:
                self.window.label.setText('이미지 로드 실패 - 불러올 수 없는 이미지 :\n' + self.__exp_file_list__[order])
            elif self.__current_image__.size == 0:
                self.window.label.setText('이미지 로드 실패 - 불러올 수 없는 이미지 :\n' + self.__exp_file_list__[order])
            else:
                print(file_path)
                cv.imshow('origin image', self.__current_image__)
                shape = list(self.__current_image__.shape)
                dst_size = 150
                if dst_size < shape[0]:
                    shape[1] = int(shape[1] / shape[0] * dst_size)
                    shape[0] = dst_size
                if dst_size < shape[1]:
                    shape[0] = int(shape[0] / shape[1] * dst_size)
                    shape[1] = dst_size
                self.__current_image__ = cv.resize(self.__current_image__, tuple(shape[0:2]))
                self.__detection_list__ = ip.detect_from_hs_hist(self.__hist__,
                                                                 self.__current_image__,
                                                                 threshold_hist_percent=0.15,
                                                                 # histogram 최대 값보다 n% 이상인 것은 모두 처리
                                                                 threshold_area_percent=0.15,
                                                                 # 찾은 면적비가 n% 이상일 때
                                                                 quantize_level=self.quantize_level,
                                                                 square_size=20,
                                                                 square_area_shift_size=10
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
            self.__rect_drawn_image__ = self.__current_image__.copy()
            return
        if len(self.__detection_list__) <= self.__detection_order__:
            last_idx = self.__detection_order__ - 1
            user_info = self.__detection_user_info_list__[last_idx]
            if user_info == 1:
                ip.draw_rect(self.__rect_drawn_image__, self.__detection_list__[last_idx], 0, 255, 0)
            elif user_info == -1:
                ip.draw_rect(self.__rect_drawn_image__, self.__detection_list__[last_idx], 255, 0, 0)
            ip.draw_text(self.__rect_drawn_image__, 'done, save and change image.', (0, 400))
            return

        self.__rect_drawn_image__ = self.__current_image__.copy()
        for idx in range(0, len(self.__detection_user_info_list__)):
            user_info = self.__detection_user_info_list__[idx]
            if user_info == 1:
                # btn_o 를 누른 검출정보
                ip.draw_rect(self.__rect_drawn_image__, self.__detection_list__[idx], 0, 255, 0)
            elif user_info == -1:
                # btn_x 를 누른 검출정보
                ip.draw_rect(self.__rect_drawn_image__, self.__detection_list__[idx], 255, 0, 0)
            else:
                # 아직 버튼을 누르지 않은 검출정보
                ip.draw_rect(self.__rect_drawn_image__, self.__detection_list__[idx], 255, 255, 0)
                for remain_idx in range(idx + 1, len(self.__detection_user_info_list__)):
                    ip.draw_rect(self.__rect_drawn_image__, self.__detection_list__[remain_idx], 127, 127, 127)
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
