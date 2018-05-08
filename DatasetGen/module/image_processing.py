from PyQt5.QtGui import QImage, QPixmap
from cv2 import *
import numpy as np


class ImageProcessing:
    @staticmethod
    # 기본적으로 메모리가 공유되니 필요시 깊은 복사를 사용해야 함.
    def pixmap_to_ndarray(qPixmap: QPixmap) -> np.ndarray:
        bits = qPixmap.toImage().bits()
        height = qPixmap.height()
        width = qPixmap.width()
        channel = 3
        if qPixmap.hasAlphaChannel():
            channel += 1
        bits.setsize(height * width * channel)
        ndarray = np.ndarray(shape=(height, width, channel), buffer=bits, dtype=np.uint8)
        return ndarray

    @staticmethod
    def ndarray_to_pixmap(ndarray: np.ndarray) -> QPixmap:
        if ndarray.size == 0:
            print('warn : ndarray size = 0')
            return QPixmap()
        rgb_image = cvtColor(ndarray, COLOR_BGR2RGB)
        q_image = QImage(rgb_image, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    @staticmethod
    def multipleHistogramBackProjection(trust_dir_name, quantize_level=64) -> (np.ndarray, int):
        # todo : 여기서부터 quantize 모두 nparray 써야함.
        read_file_count = 0
        file_list = os.listdir(trust_dir_name)
        ret = np.array([[0] * quantize_level] * quantize_level, dtype=np.float64)
        for file in file_list:
            if file.endswith('png') | file.endswith('jpg'):
                file_path = trust_dir_name + file
                img = imread(file_path, IMREAD_COLOR)
                if len(np.shape(img)) != 3:
                    continue
                hsv_img = cvtColor(img, COLOR_BGR2HSV)
                ret += ImageProcessing.quantizedHsHistogram(hsv_img, quantize_level)
                read_file_count += 1
                print(file+' : back projection 완료')
            # ------
            # video
            #
            # if file.endswith('avi') | file.endswith('mp4'):
            #     cap = VideoCapture(file)
            #     can_readable, frame = cap.read()
            #     while can_readable:
            #         hsv_img = cvtColor(frame, COLOR_BGR2HSV)
            #         histImg(hsv_img)
            #         read_file_count += 1
            #         can_readable, frame = cap.read()
            # ------
        ret /= read_file_count
        return ret, read_file_count

    @staticmethod
    def quantizedHsHistogram(hsv_img: np.ndarray, quantize_level=64) -> np.ndarray:
        # https://docs.opencv.org/3.3.1/dc/df6/tutorial_py_histogram_backprojection.html
        hist = np.array([[0] * quantize_level] * quantize_level, dtype=np.float64)
        rows = hsv_img.shape[0]
        cols = hsv_img.shape[1]
        for y in range(0, rows):
            for x in range(0, cols):
                pixel = hsv_img[y, x]
                qH = ImageProcessing.quantize(pixel[0], quantize_level)
                qS = ImageProcessing.quantize(pixel[1], quantize_level)
                hist[qH, qS] += 1
        hist /= (rows * cols)
        return hist

    @staticmethod
    def quantize(value, quantize_level=64) -> int:
        return int(round(value / 255.0 * (quantize_level - 1)))

    # trustability = 1이 최대
    # 픽셀의 히스토그램 신뢰도가 0.2차이가 나는 애들이 50% 일 때 해당 square를 검출함.
    # square 는 squareAreaShiftSize(areaSize 의 반)씩 시프트함.
    # return : 4개의 값을 가진 튜플의 리스트를 반환.
    @staticmethod
    def detect_from_hs_hist(hist: np.ndarray, img: np.ndarray,
                            quantize_level=64, hist_trustability=0.2, square_area_size=10,
                            square_area_shift_size=5, square_area_threshold=0.5) -> list:
        assert hist.shape == (quantize_level, quantize_level)
        assert len(img.shape) == 3

        detection_list = []
        padding = int(square_area_size / 2)
        rows = img.shape[0]
        cols = img.shape[1]
        arr = [[0, 0] * rows for i in range(cols)]
        for y in range(rows):
            for x in range(cols):
                hsv_pixel = img[y, x]
                hQ_val = ImageProcessing.quantize(hsv_pixel[0], quantize_level)
                sQ_val = ImageProcessing.quantize(hsv_pixel[1], quantize_level)
                if hist_trustability <= hist[hQ_val, sQ_val]:
                    arr[y][x] = 1
                else:
                    arr[y][x] = 0
        for y in range(rows):
            s = 0
            for x in range(cols):
                v = arr[y][x]
                if y > 0:
                    arr[y][x] += arr[y - 1][x]
                arr[y][x] += s
                s += v
        for y in range(padding, rows - padding, square_area_shift_size):
            for x in range(padding, cols - padding, square_area_shift_size):
                s = arr[y][x] - arr[y - square_area_shift_size - 1][x] - arr[y][x - square_area_shift_size - 1] + \
                    arr[y - square_area_shift_size - 1][x - square_area_shift_size - 1]
                areaTrustability = s / (square_area_size ** 2)
                if square_area_threshold <= areaTrustability:
                    detection_list.append((x - square_area_size, y - square_area_size, x + square_area_size - 1,
                                           y + square_area_size - 1))
                    print(str(y) + "," + str(x) + " : value(" + str(areaTrustability) + ")")
        return detection_list

    @staticmethod
    def drawRect(img: np.ndarray, rect: tuple, r: int, g: int, b: int) -> None:
        assert len(rect) == 4
        rectangle(img, rect[:2], rect[2:], (b, g, r), 2)

    @staticmethod
    # ascii 문자만 가능함.
    def drawText(img: np.ndarray, text: str, position: tuple) -> None:
        assert len(position) == 2
        cv2.putText(img, text, position, FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

# if __name__ == "__main__":
#     # todo : 삭제해야함.
#     cur_dir = os.path.dirname(__file__)
#     # sys.path.insert(cur_dir + "/../")
#
#     trust_dir_name = cur_dir + '/trustGroup/'
#     exp_dir_name = cur_dir + '/experimentalGroup/'
#     hs_histogram, trustGroupCount = ImageProcessing.multipleHistogramBackProjection(trust_dir_name)
#     experimentalGroup = os.listdir(exp_dir_name)
#     print('generate dataset with %d experimentalGroup, %d trustGroup, trustGroup Size is (?,?)'
#           % (len(experimentalGroup), trustGroupCount))
