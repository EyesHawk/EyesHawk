import time
from cv2 import cv2
from matplotlib import pyplot as plot
import tensorflow as tf


SHOW_PLOT = True
SHOW_PLOT_REAL_TIME = False
SHOW_VIDEO_REAL_TIME = True


# frame 을 가져오는 모듈
def get_frame_list(filename):
    if not isinstance(filename, str):
        raise TypeError("filename 은 문자열이어야 합니다.")

    vc = cv2.VideoCapture()
    vc.open(filename)

    if not vc.isOpened():
        print("파일이 열리지 않습니다.")
        return

    frame_list = []
    _success, _frame = vc.read()
    while _success :
        frame_list.append(_frame)
        _success, _frame = vc.read()
    return frame_list


hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print(sess.run(hello))

vc = cv2.VideoCapture()
vc.open("resource/sample_seq.mp4")
width = vc.get(3)
height = vc.get(4)
fps = vc.get(5)

print("start video(" + str(width) + "," + str(height) + "," + str(fps) + ")")

image_list = []
progress_time_list = []

if vc.isOpened():
    canShowable, frame = vc.read()
    if SHOW_PLOT_REAL_TIME:
        plot.figure()
        plot.show(block=False)
    while canShowable:
        start = time.clock()

        image_list.append(frame)
        half = cv2.resize(frame, None, fx=0.5, fy=0.5)
        if SHOW_VIDEO_REAL_TIME:
            cv2.imshow("sample", half)
        canShowable, frame = vc.read()

        end = time.clock()

        progress_time_list.append(end-start)

        if SHOW_PLOT_REAL_TIME:
            plot.plot(progress_time_list)
            plot.draw()
            plot.pause(0.0001)

        if cv2.waitKey(int(fps)) & 0xFF == ord('q'):
            plot.close()
            break

print("end of video")
module_image_list = get_frame_list("resource/sample_seq.mp4")
print("image count : " + str(len(image_list)) + " / module's image count : " + str(len(module_image_list)))

vc.release()
cv2.destroyAllWindows()

if SHOW_PLOT:
    plot.figure()
    plot.plot(progress_time_list)
    plot.show()

plot.close()
