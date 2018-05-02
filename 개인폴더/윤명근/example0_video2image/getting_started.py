#
# 1. tensorflow 라이브러리 체크 : Hello, Tensorflow 출력
# 2. matplotlib 사용해보기 : 원본 영상을 가로 세로 2배 축소했을 때 걸리는 시간을 그래프로 그려봄.
# 3. frame 단위로 저장하는 모듈 작성 : module.video2frame
# 4. 그 외 opencv 를 이용한 비디오 정보 읽기 등.
#


if __name__ == "__main__":
    import os
    cur_dir = os.path.dirname(__file__)
    import sys
    sys.path.insert(0, '../')
    import time
    import tensorflow as tf
    from matplotlib import pyplot as plot
    from module.video2frame import *  # from cv2 import cv2


    SHOW_PLOT = True
    SHOW_PLOT_REAL_TIME = False
    SHOW_VIDEO_REAL_TIME = True

    hello = tf.constant('Hello, Tensorflow!')
    sess = tf.Session()
    print(sess.run(hello))

    vc = cv2.VideoCapture()
    vc.open(cur_dir + "/resource/sample_seq.mp4")
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
            image_list.append(frame)

            start = time.clock()
            half = cv2.resize(frame, None, fx=0.5, fy=0.5)
            end = time.clock()
            progress_time_list.append(end - start)

            if SHOW_VIDEO_REAL_TIME:
                cv2.imshow("sample", half)
            canShowable, frame = vc.read()

            if SHOW_PLOT_REAL_TIME:
                plot.plot(progress_time_list)
                plot.draw()
                plot.pause(0.0001)

            if cv2.waitKey(int(fps)) & 0xFF == ord('q'):
                plot.close()
                break

    print("end of video")
    module_image_list = get_frame_list(cur_dir+"/resource/sample_seq.mp4")
    print("progress time list count : " + str(len(progress_time_list)) +
          " / module's whole image count : " + str(len(module_image_list)))

    vc.release()
    cv2.destroyAllWindows()

    if SHOW_PLOT:
        plot.figure()
        plot.plot(progress_time_list)
        plot.show()

    plot.close()
