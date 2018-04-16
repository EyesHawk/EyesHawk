from cv2 import cv2


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
    while _success:
        frame_list.append(_frame)
        _success, _frame = vc.read()
    return frame_list
