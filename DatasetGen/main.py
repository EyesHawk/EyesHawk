import sys
from PyQt5.QtWidgets import QApplication
from DatasetGen.module.image_processing import *
from DatasetGen.gui.window import MyWindow

if __name__ == '__main__':
    # cur_dir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
