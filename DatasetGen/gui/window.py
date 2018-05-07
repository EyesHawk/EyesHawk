import os
from PyQt5.QtWidgets import QMainWindow, QPushButton, QAction, QLabel, QTextEdit, QMessageBox
from PyQt5.QtGui import QIcon
from DatasetGen.gui.connect_func import Interaction as Ui


class MyWindow(QMainWindow):
    def __init__(self,
                 quantize_level=64,
                 trust_dir_name=os.path.dirname(__file__) + '/../resource/trustGroup/',
                 experimental_dir_name=os.path.dirname(__file__) + '/../resource/experimentalGroup/',
                 output_dir_name=os.path.dirname(__file__) + '/../resource/output/'):
        super().__init__()

        self.ui = Ui(self, quantize_level, trust_dir_name, experimental_dir_name, output_dir_name)

        self.setWindowTitle("호크아이")
        self.setGeometry(300, 300, 1200, 500)  # 위치 가로 세로

        self.btn_next = QPushButton("다음", self)  # 버튼 정의
        self.btn_next.move(630, 200)
        self.btn_next.resize(100, 100)
        self.btn_next.clicked.connect(self.ui.click_next_image)  # image_load_next)

        self.btn_prev = QPushButton("이전", self)  # 버튼 정의
        self.btn_prev.move(30, 200)
        self.btn_prev.resize(100, 100)
        self.btn_prev.clicked.connect(self.ui.click_prev_image)

        self.btn_x = QPushButton("X", self)  # 버튼 정의
        self.btn_x.move(120, 450)
        self.btn_x.clicked.connect(self.ui.click_btn_x)

        self.btn_refresh = QPushButton("refresh", self)  # 버튼 정의
        self.btn_refresh.move(320, 450)
        self.btn_refresh.clicked.connect(self.ui.click_btn_refresh)

        self.btn_o = QPushButton("O", self)  # 버튼 정의
        self.btn_o.move(520, 450)
        self.btn_o.clicked.connect(self.ui.click_btn_o)

        self.btn_save = QPushButton("저장", self)  # 버튼 정의
        self.btn_save.move(900, 450)
        self.btn_save.clicked.connect(self.ui.click_btn_save)

        self.openFile = QAction(QIcon('open.png'), 'Open', self)
        self.openFile.setShortcut('Ctrl+O')
        self.openFile.setStatusTip('Open new File')
        self.openFile.triggered.connect(self.ui.open_file)

        self.menuBar = self.menuBar()
        self.fileMenu = self.menuBar.addMenu('&File')
        self.fileMenu.addAction(self.openFile)

        self.statusBar()
        self.label = QLabel(self)
        self.textEdit = QTextEdit(self)
        self.textEdit.move(830, 80)
        self.textEdit.resize(300, 300)

        self.ui.initialize()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
