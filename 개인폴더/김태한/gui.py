import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QPixmap


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("호크아이")
        self.setGeometry(300, 300, 1200, 500) # 위치 가로 세로
        
        btn_next = QPushButton("다음", self)    # 버튼 정의
        btn_next.move(630, 200)
        btn_next.clicked.connect(self.image_load_next) 
        
        btn_previous = QPushButton("이전", self)    # 버튼 정의
        btn_previous.move(30, 200)
        btn_previous.clicked.connect(self.image_load_previous) 

        btn_x = QPushButton("X", self)    # 버튼 정의
        btn_x.move(120, 450)
        btn_x.clicked.connect(self.btn_x)

        btn_refresh = QPushButton("refresh", self)    # 버튼 정의
        btn_refresh.move(320, 450)
        btn_refresh.clicked.connect(self.btn_refresh)

        btn_o = QPushButton("O", self)    # 버튼 정의
        btn_o.move(520, 450)
        btn_o.clicked.connect(self.btn_o) 

        btn_o = QPushButton("저장", self)    # 버튼 정의
        btn_o.move(900, 450)
        btn_o.clicked.connect(self.btn_save) 
        
        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.openfile)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
                
        self.statusBar()
        self.label = QLabel(self)
        self.textedit = QTextEdit(self)
        self.textedit.move(830, 80)
        self.textedit.resize(300, 300)

    def image_load_next(self):

        pixmap = QPixmap('test0.png')   #파일 이름 넣으면 됨 단 같은 경로만
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(),pixmap.height())
        self.label.move(300, 200)

    def image_load_previous(self):

        pixmap = QPixmap('test1.png')        #파일 이름 넣으면 됨 단 같은 경로만
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(),pixmap.height())
        self.label.move(300, 200)

    def btn_o(self):
        self.textedit.setText("입력 text")

    def btn_x(self):
        print(self.textedit.toPlainText())

    def btn_refresh(self):
        self.textedit.setText("")

    def btn_save(self):
        f = open("test.txt", 'w')
        f.write(self.textedit.toPlainText())
        
    def openfile(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        
       ## if fname[0]:
         ##   f = open(fname[0], 'r')

           ## with f:
             ##   data = f.read()
               ## self.textEdit.setText(data)        
        

    def showDialog(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter your name:')
        
        if ok:
            self.le.setText(str(text))
        
    def btn1_clicked(self):  ## 버튼 실행 함수
        print("btn 1")   
      
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
