import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("호크아이")
        self.setGeometry(300, 300, 500, 300) # 위치 가로 세로
        
        btn1 = QPushButton("btn1", self)    # 버튼 정의
        btn1.move(230, 20)
        btn1.clicked.connect(self.btn1_clicked) 
        

        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.openfile)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        
        btn = QPushButton('Dialog', self)
        btn.move(20, 20)
        btn.clicked.connect(self.showDialog)
        
        self.le = QLineEdit(self)
        self.le.move(130, 22)


        layout = QHBoxLayout()
        layout.addWidget(btn1)
        layout.addWidget(btn)

        self.setLayout(layout)
        self.statusBar()
        
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

    def contextMenuEvent(self, event):
       
        cmenu = QMenu(self)
           
        newAct = cmenu.addAction("New")
        opnAct = cmenu.addAction("Open")
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.pos()))
           
        if action == quitAct:
            qApp.quit() 

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
