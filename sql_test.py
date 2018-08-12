import mysql.connector
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from cameralistView import Ui_MainWindow

class MyApp(QMainWindow):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def array_2_table(self, array, qtable):
        qtable.setColumnCount(2) 
        qtable.setRowCount(78)
        for row in range(600):
              for column in range(10):
                  qtable.setItem(row,column,QTableWidgetItem(QString("%1").arg(array[row][column])))

if __name__ == '__main__':
    cnx = mysql.connector.connect(host = "localhost", user='root', password='Tao190139', database='mydb', auth_plugin='caching_sha2_password')
    cursor = cnx.cursor()
    cursor.execute("SELECT * from province")
    data = cursor.fetchall()
    result = {}
    for item in data:
        result.update({str(item[0]):item[1]})
    print(result)
    app = QApplication(sys.argv)
    myapp = MyApp()
    myapp.array_2_table(result, myapp.ui.tableWidget)
    myapp.show()
    cnx.close()
    sys.exit(app.exec_())
