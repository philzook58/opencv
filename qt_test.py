import sys
from PyQt4 import QtGui, QtCore

app = QtGui.QApplication(sys.argv)
widget = QtGui.QWidget()

widget.setGeometry(200, 100, 400, 300)
widget.setWindowTitle('PyQt Application')

slider = QtGui.QSlider(QtCore.Qt.Horizontal, widget)
slider.setGeometry(10, 10, 200, 30)
slider.setFocusPolicy(QtCore.Qt.NoFocus)

def getValue(value):
    print value
widget.connect(slider, QtCore.SIGNAL('valueChanged(int)'), getValue)


widget.show()

sys.exit(app.exec_())
