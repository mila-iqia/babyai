import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPolygon
from PyQt5.QtCore import QPoint, QSize, QRect

class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.img = QImage(width, height, QImage.Format_RGB888)
        self.painter = QPainter()

    def close(self):
        """
        Deallocate resources used
        """
        pass

    def beginFrame(self):
        self.painter.begin(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing, False)

        # Clear the background
        self.painter.setBrush(QColor(0, 0, 0))
        self.painter.drawRect(0, 0, self.width - 1, self.height - 1)

    def endFrame(self):
        self.painter.end()

    def getPixmap(self):
        return QPixmap.fromImage(self.img)

    def getArray(self):
        """
        Get a numpy array of RGB pixel values.
        The size argument should be (3,w,h)
        """

        width = self.width
        height = self.height
        shape = (width, height, 3)

        # Copy the pixel data to a numpy array
        #output = np.ndarray(shape=shape, dtype='uint8')
        #for y in range(0, height):
        #    for x in range(0, width):
        #        pix = self.img.pixel(x, y)
        #        r = (pix >> 16) & 0xFF
        #        g = (pix >>  8) & 0xFF
        #        b = (pix >>  0) & 0xFF
        #        output[x, y, 0] = r
        #        output[x, y, 1] = g
        #        output[x, y, 2] = b

        numBytes = self.width * self.height * 3
        buf = self.img.bits().asstring(numBytes)
        output = np.frombuffer(buf, dtype='uint8')
        output = output.reshape(shape)

        return output

    def push(self):
        self.painter.save()

    def pop(self):
        self.painter.restore()

    def rotate(self, degrees):
        self.painter.rotate(degrees)

    def translate(self, x, y):
        self.painter.translate(x, y)

    def scale(self, x, y):
        self.painter.scale(x, y)

    def setLineColor(self, r, g, b, a=255):
        self.painter.setPen(QColor(r, g, b, a))

    def setColor(self, r, g, b, a=255):
        self.painter.setBrush(QColor(r, g, b, a))

    def drawLine(self, x0, y0, x1, y1):
        self.painter.drawLine(x0, y0, x1, y1)

    def drawCircle(self, x, y, r):
        center = QPoint(x, y)
        self.painter.drawEllipse(center, r, r)

    def drawPolygon(self, points):
        """Takes a list of points (tuples) as input"""
        points = map(lambda p: QPoint(p[0], p[1]), points)
        self.painter.drawPolygon(QPolygon(points))

    def fillRect(self, x, y, width, height, r, g, b, a=255):
        self.painter.fillRect(QRect(x, y, width, height), QColor(r, g, b, a))
