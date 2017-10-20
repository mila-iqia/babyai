from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPolygon
from PyQt5.QtCore import QPoint

class Renderer:

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.img = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        self.painter = QPainter()

    def beginFrame(self):
        self.painter.begin(self.img)

        # Clear the background
        self.painter.setBrush(QColor(0, 0, 0))
        self.painter.drawRect(0, 0, self.width - 1, self.height - 1)

    def endFrame(self):
        self.painter.end()

    def getPixmap(self):
        return QPixmap.fromImage(self.img)

    # TODO: function to downsample and get numpy array

    def push(self):
        self.painter.save()

    def pop(self):
        self.painter.restore()

    def rotate(self, degrees):
        self.painter.rotate(degrees)

    def translate(self, x, y):
        self.painter.translate(x, y)

    def setLineColor(self, r, g, b):
        self.painter.setPen(QColor(r, g, b))

    def setColor(self, r, g, b):
        self.painter.setBrush(QColor(r, g, b))

    def drawLine(self, x0, y0, x1, y1):
        self.painter.drawLine(x0, y0, x1, y1)

    def drawCircle(self, x, y, r):
        center = QPoint(x, y)
        self.painter.drawEllipse(center, r, r)

    def drawPolygon(self, points):
        """Takes a list of points (tuples) as input"""
        points = map(lambda p: QPoint(p[0], p[1]), points)
        self.painter.drawPolygon(QPolygon(points))
