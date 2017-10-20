#!/usr/bin/env python3

from __future__ import division, print_function

import gym
import gym_aigame

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

class AIGameWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.resize(512, 512)
        self.setWindowTitle('Baby AI Game')

        # Image widget to draw into
        imgHBox = self.createImageArea()

        # Text edit boxes
        textHBox = self.createTextEdits()

        # Row of buttons
        buttonHBox = self.createButtons()

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        vbox = QVBoxLayout()
        vbox.addLayout(imgHBox)
        vbox.addLayout(textHBox)
        vbox.addLayout(buttonHBox)
        mainWidget.setLayout(vbox)

        # Show the application window
        self.show()

    def createImageArea(self):
        """Create the image area to render into"""

        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        img = QImage(256, 256, QImage.Format_ARGB32_Premultiplied)
        painter = QPainter()
        painter.begin(img)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(0, 0, 255, 255)
        painter.end()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.imgLabel)
        hbox.addStretch(1)

        return hbox

    def createTextEdits(self):
        """Create the text edit boxes"""

        self.missionBox = QTextEdit()
        self.missionBox.textChanged.connect(self.setMission)
        vboxLeft = QVBoxLayout()
        vboxLeft.addWidget(QLabel("General mission"))
        vboxLeft.addWidget(self.missionBox)

        self.adviceBox = QTextEdit()
        self.adviceBox.textChanged.connect(self.setAdvice)
        vboxRight = QVBoxLayout()
        vboxRight.addWidget(QLabel("Contextual advice"))
        vboxRight.addWidget(self.adviceBox)

        hbox = QHBoxLayout()
        hbox.addLayout(vboxLeft)
        hbox.addLayout(vboxRight)

        return hbox

    def createButtons(self):
        """Create the row of UI buttons"""

        stepButton = QPushButton("Step")
        stepButton.clicked.connect(self.manualStep)

        plusButton = QPushButton("+ Reward")
        plusButton.clicked.connect(self.plusReward)

        minusButton = QPushButton("- Reward")
        minusButton.clicked.connect(self.minusReward)

        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.valueChanged.connect(self.setFrameRate)

        self.fpsLabel = QLabel("Manual")
        self.fpsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.fpsLabel.setAlignment(Qt.AlignCenter)
        self.fpsLabel.setMinimumSize(80, 10)


        # Assemble the buttons into a horizontal layout
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(stepButton)
        hbox.addWidget(slider)
        hbox.addWidget(self.fpsLabel)
        hbox.addStretch(1)
        hbox.addWidget(plusButton)
        hbox.addWidget(minusButton)
        hbox.addStretch(1)

        return hbox

    def setMission(self):
        # TODO: connect to agent code
        print('set mission: ' + self.missionBox.toPlainText())

    def setAdvice(self):
        # TODO: connect to agent code
        print('set advice: ' + self.adviceBox.toPlainText())

    def manualStep(self):
        print('manual step')

    def plusReward(self):
        print('+Reward')

    def minusReward(self):
        print('-Reward')

    def setFrameRate(self, value):
        print('Set frame rate: %s' % value)

        if value == 0:
            self.fpsLabel.setText("Manual")
        elif value == 100:
            self.fpsLabel.setText("Fastest")
        else:
            self.fpsLabel.setText("%s FPS" % value)

if __name__ == '__main__':

    env = gym.make('AI-Game-v0')
    env.reset()

    app = QApplication(sys.argv)
    ex = AIGameWindow()
    sys.exit(app.exec_())
