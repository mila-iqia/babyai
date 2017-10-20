#!/usr/bin/env python3

from __future__ import division, print_function

import gym
import gym_aigame

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QTextEdit
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
        imgHBox = QHBoxLayout()
        imgLabel = QLabel()
        imgHBox.addStretch(1)
        imgHBox.addWidget(imgLabel)
        imgHBox.addStretch(1)

        img = QImage(256, 256,QImage.Format_ARGB32_Premultiplied)
        painter = QPainter()
        painter.begin(img)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(0, 0, 255, 255)
        painter.end()
        imgLabel.setPixmap(QPixmap.fromImage(img))


        # Text edit boxes
        self.missionBox = QTextEdit("General mission")
        self.missionBox.textChanged.connect(self.setMission)
        self.adviceBox = QTextEdit("Contextual advice")
        self.adviceBox.textChanged.connect(self.setAdvice)
        textHBox = QHBoxLayout()
        textHBox.addWidget(self.missionBox)
        textHBox.addWidget(self.adviceBox)

        # Row of buttons

        nextStepBtn = QPushButton("Step")
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



        buttonHBox = QHBoxLayout()
        buttonHBox.addStretch(1)
        buttonHBox.addWidget(nextStepBtn)
        buttonHBox.addWidget(slider)
        buttonHBox.addStretch(1)
        buttonHBox.addWidget(plusButton)
        buttonHBox.addWidget(minusButton)
        buttonHBox.addStretch(1)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        vbox = QVBoxLayout()
        vbox.addLayout(imgHBox)
        vbox.addLayout(textHBox)
        vbox.addLayout(buttonHBox)
        mainWidget.setLayout(vbox)

        # Show the window
        self.show()

    def setMission(self):
        # TODO: connect to agent code
        print('set mission: ' + self.missionBox.toPlainText())

    def setAdvice(self):
        # TODO: connect to agent code
        print('set advice: ' + self.adviceBox.toPlainText())

    def plusReward(self):
        print('+Reward')

    def minusReward(self):
        print('-Reward')

    def setFrameRate(self, value):
        print('Sew frame rate: %s' % value)


if __name__ == '__main__':

    env = gym.make('AI-Game-v0')
    env.reset()

    app = QApplication(sys.argv)
    ex = AIGameWindow()
    sys.exit(app.exec_())
