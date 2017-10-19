#!/usr/bin/env python3

from __future__ import division, print_function

import gym
import gym_aigame

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QTextEdit
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout
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
        missionBox = QTextEdit("General mission")
        adviceBox = QTextEdit("Contextual advice")
        textHBox = QHBoxLayout()
        textHBox.addWidget(missionBox)
        textHBox.addWidget(adviceBox)

        # Row of buttons
        stepModeBtn = QPushButton("Step Mode")
        stepModeBtn.setCheckable(True)
        liveModeBtn = QPushButton("MoveMode")
        liveModeBtn.setCheckable(True)
        nextStepBtn = QPushButton("Next Step")
        addButton = QPushButton("+ Reward")
        subButton = QPushButton("- Reward")
        buttonHBox = QHBoxLayout()
        buttonHBox.addStretch(1)
        buttonHBox.addWidget(stepModeBtn)
        buttonHBox.addWidget(liveModeBtn)
        buttonHBox.addWidget(nextStepBtn)
        buttonHBox.addWidget(addButton)
        buttonHBox.addWidget(subButton)
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

if __name__ == '__main__':

    env = gym.make('AI-Game-v0')
    env.reset()

    app = QApplication(sys.argv)
    ex = AIGameWindow()
    sys.exit(app.exec_())
