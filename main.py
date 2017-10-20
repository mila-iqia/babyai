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

    def __init__(
        self,
        missionEditCb,
        adviceEditCb,
        stepEnvCb,
        plusRewardCb,
        minusRewardCb
    ):
        super().__init__()
        self.initUI()

        self.missionEditCb = missionEditCb
        self.adviceEditCb = adviceEditCb
        self.stepEnvCb = stepEnvCb
        self.plusRewardCb = plusRewardCb
        self.minusRewardCb = minusRewardCb

    def initUI(self):
        """Create and connect the UI elements"""

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
        self.missionBox.textChanged.connect(self.missionEdit)
        vboxLeft = QVBoxLayout()
        vboxLeft.addWidget(QLabel("General mission"))
        vboxLeft.addWidget(self.missionBox)

        self.adviceBox = QTextEdit()
        self.adviceBox.textChanged.connect(self.adviceEdit)
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
        stepButton.clicked.connect(self.stepButton)

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

    def missionEdit(self):
        self.missionEditCb(self.missionBox.toPlainText())

    def adviceEdit(self):
        self.adviceEditCb(self.adviceBox.toPlainText())

    def plusReward(self):
        self.plusRewardCb()

    def minusReward(self):
        self.minusRewardCb()

    def stepButton(self):
        self.stepEnvCb()

    def setFrameRate(self, value):
        """Set the frame rate limit. Zero for manual stepping."""

        print('Set frame rate: %s' % value)

        if value == 0:
            self.fpsLabel.setText("Manual")
        elif value == 100:
            self.fpsLabel.setText("Fastest")
        else:
            self.fpsLabel.setText("%s FPS" % value)

    def setPixmap(self, pixmap):
        """Set the image to be displayed in the image area"""
        self.imgLabel.setPixmap(pixmap)

def main():

    window = None
    env = None

    def missionEdit(text):
        print('new mission: ' + text)

    def adviceEdit(text):
        print('new advice: ' + text)

    def plusReward():
        print('+ reward')

    def minusReward():
        print('- reward')

    def stepEnv():
        print('step')

        obs, reward, done, info = env.step(0)

        env.render()
        window.setPixmap(env.renderer.getPixmap())

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(
        missionEditCb = missionEdit,
        adviceEditCb = adviceEdit,
        stepEnvCb = stepEnv,
        plusRewardCb = plusReward,
        minusRewardCb = minusReward
    )

    env = gym.make('AI-Game-v0')
    obs = env.reset()

    env.render()
    window.setPixmap(env.renderer.getPixmap())

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
