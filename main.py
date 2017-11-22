#!/usr/bin/env python3

import time
import sys
import threading

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

import gym
from gym_aigame.envs import AIGameEnv, Annotator
from model.training import State, selectAction, storeTrans

class AIGameWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

        # By default, manual stepping only
        self.fpsLimit = 0

        self.env = gym.make('AI-Game-v0')
        self.env = Annotator(self.env, saveOnClose=True)

        self.state = None

        self.resetEnv()

        self.stepTimer = QTimer()
        self.stepTimer.setInterval(0)
        self.stepTimer.setSingleShot(False)
        self.stepTimer.timeout.connect(self.stepClicked)

    def initUI(self):
        """Create and connect the UI elements"""

        self.resize(512, 512)
        self.setWindowTitle('Baby AI Game')

        # Full render view (large view)
        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        leftBox = QVBoxLayout()
        leftBox.addStretch(1)
        leftBox.addWidget(self.imgLabel)
        leftBox.addStretch(1)

        # Area on the right of the large view
        rightBox = self.createRightArea()

        # Arrange widgets horizontally
        hbox = QHBoxLayout()
        hbox.addLayout(leftBox)
        hbox.addLayout(rightBox)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainWidget.setLayout(hbox)

        # Show the application window
        self.show()
        self.setFocus()

    def createRightArea(self):
        # Agent render view (partially observable)
        self.obsImgLabel = QLabel()
        self.obsImgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        miniViewBox = QHBoxLayout()
        miniViewBox.addStretch(1)
        miniViewBox.addWidget(self.obsImgLabel)
        miniViewBox.addStretch(1)

        self.missionBox = QTextEdit()
        self.missionBox.setMinimumSize(500, 100)
        self.missionBox.textChanged.connect(self.missionEdit)

        self.adviceBox = QTextEdit()
        self.adviceBox.setMinimumSize(500, 100)
        self.adviceBox.textChanged.connect(self.adviceEdit)

        buttonBox = self.createButtons()

        self.stepsLabel = QLabel()
        self.stepsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stepsLabel.setAlignment(Qt.AlignCenter)
        self.stepsLabel.setMinimumSize(60, 10)
        stepsBox = QHBoxLayout()
        stepsBox.addStretch(1)
        stepsBox.addWidget(QLabel("Steps remaining"))
        stepsBox.addWidget(self.stepsLabel)
        stepsBox.addStretch(1)

        # Stack everything up in a vetical layout
        vbox = QVBoxLayout()
        vbox.addLayout(miniViewBox)
        vbox.addLayout(stepsBox)
        vbox.addWidget(QLabel("General mission"))
        vbox.addWidget(self.missionBox)
        vbox.addWidget(QLabel("Contextual advice"))
        vbox.addWidget(self.adviceBox)
        vbox.addLayout(buttonBox)

        return vbox

    def createButtons(self):
        """Create the row of UI buttons"""

        stepButton = QPushButton("Step")
        stepButton.clicked.connect(self.stepClicked)

        minusButton = QPushButton("- Reward")
        minusButton.clicked.connect(self.minusReward)

        plusButton = QPushButton("+ Reward")
        plusButton.clicked.connect(self.plusReward)

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
        hbox.addWidget(minusButton)
        hbox.addWidget(plusButton)
        hbox.addStretch(1)

        return hbox

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.stepEnv(AIGameEnv.ACTION_LEFT)
        elif e.key() == Qt.Key_Right:
            self.stepEnv(AIGameEnv.ACTION_RIGHT)
        elif e.key() == Qt.Key_Up:
            self.stepEnv(AIGameEnv.ACTION_FORWARD)
        elif e.key() == Qt.Key_Space:
            self.stepEnv(AIGameEnv.ACTION_TOGGLE)

        elif e.key() == Qt.Key_PageUp:
            self.plusReward()
        elif e.key() == Qt.Key_PageDown:
            self.minusReward()

    def mousePressEvent(self, event):
        """
        Clear the focus of the text boxes if somewhere
        else on the window is clicked
        """

        focused = QApplication.focusWidget()
        if focused == self.missionBox:
            self.missionBox.clearFocus()
        if focused == self.adviceBox:
            self.adviceBox.clearFocus()
        QMainWindow.mousePressEvent(self, event)

    def missionEdit(self):
        text = self.missionBox.toPlainText()
        print('new mission: ' + text)
        self.state = State(self.state.image, text, self.state.advice)

    def adviceEdit(self):
        text = self.adviceBox.toPlainText()
        print('new advice: ' + text)
        self.state = State(self.state.image, self.state.mission, text)
        self.env.setAdvice(text)

    def plusReward(self):
        print('+reward')
        self.env.setReward(1)

    def minusReward(self):
        print('-reward')
        self.env.setReward(-1)

    def stepClicked(self):
        self.stepEnv(action=None)

    def setFrameRate(self, value):
        """Set the frame rate limit. Zero for manual stepping."""

        print('Set frame rate: %s' % value)

        self.fpsLimit = int(value)

        if value == 0:
            self.fpsLabel.setText("Manual")
            self.stepTimer.stop()

        elif value == 100:
            self.fpsLabel.setText("Fastest")
            self.stepTimer.setInterval(0)
            self.stepTimer.start()

        else:
            self.fpsLabel.setText("%s FPS" % value)
            self.stepTimer.setInterval(int(1000 / self.fpsLimit))
            self.stepTimer.start()

    def resetEnv(self):
        obs = self.env.reset()

        self.showEnv(obs)

        mission = "Get to the green goal square"
        self.state = State(obs, mission, mission)
        self.missionBox.setPlainText(mission)

    def showEnv(self, obs):
        stepsRem = self.env.getStepsRemaining()

        # Render and display the environment
        pixmap = self.env.render()
        self.imgLabel.setPixmap(pixmap)

        # Display the agent's view
        obsW = obs.shape[0]
        obsH = obs.shape[1]
        obsImg = QImage(obsW, obsH, QImage.Format_ARGB32_Premultiplied)
        for y in range(0, obsH):
            for x in range(0, obsW):
                r = int(obs[x, y, 0])
                g = int(obs[x, y, 1])
                b = int(obs[x, y, 2])
                # ARGB
                pix = (255 << 24) + (r << 16) + (g << 8) + (b << 0)
                obsImg.setPixel(x, y, pix)
        self.obsImgLabel.setPixmap(QPixmap.fromImage(obsImg))

        # Set the steps remaining display
        self.stepsLabel.setText(str(stepsRem))

    def stepEnv(self, action=None):
        #print('stepEnv')
        #print('action=%s' % action)

        prevState = self.state

        # If no manual action was specified by the user
        if action == None:
            action = selectAction(self.state)

        obs, reward, done, info = self.env.step(action)
        #print(reward)

        self.showEnv(obs)

        newState = State(obs, prevState.mission, "")

        # Store the state transition and reward
        storeTrans(prevState, action, newState, reward)

        if done:
            self.resetEnv()

    def stepLoop(self):
        """Auto stepping loop, runs in its own thread"""

        print('stepLoop')

        while True:
            if self.fpsLimit == 0:
                time.sleep(0.1)
                continue

            if self.fpsLimit < 100:
                time.sleep(0.1)

            self.stepEnv()


def main():
    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow()

    # Run the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
