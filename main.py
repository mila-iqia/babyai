#!/usr/bin/env python3

from __future__ import division, print_function

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

import gym
from gym_aigame.envs import AIGameEnv
from training import State, selectAction, storeTrans

class AIGameWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

        self.env = gym.make('AI-Game-v0')

        self.state = None

        self.resetEnv()

    def initUI(self):
        """Create and connect the UI elements"""

        self.resize(512, 512)
        self.setWindowTitle('Baby AI Game')

        # Full render view (large view)
        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        # Area on the right of the large view
        rightBox = self.createRightArea()

        # Arrange widgets horizontally
        hbox = QHBoxLayout()
        hbox.addWidget(self.imgLabel)
        hbox.addLayout(rightBox)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainWidget.setLayout(hbox)

        # Show the application window
        self.show()
        self.setFocus()

    def createRightArea(self):
        # Agent observation view (small view)
        self.obsLabel = QLabel()
        self.obsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

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

        # Center the agent view
        obsBox = QHBoxLayout()
        obsBox.addStretch(1)
        obsBox.addWidget(self.obsLabel)
        obsBox.addStretch(1)

        # Stack everything up in a vetical layout
        vbox = QVBoxLayout()
        vbox.addLayout(obsBox)
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
        stepButton.clicked.connect(self.stepEnv)

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
        elif e.key() == Qt.Key_Down:
            self.stepEnv(AIGameEnv.ACTION_BACK)
        elif e.key() == Qt.Key_Space:
            self.stepEnv(AIGameEnv.ACTION_PICKUP)
        elif e.key() == Qt.Key_Return:
            self.stepEnv(AIGameEnv.ACTION_TOGGLE)

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

    def plusReward(self):
        print('+reward')
        # TODO

    def minusReward(self):
        print('-reward')
        # TODO

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
        """Set the image to be displayed in the full render area"""
        self.imgLabel.setPixmap(pixmap)

    def setObsPixmap(self, pixmap):
        """Set the image to be displayed in the agent observation area"""
        self.obsLabel.setPixmap(pixmap)

    def resetEnv(self):
        obs = self.env.reset()

        self.showEnv(obs)

        mission = "Get to the green goal square"
        self.state = State(obs, mission, "")
        self.missionBox.setPlainText(mission)

    def showEnv(self, obs):
        # Render and display the environment
        self.env.render()
        self.setPixmap(self.env.renderer.getPixmap())

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
        self.setObsPixmap(QPixmap.fromImage(obsImg))

        # Set the steps remaining display
        stepsRem = self.env.getStepsRemaining()
        self.stepsLabel.setText(str(stepsRem))

    def stepEnv(self, action=None):
        print('step')

        prevState = self.state

        # If no manual action was specified by the user
        if action == None:
            action = selectAction(self.state)

        obs, reward, done, info = self.env.step(action)

        self.showEnv(obs)

        print(reward)

        newState = State(obs, prevState.mission, "")

        storeTrans(prevState, action, newState, reward)

        if done:
            self.resetEnv()


def main():
    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow()

    # Run the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
