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
from training import selectAction

class AIGameWindow(QMainWindow):

    def __init__(
        self,
        missionEditCb,
        adviceEditCb,
        stepEnvCb,
        plusRewardCb,
        minusRewardCb,
        actionCb
    ):
        super().__init__()
        self.initUI()

        self.missionEditCb = missionEditCb
        self.adviceEditCb = adviceEditCb
        self.stepEnvCb = stepEnvCb
        self.plusRewardCb = plusRewardCb
        self.minusRewardCb = minusRewardCb
        self.actionCb = actionCb

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
        self.setFocus()

    def createImageArea(self):
        """Create the image area to render into"""

        # Full render view (large view)
        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        # Agent observation view (small view)
        self.obsLabel = QLabel()
        self.obsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        # Center the agent view vertically
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.obsLabel)
        vbox.addStretch(1)

        hbox = QHBoxLayout()
        #hbox.addStretch(1)
        hbox.addWidget(self.imgLabel)
        hbox.addStretch(1)
        hbox.addLayout(vbox)
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

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.actionCb(AIGameEnv.ACTION_LEFT)
        elif e.key() == Qt.Key_Right:
            self.actionCb(AIGameEnv.ACTION_RIGHT)
        elif e.key() == Qt.Key_Up:
            self.actionCb(AIGameEnv.ACTION_FORWARD)
        elif e.key() == Qt.Key_Down:
            self.actionCb(AIGameEnv.ACTION_BACK)
        elif e.key() == Qt.Key_Space:
            self.actionCb(AIGameEnv.ACTION_PICKUP)

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
        self.missionEditCb(self.missionBox.toPlainText())

    def adviceEdit(self):
        self.adviceEditCb(self.adviceBox.toPlainText())

    def plusReward(self):
        self.plusRewardCb()

    def minusReward(self):
        self.minusRewardCb()

    def stepButton(self):
        self.stepEnvCb()

    def setMission(self, text):
        self.missionBox.setPlainText(text)

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

def main():

    window = None
    env = None

    state = {
        "image": None,
        "mission": "",
        "advice": ""
    }

    def missionEdit(text):
        print('new mission: ' + text)
        state['mission'] = text

    def adviceEdit(text):
        print('new advice: ' + text)
        state['advice'] = text

    def plusReward():
        print('+ reward')
        # TODO: hook this up to something

    def minusReward():
        print('- reward')
        # TODO: hook this up to something

    def showEnv(obs):
        # Render and display the environment
        env.render()
        window.setPixmap(env.renderer.getPixmap())

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
        window.setObsPixmap(QPixmap.fromImage(obsImg))

    def stepEnv(action=None):
        print('step')

        # If no manual action was specified by the user
        if action == None:
            action = selectAction(state['mission'], state['advice'], state['image'])

        obs, reward, done, info = env.step(action)
        showEnv(obs)
        print(reward)

        state['image'] = obs

        if done:
            env.reset()
            obs = env.reset()
            showEnv(obs)
            state['image'] = obs

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(
        missionEditCb = missionEdit,
        adviceEditCb = adviceEdit,
        stepEnvCb = stepEnv,
        plusRewardCb = plusReward,
        minusRewardCb = minusReward,
        actionCb = stepEnv
    )

    env = gym.make('AI-Game-v0')
    obs = env.reset()
    showEnv(obs)
    state['image'] = obs

    # Initial mission is hardcoded
    window.setMission("Get to the green goal square")

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
