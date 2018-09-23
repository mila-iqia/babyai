#!/usr/bin/env python3

import time
import sys
import threading
import copy
import random
from optparse import OptionParser

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

# Gym environment used by the Baby AI Game
import gym
import gym_minigrid
from gym_minigrid import minigrid

import babyai


class ImgWidget(QLabel):
    """
    Widget to intercept clicks on the full image view
    """
    def __init__(self, window):
        super().__init__()
        self.window = window

    def mousePressEvent(self, event):
        self.window.imageClick(event.x(), event.y())


class AIGameWindow(QMainWindow):
    """Application window for the baby AI game"""

    def __init__(self, env):
        super().__init__()
        self.initUI()

        # By default, manual stepping only
        self.fpsLimit = 0

        self.env = env
        self.lastObs = None

        self.resetEnv()

        self.stepTimer = QTimer()
        self.stepTimer.setInterval(0)
        self.stepTimer.setSingleShot(False)
        self.stepTimer.timeout.connect(self.stepClicked)

        # Pointing and naming data
        self.pointingData = []

    def initUI(self):
        """Create and connect the UI elements"""

        self.resize(512, 512)
        self.setWindowTitle('Baby AI Game')

        # Full render view (large view)
        self.imgLabel = ImgWidget(self)
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

        buttonBox = self.createButtons()

        self.stepsLabel = QLabel()
        self.stepsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stepsLabel.setAlignment(Qt.AlignCenter)
        self.stepsLabel.setMinimumSize(60, 10)
        resetBtn = QPushButton("Reset")
        resetBtn.clicked.connect(self.resetEnv)
        stepsBox = QHBoxLayout()
        stepsBox.addStretch(1)
        stepsBox.addWidget(QLabel("Steps remaining"))
        stepsBox.addWidget(self.stepsLabel)
        stepsBox.addWidget(resetBtn)
        stepsBox.addStretch(1)

        hline2 = QFrame()
        hline2.setFrameShape(QFrame.HLine)
        hline2.setFrameShadow(QFrame.Sunken)

        # Stack everything up in a vetical layout
        vbox = QVBoxLayout()
        vbox.addLayout(miniViewBox)
        vbox.addLayout(stepsBox)
        vbox.addWidget(hline2)
        vbox.addWidget(QLabel("Mission"))
        vbox.addWidget(self.missionBox)
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
        # Manual agent control
        actions = self.env.unwrapped.actions

        if e.key() == Qt.Key_Left:
            self.stepEnv(actions.left)
        elif e.key() == Qt.Key_Right:
            self.stepEnv(actions.right)
        elif e.key() == Qt.Key_Up:
            self.stepEnv(actions.forward)

        elif e.key() == Qt.Key_PageUp:
            self.stepEnv(actions.pickup)
        elif e.key() == Qt.Key_PageDown:
            self.stepEnv(actions.drop)
        elif e.key() == Qt.Key_Space:
            self.stepEnv(actions.toggle)
        elif e.key() == Qt.Key_Return:
            self.stepEnv(actions.done)

        elif e.key() == Qt.Key_Backspace:
            self.resetEnv()
        elif e.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        """
        Clear the focus of the text boxes and buttons if somewhere
        else on the window is clicked
        """

        # Set the focus on the full render image
        self.imgLabel.setFocus()

        QMainWindow.mousePressEvent(self, event)

    def imageClick(self, x, y):
        """
        Pointing and naming logic
        """

        # Set the focus on the full render image
        self.imgLabel.setFocus()

        env = self.env.unwrapped
        imgW = self.imgLabel.size().width()
        imgH = self.imgLabel.size().height()

        i = (env.grid.width * x) // imgW
        j = (env.grid.height * y) // imgH
        assert i < env.grid.width
        assert j < env.grid.height

        print('grid clicked: i=%d, j=%d' % (i, j))

        desc, ok = QInputDialog.getText(self, 'Pointing & Naming', 'Enter Description:')
        desc = str(desc)

        if not ok or len(desc) == 0:
            return

        pointObj = env.grid.get(i, j)

        if pointObj is None:
            return

        print('description: "%s"' % desc)
        print('object: %s %s' % (pointObj.color, pointObj.type))

        viewSz = minigrid.AGENT_VIEW_SIZE

        NUM_TARGET = 50
        numItrs = 0
        numPos = 0
        numNeg = 0

        while (numPos < NUM_TARGET or numNeg < NUM_TARGET) and numItrs < 300:
            env2 = copy.deepcopy(env)

            # Randomly place the agent around the selected point
            x, y = i, j
            x += random.randint(-viewSz, viewSz)
            y += random.randint(-viewSz, viewSz)
            x = max(0, min(x, env2.grid.width - 1))
            y = max(0, min(y, env2.grid.height - 1))
            env2.agent_pos = (x, y)
            env2.agent_dir = random.randint(0, 3)

            # Don't want to place the agent on top of something
            if env2.grid.get(*env2.agent_pos) != None:
                continue

            agent_sees = env2.agent_sees(i, j)

            obs = env2.gen_obs()
            img = obs['image'] if isinstance(obs, dict) else obs
            obsGrid = minigrid.Grid.decode(img)

            datum = {
                'desc': desc,
                'img': img,
                'pos': (i, j),
                'present': agent_sees
            }

            if agent_sees and numPos < NUM_TARGET:
                self.pointingData.append(datum)
                numPos += 1

            if not agent_sees and numNeg < NUM_TARGET:
                # Don't want identical object in mismatch examples
                if (pointObj.color, pointObj.type) not in obsGrid:
                    self.pointingData.append(datum)
                    numNeg += 1

            numItrs += 1

        print('positive examples: %d' % numPos)
        print('negative examples: %d' % numNeg)
        print('total examples: %d' % len(self.pointingData))

    def missionEdit(self):
        # The agent will get the mission as an observation
        # before performing the next action
        text = self.missionBox.toPlainText()
        self.lastObs['mission'] = text

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
        self.lastObs = obs
        self.showEnv(obs)

    def showEnv(self, obs):
        unwrapped = self.env.unwrapped

        # Render and display the environment
        pixmap = self.env.render(mode='pixmap')
        self.imgLabel.setPixmap(pixmap)

        # Render and display the agent's view
        image = obs['image']
        obsPixmap = unwrapped.get_obs_render(image)
        self.obsImgLabel.setPixmap(obsPixmap)

        # Update the mission text
        mission = obs['mission']
        self.missionBox.setPlainText(mission)

        # Set the steps remaining
        stepsRem = unwrapped.steps_remaining
        self.stepsLabel.setText(str(stepsRem))

    def stepEnv(self, action=None):
        # If no manual action was specified by the user
        if action == None:
            action = random.randint(0, self.env.action_space.n - 1)

        obs, reward, done, info = self.env.step(action)

        self.showEnv(obs)
        self.lastObs = obs

        if done:
            self.resetEnv()


def main(argv):
    parser = OptionParser()
    parser.add_option(
        "--env-name",
        help="gym environment to load",
        default='BabyAI-BossLevel-v0'
    )
    parser.add_option(
        "--seed",
        type=int,
        help="gym environment seed",
        default=0
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)
    env.seed(options.seed)

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(env)

    # Run the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv)
