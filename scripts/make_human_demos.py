#!/usr/bin/env python3

import sys
import copy
import random
import argparse
import gym
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout

import babyai.utils as utils
import blosc

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be loaded (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--shift", type=int, default=None,
                    help="number of times the environment is reset at the beginning (default: NUM_DEMOS)")
parser.add_argument("--full-view", action="store_true", default=False,
                    help="show the full environment view")
args = parser.parse_args()

class ImgWidget(QLabel):
    """
    Widget to intercept clicks on the full image view
    """
    def __init__(self, window):
        super().__init__()
        self.window = window

class AIGameWindow(QMainWindow):
    """Application window for the baby AI game"""

    def __init__(self, env):
        super().__init__()
        self.initUI()

        # By default, manual stepping only
        self.fpsLimit = 0

        self.env = env
        self.lastObs = None

        # Demonstrations
        self.demos_path = utils.get_demos_path(args.demos, args.env, origin="human", valid=False)
        self.demos = utils.load_demos(self.demos_path, raise_not_found=False)
        utils.synthesize_demos(self.demos)


        self.shift = len(self.demos) if args.shift is None else args.shift

        self.shiftEnv()

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

        buttonBox = self.createButtons()

        self.stepsLabel = QLabel()
        self.stepsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stepsLabel.setAlignment(Qt.AlignCenter)
        self.stepsLabel.setMinimumSize(60, 10)
        restartBtn = QPushButton("Restart")
        restartBtn.clicked.connect(self.shiftEnv)
        stepsBox = QHBoxLayout()
        stepsBox.addStretch(1)
        stepsBox.addWidget(QLabel("Steps remaining"))
        stepsBox.addWidget(self.stepsLabel)
        stepsBox.addWidget(restartBtn)
        stepsBox.addStretch(1)
        stepsBox.addStretch(1)

        hline2 = QFrame()
        hline2.setFrameShape(QFrame.HLine)
        hline2.setFrameShadow(QFrame.Sunken)

        # Stack everything up in a vetical layout
        vbox = QVBoxLayout()
        vbox.addLayout(miniViewBox)
        vbox.addLayout(stepsBox)
        vbox.addWidget(hline2)
        vbox.addWidget(QLabel(""))
        vbox.addWidget(self.missionBox)
        vbox.addLayout(buttonBox)

        return vbox

    def createButtons(self):
        """Create the row of UI buttons"""

        # Assemble the buttons into a horizontal layout
        hbox = QHBoxLayout()
        hbox.addStretch(1)
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

        elif e.key() == Qt.Key_Backspace:
            self.shiftEnv()
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

    def shiftEnv(self):
        assert self.shift <= len(self.demos)

        self.env.seed(args.seed)
        self.resetEnv()
        for _ in range(self.shift):
            self.resetEnv()

    def resetEnv(self):
        self.current_demo = []

        self.current_demo = []
        self.current_actions = []
        self.current_images = []
        self.current_directions = []

        obs = self.env.reset()
        self.lastObs = obs
        self.showEnv(obs)

        self.current_mission = obs['mission']

        self.missionBox.setText(obs["mission"])

    def showEnv(self, obs):
        unwrapped = self.env.unwrapped

        # Render and display the environment
        if args.full_view:
            pixmap = self.env.render(mode='pixmap')
            self.imgLabel.setPixmap(pixmap)

        # Render and display the agent's view
        image = obs['image']
        obsPixmap = unwrapped.get_obs_render(image)
        self.obsImgLabel.setPixmap(obsPixmap)

        # Set the steps remaining
        stepsRem = unwrapped.steps_remaining
        self.stepsLabel.setText(str(stepsRem))

    def stepEnv(self, action=None):
        # If no manual action was specified by the user
        if action is None:
            action = random.randint(0, self.env.action_space.n - 1)
        action = int(action)

        obs, reward, done, info = self.env.step(action)

        self.current_actions.append(action)
        self.current_images.append(obs['image'])
        self.current_directions.append(obs['direction'])

        self.showEnv(obs)
        self.lastObs = obs

        if done:
            if reward > 0:  # i.e. we did not lose
                if self.shift < len(self.demos):
                    self.demos[self.shift] = self.current_demo, self.shift
                else:
                    self.demos.append((self.current_mission,
                                       blosc.pack_array(np.array(self.current_images)),
                                       self.current_directions,
                                       self.current_actions))
                utils.save_demos(self.demos, self.demos_path)
                self.missionBox.append('Demonstrations are saved.')
                utils.synthesize_demos(self.demos)

                self.shift += 1
                self.resetEnv()
            else:
                self.shiftEnv()

def main(argv):
    # Generate environment
    env = gym.make(args.env)

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(env)

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(sys.argv)
