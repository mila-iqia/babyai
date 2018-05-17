#!/usr/bin/env python3

import sys
import copy
import random
from optparse import OptionParser

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout

import time

import datetime

# Gym environment used by the Baby AI Game
import gym
from gym_minigrid import minigrid

import utils

import levels

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

    def __init__(self, env, env_name, seed):
        super().__init__()
        self.initUI()

        # By default, manual stepping only
        self.fpsLimit = 0

        self.env = env
        self.lastObs = None

        # Demonstrations data
        self.demos = []
        self.env_name = env_name
        self.seed = seed

        self.current_demo = []

        self.env.seed(self.seed)
        self.resetEnv(starting=2)

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
        restartBtn.clicked.connect(self.restart)
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

        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.saveClicked)

        # Assemble the buttons into a horizontal layout
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(saveButton)
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
            self.restart()
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


    def saveClicked(self):
        self.missionBox.append('Saving...')
        self.missionBox.append('There are {} demonstrations'.format(len(self.demos)))
        len_demos = ', '.join([str(len(demo)) for demo in self.demos])
        self.missionBox.append('The demo sizes are respectively: {}.'.format(len_demos))

        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        demos_name = "{}/human/{}_demos_seed_{}_{}.demo".format(self.env_name,
                                                                len(self.demos),
                                                                self.seed,
                                                                suffix)
        if len(self.demos) > 0:
            utils.save_demos(self.demos, demos_name)
            self.missionBox.append('Saved... Please quit now !')
        else:
            self.missionBox.append('Nothing to save !')

    def restart(self):
        self.env.seed(self.seed)
        for _ in range(len(self.demos)):
            self.resetEnv(starting=0)
        self.resetEnv(starting=1)

    def resetEnv(self, starting=1):
        self.current_demo = []
        obs = self.env.reset()
        self.lastObs = obs
        self.showEnv(obs, starting=starting)


    def showEnv(self, obs, starting=0):
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
        text = "MISSION: {}. To restart on the same instance, click Restart".format(mission)
        if starting == 2:
            self.missionBox.setPlainText(text)
        elif starting == 1:
            self.missionBox.append(text)

        # Set the steps remaining
        stepsRem = unwrapped.steps_remaining
        self.stepsLabel.setText(str(stepsRem))

    def stepEnv(self, action=None):
        # If no manual action was specified by the user
        if action is None:
            action = random.randint(0, self.env.action_space.n - 1)

        obs, reward, done, info = self.env.step(action)

        self.current_demo.append((self.lastObs, action))

        self.showEnv(obs)
        self.lastObs = obs

        if done:
            self.demos.append(self.current_demo)
            self.resetEnv()

def main(argv):
    parser = OptionParser()
    parser.add_option(
        "--env-name",
        help="gym environment to load",
        default='BabyAI-FindObj-v0'
    )
    parser.add_option("--seed", type=int, default=1337,
                      help="random seed (default: 1337)")
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(env, options.env_name.split('-')[1], options.seed)

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(sys.argv)
