#!/usr/bin/env python3

import sys
import copy
import random
import argparse
import datetime
import gym
import levels
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be loaded (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--shift", type=int, default=None,
                    help="number of times the environment is reset at the beginning (default: NUM_DEMOS")
parser.add_argument("--full-view", action="store_true", default=False,
                    help="show the full environment view")
args = parser.parse_args()

def synthesize_demos(demos):
    print('{} demonstrations'.format(len(demos)))
    demo_lens = [len(demo) for demo in demos]
    if len(demos) > 0:
        print('Demo sizes: {}'.format(demo_lens))
        print('Synthesis: {}'.format(utils.synthesize(demo_lens)))

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

        # Demonstrations
        self.demos = utils.load_demos(args.env, human=True)
        synthesize_demos(self.demos)
        self.current_demo = []

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

    def shiftEnv(self):
        assert self.shift <= len(self.demos)

        self.env.seed(args.seed)
        self.resetEnv()
        for _ in range(self.shift):
            self.resetEnv()

    def resetEnv(self):
        self.current_demo = []
        
        obs = self.env.reset()
        self.lastObs = obs
        self.showEnv(obs)

        self.missionBox.setText(obs["mission"])

    def showEnv(self, obs):
        unwrapped = self.env.unwrapped

        # Render and display the environment
        if args.show_full_view:
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

        obs, reward, done, info = self.env.step(action)

        self.current_demo.append((self.lastObs, action))

        self.showEnv(obs)
        self.lastObs = obs

        if done:
            if reward > 0:  # i.e. we did not lose
                if self.shift < len(self.demos):
                    self.demos[self.shift] = self.current_demo
                else:
                    self.demos.append(self.current_demo)
                utils.save_demos(self.demos, args.env, human=True)
                self.missionBox.append('Demonstrations are saved.')
                synthesize_demos(self.demos)

                self.shift += 1
                self.resetEnv()
            else:
                self.shiftEnv()

def main(argv):
    # Load the gym environment
    env = gym.make(args.env)

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(env)

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(sys.argv)
