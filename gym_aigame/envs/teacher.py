import pickle
import gym
from gym import Wrapper
import pickle
import gym
from gym import Wrapper
import numpy as np
import sys
import os
directory=os.getcwd()
directory=directory+'/gym_aigame/envs'
if not directory in sys.path:
    sys.path.insert(0,directory)
print("adding directory path")


import shortestPath



class Teacher(Wrapper):
    def __init__(self, env):
        super(Teacher, self).__init__(env)
       
        self.Dico={'continue':['continue',"go forward", 'go on', 'keep the same direction!', 'okay continue this way','advance one case'],
                   'left':['go left','turn left', 'now go to the left', 'go to the left!...The other left!!!', 'go on your left'],
                   'right':['go right', 'turn right', 'now go to the right', 'tous à Tribor !!', 'go on your right','good job! now go to the right'],
                   'turn back': ['turn back','go backward','turn yourself', 'make a U turn', 'look behind you...' ],             
                   'key':['take the key!', 'pick up the key', 'pick it up', 'take it', 'key !!!', 'catch the key','toggle the key'   ],
                   'door':['open the door', 'toggle the door', 'open it', 'tire la chevillette et la bobinette cherra...']}

    def _close(self):
        super(Teacher, self)._close()
        
        
    def _reset(self, **kwargs):
        """
        Called at the start of an episode
        """
        
    
        obs = self.env.reset(**kwargs)
        
        if not isinstance(obs, dict):
                obs = { "image": obs, 'mission':'' }
            
        obs['advice']=self.generateAdvice()[1]
        
        return (obs)

    
    def _step(self, action):
        """
        Called at every action
        """
        
        obs, reward, done, info = self.env.step(action)

        
        advice=self.generateAdvice()[1]

        obs = {
            "image": obs,
            "advice": advice
        }        



        return obs, reward, done, info

    
    
    
    

 
    def chooseExpression(self, keyOfDic):
        value=self.Dico[keyOfDic]
        numbers=len(value)
        idx=np.random.randint(0,numbers)
        return(value[idx])
        

       
    
    def pickUpTheKey(self):
        if self.env.carrying == None:
            return((False, self.chooseExpression("key")))
        else:
            return((True,""))      
            
    def ToggleTheDoor(self,doorPos):
        if (not self.env.grid.get(doorPos[0],doorPos[1]).isOpen):
            return((False, self.chooseExpression("door")))
        else:
            return((True,"door already opened"))  
            
            
            
    
    def goDown(self):
        DirVec=self.env.getDirVec()
        if DirVec==(1,0):
            return(self.chooseExpression("right"))
        elif DirVec==(-1,0):
            return(self.chooseExpression("left"))
        elif DirVec==(0,1):
            return(self.chooseExpression("continue"))
        elif DirVec==(0,-1):
            return(self.chooseExpression("turn back"))
        
    
    def goRight(self):
        DirVec=self.env.getDirVec()
        if DirVec==(1,0):
            return(self.chooseExpression("continue"))
        elif DirVec==(-1,0):
            return(self.chooseExpression("turn back"))
        elif DirVec==(0,1):
            return(self.chooseExpression("left"))
        elif DirVec==(0,-1):
            return(self.chooseExpression("right"))
            
    def goUp(self):
        DirVec=self.env.getDirVec()
        if DirVec==(1,0):
            return(self.chooseExpression("left"))
        elif DirVec==(-1,0):
            return(self.chooseExpression("right"))
        elif DirVec==(0,1):
            return(self.chooseExpression("turn back"))
        elif DirVec==(0,-1):
            return(self.chooseExpression("continue"))
    
    def goLeft(self):
        DirVec=self.env.getDirVec()
        if DirVec==(1,0):
            return(self.chooseExpression("turn back"))
        elif DirVec==(-1,0):
            return(self.chooseExpression("continue"))
        elif DirVec==(0,1):
            return(self.chooseExpression("right"))
        elif DirVec==(0,-1):
            return(self.chooseExpression("left"))
    
    def inFrontOf(self,ojectivePos):
        #return a tuple (bool,advice), if bool=True the agent is in front of the obective, else the advice
        #helps him to set his position
        
        
        #get the desired direction
        currentPos=self.env.agentPos
        delta=ojectivePos[0]-currentPos[0],ojectivePos[1]-currentPos[1]
        #print("delta : ", delta)
        if delta==(1,0):
            obj=0
        elif delta==(-1,0):
            obj=2
        elif delta==(0,1):
            obj=1
        elif delta==(0,-1):
            obj=3
        
        #working modulo 4 on the agent dir
        currentDir=self.env.agentDir
        diff=(obj-currentDir)%4
        if diff==0:
            return((True,''))
        elif diff==1:
            return((False,self.chooseExpression("right")))
        elif diff==2:
            return((False,self.chooseExpression("turn back")))
        elif diff==3:
            return((False,self.chooseExpression("left")))
                
        
    def getPosKey(self):
        for i in range(self.env.gridSize):
            for j in range(self.env.gridSize):
                if (self.env.grid.get(i,j) != None):
                    if self.env.grid.get(i,j).type == 'key':
                        outputPos=(i,j)                        
                        print(" key found in position : ", outputPos)
                        return(outputPos)
                        
        print("key not found...")
        return(False)
        
    def getPosDoor(self):
        for i in range(self.env.gridSize):
            for j in range(self.env.gridSize):
                if (self.env.grid.get(i,j) != None):
                    if self.env.grid.get(i,j).type == 'door':
                        outputPos=(i,j)                        
                        return(outputPos)
                        
        print("door not found...")
        return(False)
                
        
#    def nextTo(self,objectivePos):
#        currentPos=self.env.agentPos
#        delta=objectivePos[0]-currentPos[0],objectivePos[1]-currentPos[1]
#
#
#        if np.abs(delta[0])>1:
#            if delta[0]>0:
#                return((False,self.goRight()))
#            elif delta[0]<0:
#                return((False,self.goLeft()))
#            
#        if (np.abs(delta[1])>1 or (np.abs(delta[1])==1 and np.abs(delta[0])==1)):
#            if delta[1]>0:
#                return((False,self.goDown()))
#            elif delta[1]<0:
#                return((False,self.goUp()))
#        
#        #print("you reached your objective, you need to get the right orientation now")
#        return((True,''))
#        
#        
        
    def nextTo(self, objective):
        currentPos=self.env.agentPos
        delta=objective[0]-currentPos[0],objective[1]-currentPos[1]
        
        if (np.abs(delta[1]) + np.abs(delta[0]) >1):                
            img=self.state2maze()        
            start=self.env.agentPos[1],self.env.agentPos[0]
            goal=objective[1],objective[0]
            seq=shortestPath.find_path_bfs(img,start,goal)
            direction=seq[0]
            if direction=='N':
                return((False, self.goUp()))
            if direction=='S':
                return((False, self.goDown()))
            if direction=='E':
                return((False, self.goRight()))
            if direction=='W':
                return((False, self.goLeft()))
        else:
            return((True, ' you are next to {}'.format(objective)))
        
        
    def reach(self, objective):
        currentPos=self.env.agentPos
        delta=objective[0]-currentPos[0],objective[1]-currentPos[1]
        
        if (np.abs(delta[1]) + np.abs(delta[0]) >0):                
            img=self.state2maze()        
            start=self.env.agentPos[1],self.env.agentPos[0]
            goal=objective[1],objective[0]
            seq=shortestPath.find_path_bfs(img,start,goal)
            direction=seq[0]
            if direction=='N':
                return((False, self.goUp()))
            if direction=='S':
                return((False, self.goDown()))
            if direction=='E':
                return((False, self.goRight()))
            if direction=='W':
                return((False, self.goLeft()))
        else:
            return((True, ''))
    
#    def reach(self,objectivePos):
#        currentPos=self.env.agentPos
#        delta=objectivePos[0]-currentPos[0],objectivePos[1]-currentPos[1]
#
#
#        if np.abs(delta[0])>0:
#            if delta[0]>0:
#                return((False,self.goRight()))
#            elif delta[0]<0:
#                return((False,self.goLeft()))
#            
#        if (np.abs(delta[1])>0):
#            if delta[1]>0:
#                return((False,self.goDown()))
#            elif delta[1]<0:
#                return((False,self.goUp()))
#        
#        #print("you reached your objective, you need to get the right orientation now")
#        return((True,''))
            

        
    def getKey(self):
        objectivePos=self.env.keyPos     
        
        isNextTo,adviceDirection=self.nextTo(objectivePos)             
        if (not isNextTo):
            return((False,adviceDirection))               
        else:
            isOriented,adviceOrienation=self.inFrontOf(objectivePos)
            if (not isOriented):
                return((False,adviceOrienation))
            else:
                hasKey,adviceKey =self.pickUpTheKey()
                if (not hasKey):
                    return((False,adviceKey))        
        return((True,''))
            
            
    def openTheDoor(self,doorPos):
        objectivePos=doorPos
        isNextTo,adviceDirection=self.nextTo(objectivePos)             
        if (not isNextTo):
            return((False,adviceDirection))               
        else:
            isOriented,adviceOrienation=self.inFrontOf(objectivePos)
            if (not isOriented):
                return((False,adviceOrienation))
            else:
                doorOpen,adviceDoor =self.ToggleTheDoor(objectivePos)
                if (not doorOpen):
                    return((False,adviceDoor))        
        return((True,''))
        

        
        
    def generateAdviceWithKeys(self):
        
        doorOpen=self.env.grid.get(self.env.doorPos[0],self.env.doorPos[1]).isOpen    
        if (not doorOpen):
            hasKey=(self.env.carrying!=None)        
            if (not hasKey):
                subgoal="current sub goal : picking the key"
                hasKey,advice=self.getKey()
                #print("advice generated : ",advice)        
            else:
                subgoal="current sub goal : opening the door"
                isOpen,advice=self.openTheDoor()
                #print("advice generated : ",advice)
          
        else:
            goal=self.env.goalPos
            subgoal="current sub goal : reaching the goal"
            finished, advice=self.reach(goal)
            #print("advice generated : ",advice)
            
        #info['advice'] = advice



        #print(" ")
        #print(" ")
        return(subgoal,advice)
        
    def state2maze(self):
        img=[[0 for i in range(self.env.gridSize)] for j in range(self.env.gridSize)]
        for i in range(self.env.gridSize):
            for j in range(self.env.gridSize):
                if (self.env.grid.get(i,j) != None):
                    if (self.env.grid.get(i,j).type == 'wall'):
                        img[j][i]=1
        return(img)
    
    
    def getCurrentDorPos(self):                
        for room in reversed(self.env.rooms):
            if room.exitDoorPos !=None:
                #print(room.exitDoorPos)
                if not self.env.grid.get(room.exitDoorPos[0],room.exitDoorPos[1]).isOpen:
                    currentRoom=room
        return(currentRoom.exitDoorPos)
    
            
        
    def generateAdvice(self):
       
#        print("currently there are X rooms : ", self.env.numRooms)
#        
#        for i in range(self.env.numRooms-1):
#            print("room ", i, " exit door ", self.env.rooms[i].exitDoorPos, " opened : ",self.env.rooms[i].exitDoor.isOpen)
#         
#        for i in range(self.env.numRooms-1):
#            print("room ", i, " exit door ", self.env.rooms[i].exitDoorPos,
#                  " opened : ",self.env.grid.get(self.env.rooms[i].exitDoorPos[0],self.env.rooms[i].exitDoorPos[1]).isOpen)
#        
        
        if not self.env.grid.get(self.env.rooms[-1].entryDoorPos[0],self.env.rooms[-1].entryDoorPos[1]).isOpen:
            doorPos=self.getCurrentDorPos()
            doorOpen=self.env.grid.get(doorPos[0],doorPos[1]).isOpen   
            #print("currently searching door : ", doorPos)
        else:
            doorOpen=True
       
        
        #doorOpen=self.env.grid.get(doorPos[0],doorPos[1]).isOpen   
        if (not doorOpen):
            subgoal="current sub goal : reaching the door in {}".format(doorPos)
            isOpen,advice=self.openTheDoor(doorPos)
                #print("advice generated : ",advice)
          
        else:
            goal=self.env.goalPos
            subgoal="current sub goal : reaching the goal"
            finished, advice=self.reach(goal)
            #print("advice generated : ",advice)
       
        
        print("advice generated : ", advice)
        #info['advice'] = advice

        #advice=seq

        print(" ")
        return(subgoal,advice)


