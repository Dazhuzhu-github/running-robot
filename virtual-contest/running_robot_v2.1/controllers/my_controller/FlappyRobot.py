# -*- coding:UTF-8 -*-
from controller import Robot

import os
import sys
import time

import numpy as np
from sklearn.linear_model import TheilSenRegressor as Model

try:
    import cv2 as cv
except:
    ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    if ros_path in sys.path:
        sys.path.remove(ros_path)
        import cv2 as cv
        sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
    else:
        raise("UNKNOWN ERROR: attention, this is not caused by ROS")

 
libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op', 'libraries',
                           'python37')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

# color 
COLOR_RANGE = [[(0, 0), (0, 0), (0, 0)],# Red
               [(0, 0), (0, 0), (0, 0)],# Green
               [(0, 0), (0, 0), (0, 0)],# Blue
               [(0, 0), (0, 0), (0, 0)],# white
               [(0, 0), (0, 0), (0, 0)],# Gray
               [(0, 0), (0, 0), (0, 0)],# Grass
               [(0, 0), (0, 0), (0, 0)],# wood
               [(0, 0), (0, 0), (0, 0)]]# Cloth
class Walk():
    def __init__(self):
        self.robot = Robot()  # 初始化Robot类以控制机器人
        self.mTimeStep = int(4*self.robot.getBasicTimeStep())  # 获取当前每一个仿真步所仿真时间mTimeStep
        self.mTimeStep2 = int(self.robot.getBasicTimeStep())
        self.HeadLed = self.robot.getLED('HeadLed')  # 获取头部LED灯
        self.EyeLed = self.robot.getLED('EyeLed')  # 获取眼部LED灯
        self.HeadLed.set(0xff0000)  # 点亮头部LED灯并设置一个颜色
        self.EyeLed.set(0xa0a0ff)  # 点亮眼部LED灯并设置一个颜色
        self.mAccelerometer = self.robot.getAccelerometer('Accelerometer')  # 获取加速度传感器
        self.mAccelerometer.enable(self.mTimeStep)  # 激活传感器，并以mTimeStep为周期更新数值

        # check fallen
        self.fup = 0
        self.fdown = 0 
        self.flr = 0 
        self.falldownMark = False 
        self.checkFallen_Enable = True  #

        # movement
        self.maxStepPerGroup = 102#93#86  # 
        self.maxStepPerGroupWhenWalk = 205
        self.midStepPerGroup = 86
        self.stepCounterPerGroup = 1 # 
        self.checkRelax_Enable = True  # 
        
        # navigation
        self.approxLocation = 0 
        self.yawIntegrater = 0
        self.valuePer2PI = 3480
        self.numberToConfirmLocation = 50 # 
        self.counterForConfirmLocation = 0
        self.approxAngleRange = [[0, self.valuePer2PI/4],
                                 [self.valuePer2PI/4, self.valuePer2PI/2],
                                 [self.valuePer2PI/2, self.valuePer2PI*3/4],
                                 [self.valuePer2PI*3/4, self.valuePer2PI]]
        self.checkGyro_Enable = True  # 
        self.navigationLevel = 0.5 
        
        # vision
        self.camera = self.robot.getCamera('Camera')
        self.camera.enable(self.mTimeStep)
        self.width = self.camera.getWidth() # 160
        self.height = self.camera.getHeight() # 120
        self.frame = None # 
        self.downSampledFrame = None # 
        self.frameHSV = None # 
        self.downSampledFrameHSV = None # 
        self.COLOR_RANGE = {
            "RED": [(0, 165, 100), (8, 255, 255)],  # Red
            "auxRED": [(172, 92, 100), (180, 255, 255)],  # Red
            "BLUE": [(102, 75, 78), (124, 255, 255)],  # Green
            "GREEN": [(35, 43, 84), (77, 255, 255)],  # Blue
            "auxGREEN": [(35, 43, 84), (77, 255, 255)],  # Blue
            "WHITE": [(0, 0, 207), (180, 20, 255)],  # white
            "GRAY": [(0, 5, 80), (180, 9, 218)],  # Gray
            "GRASS": [(35, 43, 84), (77, 255, 255)],  # Grass
            "WOOD": [(11, 43, 84), (25, 255, 255)],  # wood
            "CLOTH": [(112, 20, 40), (150, 85, 230)],  # Cloth
            "YELLOW": [(26, 43, 84), (34, 255, 255)],  # YELLOW
            "BLACK": [(0, 0, 0), (180, 255, 74)],  # BLACK
            "DOOR": [(0, 0, 0), (180, 255, 46)]  # BLACK
        }
        if self.navigationLevel == 0.5:
            self.groundColor = ["BLUE", "GREEN", "WHITE", "GRAY", "GRASS", "WOOD", "CLOTH"]
        else:
            self.groundColor = ["BLUE", "auxGREEN", "WHITE", "GRAY", "GRASS", "WOOD", "CLOTH"]
        self.singleColorMask = {
            "RED": None,  # Red
            "auxRED": None,  #
            "BLUE": None,  # Green
            "GREEN": None,  # Blue
            "auxGREEN": None,  #
            "WHITE": None,  # white
            "GRAY": None,  # Gray
            "GRASS": None,  # Grass
            "WOOD": None,  # wood
            "CLOTH": None,  # Cloth
            "YELLOW": None,  # YELLOW
            "BLACK": None,  # BLACK
            "DOOR": None
        }  # 
        self.multiColorMask = None # 

        # adjust position
        self.headUpForNavigation = 0.52
        self.headMidForNavigation = 0.3
        self.headDownForNavigation = 0.1
        self.neckLeftForNavigation = 0.5
        self.neckMidForNavigation = 0.
        self.minAdjustStepA = 20 # 
        self.usualAdjustStepA = 29# 9#30 # 
        self.usualAdjustStepY = 96# 
        self.usualAdjustStepX = 50  # 
        self.stepCounterPerAdjust = 0
        self.rho = 0  # 
        self.theta = 0  #
        self.horizontalRho = 0 # 
        self.maxBias = 0.1  # 
        self.holdStepWhenAdjust = False # 
        self.oneshotAdjust = False 
        self.auxAdjustEnable = False
        
        self.auxAdjustEnable2 = False
        self.auxStdRho = 0 # 
        self.auxStdTheta = np.pi / 4 # 
        self.avoidanceROI = [0, self.width//4]
        self.barrierGamma = 3 # 
        self.biasNavi05 = 0.22

        # section
        self.nextSction = -10 # 
        self.strategyLock = 0  # 
        self.sectionLock = False
        self.sectionFlag = [False] * 8 # 

        # section 0
        self.needtoConfirmStickState = False
        self.roiSction1 = [self.height*3//(4*4), self.height] # 
        self.thresSection0 = 0.8 # 
        self.initStepSection0 = 105#86#50  # 

        # section 1
        self.thresEnterSection1 = 320 # 
        self.thresExitSection1 = 10  #
        self.biasRhoSection1 = - 0.32# 0.39 # - 0.24 @head=0.46#0.42
        self.biasThetaSection1 = -0.34 # 0.13 @head=0.46
        self.biasRhoSection1_ = - 0.31  # 
        self.biasThetaSection1_ = 0.41  #

        # section 2
        self.finished = True
        self.mine_cascade = cv.CascadeClassifier('cascade_mines.xml')

        # section 3
        self.thresSection3 = 0.9
        self.thresEnterSection3 = 100
        self.rhoYSection3 = - 0.66
        self.thresAtCorner = True

        # section 4
        self.thresExitSection4 = 8 # 
        self.roiSction4 = [0, self.height//(3*4)]
        self.thresEnterSection4 = 20  # 
        self.thresSection4 = 0.86  #
        self.tempRhoSection4 = None
        self.tempThetaSection4 = None
        self.stepsForConfirmSection4 = self.maxStepPerGroup # 
        self.supposeInCenter = True # 

        # section 5
        self.thresEnterSection5 = 360  #
        self.thresExitSection5 = 125  # 
        self.roiSction5 = [self.width//(4*4), self.width*3//(4*4)] # 
        self.thresSection5 = 0.7
        self.treatBridgeAsGround = True
        
        # section 6（踢球）

        # section 7（台阶）
        self.thresEnterSection7 = 410  # 
        self.roiSction7 = [self.height // (4 * 2), self.height//4-1]  # 
        self.rhoYSection71 = 0.08
        self.rhoYSection72 = -0.3 # 
        self.thresSection7 = 0.8

        self.mGyro = self.robot.getGyro('Gyro')  #
        self.mGyro.enable(self.mTimeStep)  #

        self.positionSensors = []  # 初始化关节角度传感器
        self.mMotor = []
        self.minMotorPositions = []
        self.maxMotorPositions = []
        self.positionSensorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                                    'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', # 胯骨Z轴（均为左手性）
                                    'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL', # 胯骨X轴（均为左手性）， 胯骨Y轴（左胯左手性）
                                    'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL', # 膝关节（左膝左手性）， 脚踝Y轴（右脚踝左手性）
                                    'FootR', 'FootL', 'Neck', 'Head')  # 脚踝X轴， 脖子（左右）， 头（头是左手性）

        # 获取各传感器并激活，以mTimeStep为周期更新数值
        for i in range(0, len(self.positionSensorNames)):
            self.positionSensors.append(self.robot.getPositionSensor(self.positionSensorNames[i] + 'S'))
            self.positionSensors[i].enable(self.mTimeStep)

            self.mMotor.append(self.robot.getMotor(self.positionSensorNames[i]))
            self.minMotorPositions.append(self.mMotor[i].getMinPosition())
            self.maxMotorPositions.append(self.mMotor[i].getMaxPosition())

        
        self.mMotionManager = RobotisOp2MotionManager(self.robot)  # 初始化机器人动作组控制器
        self.mGaitManager = RobotisOp2GaitManager(self.robot, "config.ini")  # 初始化机器人步态控制器

    def myStep(self):
        """
      
        :return:
        """
        ret = self.robot.step(self.mTimeStep)
        if ret == -1:
            exit(0)
        self.checkIfFallen()
        if self.navigationLevel != 0:
            self.gyroCheck()
            
    def myStep2(self):
        """
       
        :return:
        """
        ret = self.robot.step(self.mTimeStep2)
        if ret == -1:
            exit(0)
        self.checkIfFallen()
        if self.navigationLevel != 0:
            self.gyroCheck()


    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms / 1000.0
        while s + startTime >= self.robot.getTime():
            self.myStep()


    def relax(self, headLeft=True, force=False):
        """
        
        :param headLeft: 
        :param force: 
        :return:
        """
        if (((self.stepCounterPerGroup > self.maxStepPerGroup and (
                self.sectionLock or (not self.sectionLock and self.nextSction != -10))) or (
                     self.stepCounterPerGroup > self.maxStepPerGroupWhenWalk and not self.sectionLock)) and self.checkRelax_Enable) or force:
            print("relax")
            self.mGaitManager.stop()
            self.wait(150)
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            if self.strategyLock < 1:
                if self.oneshotAdjust or not headLeft or (self.navigationLevel == 0.5 and not self.sectionLock):
                   
                    self.mMotor[19].setPosition(self.headUpForNavigation)
                    self.mMotor[18].setPosition(self.neckMidForNavigation)
                else:
                    self.mMotor[19].setPosition(self.headMidForNavigation)
                    self.mMotor[18].setPosition(self.neckLeftForNavigation)
            if self.navigationLevel != 0.5 or self.sectionLock:
                self.wait(480)  # 等待200ms
            else:
                self.wait(100)
            self.stepCounterPerGroup = 0
            if self.nextSction != -10:
                print("change strategy to No."+str(self.nextSction))
                self.strategyLock = self.nextSction
                self.nextSction = -10
                self.stepCounterPerGroup = 1
            return 1
        else:
            self.stepCounterPerGroup += 1
            return 0

    def findBarrier(self):
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))
        barrier = np.array(cv.erode(self.multiColorMask, kernel))
        cv.imshow("barrier", barrier)
        cv.waitKey(1)
        barrier = np.where(barrier==0, 1, 0)
       
        weightY = np.sum(np.sum(barrier, axis=1) * (np.array(range(barrier.shape[0]), dtype=np.float) / (barrier.shape[0]*2)+0.5) ** self.barrierGamma) / (np.sum(barrier) + 1e-8)
        barrier = barrier[:, self.avoidanceROI[0]:self.avoidanceROI[1]]
        weightX = np.sum(np.sum(barrier, axis=0) * ((np.append(np.array(range(barrier.shape[1]//2), dtype=np.float) / (1.25*barrier.shape[1])+0.6, np.flipud(-np.array(range(barrier.shape[1]//2), dtype=np.float) / (1.25*barrier.shape[1])-0.6))) ** self.barrierGamma)) / (np.sum(barrier) + 1e-8)
        weightTarget = (np.sum(barrier) / (250))**2.6
        if weightTarget < 1:
            weightTarget = 1
        return weightX * weightY*weightTarget
        
        
        
    def adjust(self, biasRho=0., biasTheta=0., color="GROUND", axis=1, lrSingleAdjust=False, thetaFirst=True, allowRepeat=False, force=False):
        """
        
        :param b
        :return:
        """
        if self.navigationLevel == 0.5 and not self.sectionLock:
            strategyRecoder = self.strategyLock
            while True:
                intensity = self.findBarrier()
                intensity -= 0.05 # 相
                if intensity > self.biasNavi05:
                    self.strategyLock = -1
                elif intensity < -self.biasNavi05:
                    self.strategyLock = -2
                else:
                    self.strategyLock = 0
                    self.strategyLock = strategyRecoder
                    return

                if self.strategyLock == -2:
                    print("Turn Left")
                else:
                    print("Turn Right")
                for _ in range(self.minAdjustStepA):  # 运行一组调整动作
                    self.decide()
                    self.myStep()
                self.observation(getScetion=not self.sectionLock)
                
                
        if self.stepCounterPerGroup != 0 and not force:
            return
        else:
            strategyRecoder = self.strategyLock
            actionCounter = 0 # 同
            lastAction = 0
            while self.navigationLevel != -1:
                self.posPredict(color=color, axis=axis)
                if not self.oneshotAdjust and not self.auxAdjustEnable: # 把
                    self.mMotor[18].setPosition(-self.neckLeftForNavigation)
                    for _ in range(18):
                        self.myStep()
                    self.observation(getScetion=not self.sectionLock)
                    tempRho = self.rho
                    tempTheta = self.theta
                    self.posPredict(color=color, axis=axis)
                    self.rho = (self.rho + tempRho) / 2 # 简单平均
                    self.theta = (self.theta + tempTheta) / 2
                print(self.rho, self.theta)

                if self.holdStepWhenAdjust:
                    rho_ = self.rho - biasRho
                    theta_ = self.theta - biasTheta
                    if thetaFirst and theta_ > self.maxBias * 1.4 and (allowRepeat or self.strategyLock != -2):
                        if lastAction == -1:
                            actionCounter += 1
                        else:
                            actionCounter = 0
                        self.strategyLock = -1
                        lastAction = -1
                    elif thetaFirst and theta_ < -self.maxBias * 1.4 and (allowRepeat or self.strategyLock != -1): # 
                        if lastAction == -2:
                            actionCounter += 1
                        else:
                            actionCounter = 0
                        self.strategyLock = -2
                        lastAction = -2
                    elif rho_ > self.maxBias: # 
                        if axis == 1:
                            if allowRepeat or self.strategyLock != -3:
                                if lastAction == -4:
                                    actionCounter += 1
                                else:
                                    actionCounter = 0
                                self.strategyLock = -4
                                lastAction = -4
                        else:
                            if lastAction == -6:
                                actionCounter += 1
                            else:
                                actionCounter = 0
                            self.strategyLock = -6
                            lastAction = -6
                    elif rho_ < -self.maxBias:  # 
                        if axis == 1:
                            if allowRepeat or self.strategyLock != -4:
                                if lastAction == -3:
                                    actionCounter += 1
                                else:
                                    actionCounter = 0
                                self.strategyLock = -3
                                lastAction = -3
                        else:
                            if lastAction == -5:
                                actionCounter += 1
                            else:
                                actionCounter = 0
                            self.strategyLock = -5
                            lastAction = -5
                    elif not thetaFirst and theta_ > self.maxBias * 1.4 and (allowRepeat or self.strategyLock != -2):
                        if lastAction == -1:
                            actionCounter += 1
                        else:
                            actionCounter = 0
                        self.strategyLock = -1
                        lastAction = -1
                    elif not thetaFirst and theta_ < -self.maxBias * 1.4 and (allowRepeat or self.strategyLock != -1): # 
                        if lastAction == -2:
                            actionCounter += 1
                        else:
                            actionCounter = 0
                        self.strategyLock = -2
                        lastAction = -2
                    else:
                        print("Finished Adjust")
                        break
                else:
                    totalBias = (self.rho - self.theta - biasRho + biasTheta) / 1.1
                    if totalBias > self.maxBias and (allowRepeat or self.strategyLock != -1):  # should turn
                        if lastAction == -2:
                            actionCounter += 1
                        else:
                            actionCounter = 0
                        self.strategyLock = -2
                        lastAction = -2
                    elif totalBias < -self.maxBias and (allowRepeat or self.strategyLock != -2):
                        if lastAction == -1:
                            actionCounter += 1
                        else:
                            actionCounter = 0
                        self.strategyLock = -1
                        lastAction = -1
                    else:
                        break

                if actionCounter == 4 and biasTheta == 0 and biasRho == 0:# 
                    break

                if self.strategyLock in [-3, -4]: # 
                    if self.strategyLock == -4:
                        print("Move Left")
                    else:
                        print("Move Right")
                    if lrSingleAdjust:
                        lrSingleAdjust = False
                        nSteps = int(self.usualAdjustStepY * (abs(rho_) / 0.053 or 1))
                        print("nSteps:"+str(nSteps))
                        for _ in range(nSteps):  # 
                            self.decide()
                            self.myStep()
                    else:
                        for _ in range(self.usualAdjustStepY):  # 运
                            self.decide()
                            self.myStep()
                elif self.strategyLock in [-5, -6]:# 前后平移
                    if self.strategyLock == -5:
                        print("Move Backward")
                    else:
                        print("Move Forward")
                    for _ in range(self.usualAdjustStepX): # 运行一组调整动作
                        self.decide()
                        self.myStep()
                elif self.strategyLock in [-1, -2]:
                    if self.strategyLock == -2:
                        print("Turn Left")
                    else:
                        print("Turn Right")
                    for _ in range(self.usualAdjustStepA): # 运行一组调整动作
                        self.decide()
                        self.myStep()

                self.mGaitManager.stop()
                self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                if self.oneshotAdjust:
                    self.mMotor[19].setPosition(self.headUpForNavigation)
                    self.mMotor[18].setPosition(self.neckMidForNavigation)
                else:
                    self.mMotor[19].setPosition(self.headMidForNavigation)
                    self.mMotor[18].setPosition(self.neckLeftForNavigation)
                self.wait(420)  # 等待200ms
                self.observation(getScetion=not self.sectionLock)
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            #self.wait(450)  # 等待200ms
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            self.strategyLock = strategyRecoder
            #self.strategyLock = 0
            self.observation(getScetion=not self.sectionLock)


    def posPredict(self, color="GROUND", axis=1):
        """
        
        :return:
        """
        if color == "DOOR":#
            self.rho = 0
            door = np.array(self.singleColorMask["DOOR"], dtype=np.float) / 255
            self.theta = (door.shape[1] - 2 * np.sum(np.sum(door, axis=0) * np.array(range(door.shape[1]))) / (np.sum(door) + 1e-8)) / door.shape[1]
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))
        if color in ["GROUND", "DOOR"] : # 
            mask = np.array(cv.erode(self.multiColorMask, kernel))
        else:
            mask = np.array(cv.erode(self.singleColorMask[color], kernel))
        # arr = np.zeros_like(mask)
        # arr[4:, 4:-4] = mask[4:, 4:-4]# 用
        arr = mask.copy()
        # arr[:4, :] = 0 # 
        #准
        if axis == 1:
            X, y = np.where(arr == 255)  # 注
        else:
            y, X = np.where(arr == 255)
        
        if y.shape[0] == 0 and color != "DOOR":
            self.rho = 0
            self.theta = 0
            return
      
        model = Model()
        model.fit(X.reshape(-1, 1), y.reshape(-1, ))
        y_pred = model.predict(np.array([arr.shape[1-axis]-1, 0]).reshape(-1, 1))

        if self.auxAdjustEnable2:
            self.theta = (self.auxStdTheta - np.arctan((arr.shape[0] - 1) / (y_pred[1] - y_pred[0]))) / (np.pi / 4)
            self.rho = (arr.shape[0]-1) * y_pred[0] / (y_pred[0] - y_pred[1]) * np.cos(np.arctan((arr.shape[0] - 1) / (y_pred[1] - y_pred[0])))- self.auxStdRho
            if self.theta < -1:
                self.theta %= 1
        else:
            if color == "DOOR":
                self.tempRhoSection4 = (arr.shape[axis] - 2 * y_pred[1 - axis]) / arr.shape[axis]
                self.tempThetaSection4 = 2 * (y_pred[axis] - y_pred[1 - axis]) / arr.shape[axis]
                if abs(self.tempRhoSection4 - self.rho) > 0.4:# 
                    self.rho = self.tempRhoSection4
                    self.theta = self.tempThetaSection4
            else:
                
                self.rho = (arr.shape[axis] - 2 * y_pred[1-axis]) / arr.shape[axis]
                if axis == 0:
                    self.rho = (arr.shape[axis] - y_pred[0] - y_pred[1]) / arr.shape[axis]
                self.theta = 2 * (y_pred[axis] - y_pred[1-axis]) / arr.shape[axis]
        if axis == 1:
            debug = cv.line(self.downSampledFrame, (int(y_pred[1]), 0), (int(y_pred[0]), 29), (0, 0, 255),3)
        else:
            debug = cv.line(self.downSampledFrame, (0, int(y_pred[1])), (39, int(y_pred[0])), (0, 0, 255),3)
        cv.imshow("DEBUG", debug)
        cv.imshow("target", self.singleColorMask["DOOR"])
        cv.imshow("targe2t", self.singleColorMask["RED"])
        cv.imshow("img", mask)
        cv.waitKey(1)

    
    def ifStickLifted(self):
        """
      
        :return: True
        """
        bioIMG_BLACK = cv.inRange(self.downSampledFrameHSV, self.COLOR_RANGE["BLACK"][0], self.COLOR_RANGE["BLACK"][1])
        bioIMG_YELLOW = cv.inRange(self.downSampledFrameHSV, self.COLOR_RANGE["YELLOW"][0], self.COLOR_RANGE["YELLOW"][1])
        bioIMG = cv.bitwise_or(bioIMG_BLACK, bioIMG_YELLOW)
        img = np.array(bioIMG)/255
        roiArea = img[self.roiSction1[0]:self.roiSction1[1], :].sum()
        if img.sum()*self.thresSection0 > roiArea - 10 and img.sum() < 50:
            return True
        return False


    def gyroCheck(self, forceUpdate=0):
        """
        
        :return:
        """
        self.yawIntegrater += self.mGyro.getValues()[2] - 512
        if not self.checkGyro_Enable:
            return
        value = (self.yawIntegrater + self.valuePer2PI / 8) % self.valuePer2PI
       
        if self.falldownMark:
            print("GYRO Adjust（Fall Down）")
            target = sum(self.approxAngleRange[self.approxLocation]) / 2
            # limitationRange = [target - self.valuePer2PI / 36, target + self.valuePer2PI / 36]
            targetRange = [target - self.valuePer2PI / 180, target + self.valuePer2PI / 180]
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            self.checkGyro_Enable = False
            # self.checkFallen_Enable = False
            while True:
                value = (self.yawIntegrater + self.valuePer2PI / 8) % self.valuePer2PI
                if value > targetRange[1]:
                    if value - targetRange[1] > targetRange[0] + self.valuePer2PI - value:
                        self.mGaitManager.setAAmplitude(0.5)  # 左转
                    else:
                        self.mGaitManager.setAAmplitude(-0.5)  # 右
                elif value < targetRange[0]:
                    if targetRange[0] - value < self.valuePer2PI - targetRange[1] + value:
                        self.mGaitManager.setAAmplitude(0.5)  # 左转
                    else:
                        self.mGaitManager.setAAmplitude(-0.5)  # 右
                else:
                    for _ in range(self.usualAdjustStepA):
                        self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                        self.myStep()
                    self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                    self.wait(200)  # 等待200ms
                    self.checkGyro_Enable = True
                    # self.checkFallen_Enable = True
                    # self.mGaitManager.stop()
                    self.falldownMark = False
                    print("GYRO Adjustment Finished(Fall Down)")
                    return
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()

      
        if self.navigationLevel in [1.5, 2]:
            if value < self.approxAngleRange[self.approxLocation][0] - self.valuePer2PI / 24 + 10 or \
               value > self.approxAngleRange[self.approxLocation][1] + self.valuePer2PI / 24:
                print("GYRO Adjust（Froced）")
                target = sum(self.approxAngleRange[self.approxLocation]) / 2
                # limitationRange = [target - self.valuePer2PI / 36, target + self.valuePer2PI / 36]
                targetRange = [target - self.valuePer2PI / 180, target + self.valuePer2PI / 180]
                self.mGaitManager.start()  # 步态生成器进入行走状态
                self.mGaitManager.setYAmplitude(0)
                self.mGaitManager.setXAmplitude(0)
                self.checkGyro_Enable = False
                while True:
                    value = (self.yawIntegrater + self.valuePer2PI / 8) % self.valuePer2PI
                    if value > targetRange[1]:
                        if value - targetRange[1] > targetRange[0] + self.valuePer2PI - value:
                            self.mGaitManager.setAAmplitude(0.5)  # 左转
                        else:
                            self.mGaitManager.setAAmplitude(-0.5)  # 
                    elif value < targetRange[0]:
                        if targetRange[0] - value < self.valuePer2PI - targetRange[1] + value:
                            self.mGaitManager.setAAmplitude(0.5)  # 左转
                        else:
                            self.mGaitManager.setAAmplitude(-0.5)  # 右转
                    else:
                        for _ in range(self.usualAdjustStepA):
                            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                            self.myStep()
                        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                        self.wait(200)  # 等待200ms
                        self.checkGyro_Enable = True
                        self.mGaitManager.stop()
                        print("GYRO Adjustment Finished(Forced)")
                        return
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()

       
        if self.navigationLevel == 2:
            if self.stepCounterPerGroup == 0:
                if abs(self.theta) < 0.04:
                    print("GYRO Self-Cal")
                    self.yawIntegrater = sum(self.approxAngleRange[self.approxLocation]) / 2
                    return

       
        if self.navigationLevel in [-1, 2]:
            target = sum(self.approxAngleRange[self.approxLocation]) / 2
            targetRange = [target - self.valuePer2PI / 15 + 10, target + self.valuePer2PI / 15 - 10]
            if value < targetRange[0] or value > targetRange[1]:
                print("GYRO Adjust(Navigation)")
                self.mGaitManager.start()  # 步态生成器进入行走状态
                self.mGaitManager.setYAmplitude(0)
                self.mGaitManager.setXAmplitude(0)
                self.checkGyro_Enable = False
                while True:
                    value = (self.yawIntegrater + self.valuePer2PI / 8) % self.valuePer2PI
                    if value > targetRange[1]:
                        if value - targetRange[1] > targetRange[0] + self.valuePer2PI - value:
                            self.mGaitManager.setAAmplitude(0.5)  # 左转
                        else:
                            self.mGaitManager.setAAmplitude(-0.5)  # 右转
                    elif value < targetRange[0]:
                        if targetRange[0] - value < self.valuePer2PI - targetRange[1] + value:
                            self.mGaitManager.setAAmplitude(0.5)  # 左转
                        else:
                            self.mGaitManager.setAAmplitude(-0.5)  # 右转
                    else:
                        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                        self.wait(200)  # 等待200ms
                        self.checkGyro_Enable = True
                        self.mGaitManager.stop()
                        print("GYRO Adjustment Finished(Navigation)")
                        return
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()

     
        if forceUpdate == -1:
            self.counterForConfirmLocation = 0
            self.approxLocation += 1
            print("GYRO Angle Range Update(Force)")
        elif forceUpdate != 0:
            self.counterForConfirmLocation = 0
            self.approxLocation = forceUpdate - 1
            print("GYRO Angle Range Update(Force)")
        else:
            if value < self.approxAngleRange[self.approxLocation][0]:
                pass
            elif value < self.approxAngleRange[self.approxLocation][1]:
                self.counterForConfirmLocation = 0
            elif value > self.approxAngleRange[self.approxLocation][1] and value < self.approxAngleRange[self.approxLocation+1][1]:
                self.counterForConfirmLocation += 1
                if self.counterForConfirmLocation > self.numberToConfirmLocation and self.approxLocation != 2:
                    self.approxLocation += 1
                    print("GYRO Angle Range Update")
        return


    def checkIfFallen(self):
        if not self.checkFallen_Enable:
            self.fup = 0
            self.fdown = 0
            return
        acc_tolerance = 60.0
        acc_step = 50  # 计数器上限
        acc = self.mAccelerometer.getValues()  # 通过加速度传感器获取三轴的对应值
        # 向左倒acc[0]变大
        if acc[1] < 512.0 - acc_tolerance:  # 面朝下倒地时y轴的值会变小
            self.fup += 1  # 计数器加1
        else:
            self.fup = 0  # 计数器清零
        if acc[1] > 512.0 + acc_tolerance:  # 背朝下倒地时y轴的值会变大
            self.fdown += 1  # 计数器加 1
        else:
            self.fdown = 0  # 计数器清零
        if abs(acc[0] - 512) > acc_tolerance:
            self.flr += 1
        else:
            self.flr = 0

        if self.fup > acc_step:  # 计数器超过100，即倒地时间超过100个仿真步长
            print("Fall Down")
            self.mMotionManager.playPage(10)  # 执行面朝下倒地起身动作
            self.mMotionManager.playPage(9)  # 恢复准备行走姿势
            self.fup = 0  # 计数器清零
            if self.navigationLevel != 0:
                self.falldownMark = True
        elif self.fdown > acc_step:
            print("Fall Down")
            self.mMotionManager.playPage(11)  # 执行背朝下倒地起身动作
            self.mMotionManager.playPage(9)  # 恢复准备行走姿势
            self.fdown = 0  # 计数器清零
            if self.navigationLevel != 0:
                self.falldownMark = True
        elif self.flr > acc_step:# 
            print("Fall Down(Sideways)")
            self.flr = 0
            self.mMotor[2].setPosition(1.3)
            self.mMotor[3].setPosition(-1.3)
            self.mMotor[4].setPosition(-1.6)
            self.mMotor[5].setPosition(1.6)
            self.checkFallen_Enable = False
            self.checkGyro_Enable = False
            for i in range(12):
                self.myStep()
            sum = 0
            for _ in range(10):
                acc = self.mAccelerometer.getValues()
                sum += acc[1]
                self.myStep()
            self.checkFallen_Enable = True
            self.checkGyro_Enable = True
            if sum > (512 + acc_tolerance) * 10:
                self.mMotionManager.playPage(11)  # 
                self.mMotionManager.playPage(9)  # 
                if self.navigationLevel != 0:
                    self.falldownMark = True
            elif sum < (512 - acc_tolerance) * 10:
                self.mMotionManager.playPage(10)  # 
                self.mMotionManager.playPage(9)  # 
                if self.navigationLevel != 0:
                    self.falldownMark = True


    def getSection(self):
        if self.strategyLock != 0 or self.sectionLock:
            return

        if (not self.sectionFlag[1] or not self.sectionFlag[5]) and self.singleColorMask["GREEN"].sum() > 20 * 255:
            contours, _ = cv.findContours(self.singleColorMask["GREEN"], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            hullPoints = cv.convexHull(contours[0])
            if not self.sectionFlag[1] and cv.contourArea(hullPoints) - cv.contourArea(contours[0]) > self.thresEnterSection1:
                print("Enter Section")
                self.sectionFlag[1] = True
                self.nextSction = 2
                return
            elif not self.sectionFlag[5]:
                green = np.array(self.singleColorMask["GREEN"], dtype=np.float)
                green /= 255
                roiArea = green[:, self.roiSction5[0]:self.roiSction5[1]].sum()
                if green.sum() * self.thresSection5 < roiArea - 10 and roiArea > self.thresEnterSection5:
                    self.sectionFlag[5] = True
                    print("Enter Section")
                    self.nextSction = 6
                    return

        if not self.sectionFlag[2]:# and self.singleColorMask["GRAY"].sum() > (30 * 40 * 255) / 2: # 
            contours, _ = cv.findContours(self.singleColorMask["BLACK"], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            sum = 0
            for contour in contours:
                area = cv.contourArea(contour)
                if area < 38:
                    sum += 1
            if sum > 2:
                self.sectionFlag[2] = True
                self.nextSction = 3
                return

        if not self.sectionFlag[3] or not self.sectionFlag[6] or not self.sectionFlag[7] :
            blue = np.array(self.singleColorMask["BLUE"], dtype=np.float)
            blue /= 255
            weightY = int(np.sum(np.sum(blue, axis=1)*np.array(range(blue.shape[0])))/(np.sum(blue)+1e-8))
            weightX = int(np.sum(np.sum(blue, axis=0) * np.array(range(blue.shape[1]))) / (np.sum(blue) + 1e-8))
            if weightY > 7: # 
                roiArea3 = blue[weightY - 3:weightY + 4, :].sum()
                roiArea7 = blue[weightY - 6:weightY + 7, :].sum()
                if blue.sum() * self.thresSection3 < roiArea3 - 5 and roiArea3 > self.thresEnterSection3:
                    print("Enter Section")
                    self.sectionFlag[3] = True
                    self.nextSction = 4
                    return
                elif blue.sum() * self.thresSection7 < roiArea7 - 10 and roiArea7 > self.thresEnterSection7:
                    print("EnterSection")
                    self.sectionFlag[7] = True
                    self.nextSction = 8
                    return
            if weightY > blue.shape[0] * 0.65 and weightX > blue.shape[1] * 0.6:
                roiArea6 = blue[weightY - 3:weightY + 4, weightX - 3:weightX + 4].sum()
                if blue.sum() * 0.8 < roiArea6 and roiArea6 > 15 and roiArea6 < 45:
                    print("Enter Ball Section")
                    self.sectionFlag[6] = True
                    self.nextSction = 7
                    return
                    
        if not self.sectionFlag[4]:
            door = np.array(self.singleColorMask["DOOR"], dtype=np.float)
            door /= 255
            doorl = door[:, :door.shape[1] // 2]
            doorr = door[:, door.shape[1] // 2:]
            weightXl = int(np.sum(np.sum(doorl, axis=0) * np.array(range(doorl.shape[1]))) / (np.sum(doorl) + 1e-8))
            weightXr = int(np.sum(np.sum(doorr, axis=0) * np.array(range(doorr.shape[1]))) / (np.sum(doorr) + 1e-8))
            roiAreal = doorl[:, weightXl - 4:weightXl + 5].sum()
            roiArear = doorr[:, weightXr - 4:weightXr + 5].sum()
            # print(roiAreal, roiArear, doorr.sum())
            if doorl.sum() * self.thresSection4 < roiAreal and doorr.sum() * self.thresSection4 < roiArear and roiAreal > self.thresEnterSection4 and roiArear > self.thresEnterSection4:
                print("EnterSection")
                self.sectionFlag[4] = True
                self.nextSction = 5
                return



    def observation(self, getScetion=True):
        self.frame = np.frombuffer(self.camera.getImage(), np.uint8).reshape((120, 160, 4))[:, :, 0:3]
        self.downSampledFrame = cv.pyrDown(cv.pyrDown(self.frame))
        self.frameHSV = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        self.downSampledFrameHSV = cv.cvtColor(self.downSampledFrame, cv.COLOR_BGR2HSV)
        if True: # self.strategyLock < 1:
            for color in self.singleColorMask.keys():
                self.singleColorMask[color] = cv.inRange(self.downSampledFrameHSV, self.COLOR_RANGE[color][0], self.COLOR_RANGE[color][1])
            contours, _ = cv.findContours(self.singleColorMask["GREEN"], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                contour = []
                for i in range(len(contours)):
                    hullPoints = cv.convexHull(contours[i])
                    contour.append(hullPoints)
                cv.drawContours(self.singleColorMask["auxGREEN"], contour, -1, 255,
                                                            thickness=-1)
            self.multiColorMask = self.singleColorMask["BLUE"]
            for color in self.groundColor:
                self.multiColorMask = cv.bitwise_or(self.multiColorMask, self.singleColorMask[color])
        if getScetion and self.nextSction == -10:
            self.getSection()


    def decide(self):
        if self.strategyLock == 0:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1.0)  # 设置前进幅度
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setAAmplitude(0)
            self.mMotor[19].setPosition(0.36)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == -1:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setAAmplitude(-0.3)
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == -2:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setAAmplitude(0.3)
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == -3:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setYAmplitude(-0.98)
            self.mGaitManager.setXAmplitude(0)
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == -4:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setYAmplitude(0.98)
            self.mGaitManager.setXAmplitude(0)
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == -5:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(-0.6)  # 设置前进幅度
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == -6:
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(0.6)  # 设置前进幅度
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作

        elif self.strategyLock == 1:
            if not self.needtoConfirmStickState:
                self.mGaitManager.start()  # 步态生成器进入行走状态
                self.mGaitManager.setXAmplitude(1.0)  # 设置前进幅度
                self.mGaitManager.setYAmplitude(0)
                self.mGaitManager.setAAmplitude(0)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.sectionFlag[0] = True
                return
            if not self.sectionFlag[0]:
                self.checkRelax_Enable = False
                self.sectionFlag[0] = True
                self.mGaitManager.start()  # 步态生成器进入行走状态
                self.mGaitManager.setXAmplitude(1.0)  # 设置前进幅度
                self.mGaitManager.setYAmplitude(0)
                self.mGaitManager.setAAmplitude(0)
            if self.initStepSection0 != 0:
                self.initStepSection0 -= 1
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            else:
                self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                self.wait(200)  # 等待200ms
                confirmStickState = False
                while True:
                    res = self.ifStickLifted()
                    if res:
                        if not confirmStickState:
                            self.myStep()
                            self.observation(getScetion=not self.sectionLock)
                            continue
                        self.mGaitManager.start()  # 步态生成器进入行走状态
                        self.mGaitManager.setXAmplitude(1.0)  # 设置前进幅度
                        self.mGaitManager.setYAmplitude(0)
                        self.mGaitManager.setAAmplitude(0)
                        for _ in range(105):
                            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                            self.myStep()
                        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                        self.wait(200)  # 等待200ms
                        self.strategyLock = 0
                        self.checkRelax_Enable = True
                        break
                    else:
                        confirmStickState = True
                        self.myStep()
                        self.observation(getScetion=not self.sectionLock)
                print("Exiction")

        elif self.strategyLock == 2:
            self.sectionFlag[1] = True
            # if self.navigationLevel == 0.5:
                # self.strategyLock = 0
                # print("Exion")
                # return
            self.sectionLock = True
            # self.holdStepWhenAdjust = True
            self.checkGyro_Enable = False
            #stepRecorder = self.maxStepPerGroup
            angleRecoder = self.headMidForNavigation
            #self.maxStepPerGroup = self.midStepPerGroup
            self.headMidForNavigation = 0.46
            self.strategyLock = 0
            self.maxBias = 0.06
            #self.headMidForNavigation = -0.1
            self.mMotor[19].setPosition(self.headMidForNavigation)
            self.mMotor[18].setPosition(self.neckLeftForNavigation)
            for _ in range(12):
                self.myStep()
            self.observation(getScetion=not self.sectionLock)
            print("move left 525")
            self.strategyLock = -4
            for _ in range(450):
                self.decide()
                self.myStep()
            self.strategyLock = -1
            for _ in range(24):
                self.decide()
                self.myStep()
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            self.strategyLock = 0
            # self.adjust(biasRho=self.biasRhoSection1, biasTheta=self.biasThetaSection1, thetaFirst=False, lrSingleAdjust=True, allowRepeat=True, force=True)  # 
            self.headMidForNavigation = -0.1
            self.auxAdjustEnable = True
            self.mMotionManager.playPage(9)
            self.mMotor[19].setPosition(self.headMidForNavigation)
            self.mMotor[18].setPosition(self.neckLeftForNavigation)
            for _ in range(30):
                self.myStep()
            # 
            self.stepCounterPerGroup = 0
            while np.sum(self.singleColorMask["GREEN"]) / 255 > self.thresExitSection1:
                self.observation(getScetion=not self.sectionLock)
                self.adjust(biasRho=self.biasRhoSection1_, biasTheta=self.biasThetaSection1_, allowRepeat=True)
                self.decide()
                self.myStep()  # 仿真一个步长
                self.relax()
            self.maxBias = 0.1 # 
            #self.maxStepPerGroup = stepRecorder
            self.headMidForNavigation = angleRecoder
            self.holdStepWhenAdjust = False
            self.sectionLock = False
            self.auxAdjustEnable = False
            # set robot rho=0
            self.stepCounterPerGroup = 0
            for _ in range(2 * self.maxStepPerGroup):
                self.mGaitManager.setAAmplitude(-0.12)
                self.mGaitManager.setYAmplitude(0)
                self.mGaitManager.setXAmplitude(1)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
                self.relax()
            self.strategyLock = -1
            for _ in range(50):
                self.decide()
                self.myStep()
            self.strategyLock = 0
            for _ in range(2*self.maxStepPerGroup):
                self.decide()
                self.myStep()
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            self.strategyLock = 0
            self.checkGyro_Enable = True
            self.navigationLevel = 0.5
            print("Exiction")

        elif self.strategyLock == 3:# 
            self.sectionFlag[2] = True
            self.strategyLock = 0
            if self.navigationLevel == 0.5:
                navigationFlag = True
                self.navigationLevel = 1
            else:
                navigationFlag = False
            self.sectionLock = True
            self.checkGyro_Enable = False
            self.relax(force=True)
            self.observation(getScetion=False)
            self.holdStepWhenAdjust = True
            # self.maxBias = 0.06
            self.adjust(force=True, allowRepeat=True)
            if navigationFlag:
                self.navigationLevel = 0.5
            self.holdStepWhenAdjust = False
            self.maxBias = 0.1
            #self.mMotionManager.playPage(9)
            self.mMotor[18].setPosition(self.neckMidForNavigation)
            self.wait(300)
            # TODO
            self.crossMine()
            # TODO
            self.sectionLock = False
            self.checkGyro_Enable = True
            print("Exitection")

        elif self.strategyLock == 4: # 
            self.sectionFlag[3] = True
            self.sectionLock = True
            self.holdStepWhenAdjust = True
            # self.adjust(force=True)
            # 向前直行一组直行动作
            self.strategyLock = 0
            for _ in range(self.maxStepPerGroup):
                self.decide()
                self.myStep()
            self.oneshotAdjust = True
            headAngleRecoder = self.headUpForNavigation
            adjustStepRecoder = self.usualAdjustStepA
            self.headUpForNavigation = -0.1
            self.usualAdjustStepA = self.minAdjustStepA
            self.relax(headLeft=False, force=True)
            # self.mMotor[19].setPosition(self.headDownForNavigation)
            # for _ in range(5): # 
                # self.myStep()
           
            self.horizontalRho = self.rhoYSection3
            self.relax(force=True, headLeft=False)
            self.adjust(biasRho=self.horizontalRho, color="BLUE", axis=0, allowRepeat=True, force=True)
            self.checkFallen_Enable = False
            self.flipOverThreshold() 
            self.headUpForNavigation = headAngleRecoder
            self.usualAdjustStepA = adjustStepRecoder
            self.checkFallen_Enable = True
            self.holdStepWhenAdjust = False
            self.oneshotAdjust = False
            self.wait(200)
            if self.thresAtCorner:
                self.rotate(90)
            else:
                self.strategyLock = -5
                for _ in range(2*self.maxStepPerGroup):
                    self.decide()
                    self.myStep()  # 仿真一个步长
                self.strategyLock = 0
                self.rotate(28)
            self.sectionLock = False
            self.falldownMark = False
            self.relax(force=True)
            self.observation(getScetion=False)
            self.adjust(force=True)
            print("Exition")

        elif self.strategyLock == 5:# 
            self.sectionFlag[4] = True
            if self.navigationLevel != 0.5:
                self.sectionLock = True
            paramRecorder = self.maxStepPerGroup
            self.maxStepPerGroup = self.midStepPerGroup
            self.strategyLock = 0
            while np.sum(self.singleColorMask["DOOR"]) / 255 > self.thresExitSection4:
                self.observation(getScetion=not self.sectionLock)
                if self.supposeInCenter:
                    self.adjust()
                else:
                    self.adjust(color="DOOR")
                self.decide()
                self.myStep()
                self.relax()
            if not self.supposeInCenter: # 看后，保持最后一次的赛道参数继续走
                for _ in range(self.stepsForConfirmSection4):
                    self.observation(getScetion=not self.sectionLock)
                    self.adjust(biasRho=self.tempRhoSection4, biasTheta=self.tempThetaSection4)
                    self.decide()
                    self.myStep()
                    self.relax()
            self.maxStepPerGroup = paramRecorder
            self.strategyLock = 0
            self.sectionLock = False
            print("Exit Door Section")

        elif self.strategyLock == 6:# 
            print("strategy 6")
            self.sectionFlag[5] = True
            # paramRecorder = self.maxStepPerGroup
            # self.maxStepPerGroup = self.midStepPerGroup
            self.strategyLock = 0
            recorder = self.headMidForNavigation
            if not self.treatBridgeAsGround:
                self.sectionLock = True
                self.headMidForNavigation = self.headDownForNavigation
                self.mMotor[19].setPosition(self.headMidForNavigation)
                self.mMotor[18].setPosition(self.neckLeftForNavigation)
                for _ in range(12):
                    self.myStep()
                self.wait(100)
                while np.sum(self.singleColorMask["GREEN"]) / 255 > self.thresExitSection5:
                    self.observation(getScetion=not self.sectionLock)
                    self.adjust(color="GREEN")
                    self.decide()
                    self.myStep()
                    self.relax()
                self.sectionLock = False
                self.headMidForNavigation = recorder
                # self.maxStepPerGroup = paramRecorder
                    # self.holdStepWhenAdjust = False
                self.strategyLock = 0
                self.mMotionManager.playPage(9)
                self.wait(100)
                self.adjust(force=True)
            self.sectionLock = False
            print("Extion")

        elif self.strategyLock == 7:  # 踢球
            self.sectionFlag[6] = True
            self.sectionLock = True
            print("why not take a try XD, good luck!")
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1.0)  # 设置前进
            self.mGaitManager.setAAmplitude(-0.17)
            for _ in range(int(self.maxStepPerGroup * 2)):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.setXAmplitude(0.8)  # 设置前进
            self.mGaitManager.setAAmplitude(0.8)
            for _ in range(89):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.setXAmplitude(0)  # 设置前进
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.stop()
            self.mMotionManager.playPage(9)
            self.wait(400)
            self.sectionLock = False
            self.strategyLock = 0
            print("Exit Soccer Section")


        elif self.strategyLock == 8:# 
            
            self.sectionFlag[7] = True
            self.sectionLock = True
            self.checkGyro_Enable = False
            self.checkFallen_Enable = False
            self.mMotor[18].setPosition(self.neckMidForNavigation)
            #################
            #
            # self.holdStepWhenAdjust = True
            # self.adjust(force=True)
            # # 向
            # self.strategyLock = 0
            # for _ in range(self.midStepPerGroup):
            #     self.decide()
            #     self.myStep()
            # self.oneshotAdjust = True
            # self.relax(headLeft=False, force=True)
            # self.mMotor[19].setPosition(self.headDownForNavigation)
            # for _ in range(5):  # 
            #     self.myStep()
            #
            # self.horizontalRho = self.rhoYSection7
            # self.adjust(biasRho=self.horizontalRho, color="BLUE", axis=0, force=True)
            ##################
          
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1)  # 设置前进
            for _ in range(195):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(0.4)  # 设置前进
            for _ in range(130):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(-0.09)  # 设置前进
            for _ in range(80):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.stop()
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            self.wait(500)
            # 
            # 1
            self.stepUp()
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1.0)  # 设置前进
            self.mGaitManager.setAAmplitude(-0.07)  # 设置前进
            for _ in range(140):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(0.4)  # 设置前进
            for _ in range(130):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(-0.09)  # 设置前进
            for _ in range(80):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.stop()
            self.wait(100)
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            self.wait(600)  # 等待200ms

            # 2
            self.stepUp()
            self.mGaitManager.stop()
            self.mGaitManager.start()
            self.mGaitManager.setXAmplitude(1.0)
            self.mGaitManager.setAAmplitude(-0.01)  # 设置前进
            for _ in range(140):
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            self.mGaitManager.start()
            self.mGaitManager.setXAmplitude(0.4)
            for _ in range(130):
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            self.mGaitManager.start()
            self.mGaitManager.setXAmplitude(-0.09)
            for _ in range(80):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()
            self.mGaitManager.stop()
            self.wait(100)
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            self.wait(900)

            # 3
            self.stepUp()
            self.wait(200)
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1)
            for _ in range(105):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            self.wait(400)

          
            self.checkFallen_Enable = True
            self.holdStepWhenAdjust = True
            self.oneshotAdjust = True
            headAngleRecoder = self.headUpForNavigation
            adjustStepRecoder = self.usualAdjustStepA
            self.headUpForNavigation = -0.1
            self.usualAdjustStepA = self.minAdjustStepA
            self.relax(headLeft=False, force=True)
            self.observation()
            self.horizontalRho = self.rhoYSection71
            self.adjust(biasRho=self.horizontalRho, color="GREEN", axis=0, allowRepeat=True, force=True)

            # 4
            self.stepDown()
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1.0)  #
            self.mGaitManager.setAAmplitude(0.07)
            for _ in range(105):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.stop()
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            self.wait(320)  # 等待200ms

            # 5
            self.stepDown()
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1)
            for _ in range(105):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
            self.mGaitManager.stop()
            self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
            self.wait(650)  # 等待200ms

           
            self.headUpForNavigation = -0.3
            self.relax(headLeft=False, force=True)
            self.observation()
            self.horizontalRho = self.rhoYSection72
            self.adjust(biasRho=self.horizontalRho, biasTheta=0.13, color="RED", axis=0, allowRepeat=True, force=True)
            self.headUpForNavigation = headAngleRecoder
            self.usualAdjustStepA = adjustStepRecoder
            # self.mGaitManager.start()  # 步态生成器进入行走状态
            # self.mGaitManager.setAAmplitude(0.3)
            self.mGaitManager.setAAmplitude(0)
            self.mGaitManager.setYAmplitude(0)
            self.mGaitManager.setXAmplitude(0)
            # for _ in range(10):
                # self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                # self.myStep()
            self.mMotionManager.playPage(9)
            self.wait(600)
            # 6
            self.checkFallen_Enable = False
            self.slopeDown()
            self.checkFallen_Enable = True
            self.mGaitManager.stop()
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(0.6)
            for _ in range(51):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()
            # self.mGaitManager.start()
            # self.mGaitManager.setAAmplitude(-0.5)
            # for _ in range(57):
                # self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                # self.myStep()
            # self.mMotionManager.playPage(9)
            # self.wait(300)  # 等待200ms
            self.mGaitManager.start()  # 步态生成器进入行走状态
            self.mGaitManager.setXAmplitude(1)
            for _ in range(89):
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()

            self.holdStepWhenAdjust = False
            self.oneshotAdjust = False
            self.sectionLock = False
            self.checkGyro_Enable = True
            self.checkFallen_Enable = True
            self.strategyLock = 0
            self.adjust(force=True)
            print("Exit Stairs Section")
            
            
            
    def detectMines(self):
        self.observation()
        gray = cv.cvtColor(self.frame.copy(), cv.COLOR_BGR2GRAY)
        mines = self.mine_cascade.detectMultiScale(gray, 1.1, 14, cv.CASCADE_SCALE_IMAGE, (10, 10), (40, 40))
        if len(mines) != 0:
            # print("detected " + str(len(mines)) + " mines at " + str(mines))
            pass
        else:
            return 1
        for (x, y, w, h) in mines:
            if x + w > 0.28 * self.width and x < 0.68 * self.width and y + h > 0.70 * self.height:
                if x > 0.5 * self.width:
                    print("Turn left")
                    return 2
                else:
                    print("Turn right")
                    return 3
        return 0


    def crossMine(self):
        zero_count = 0
        cnt = 0
        
        while True:
            cnt += 1
            if cnt >= 80:
                cnt = 0
                event = self.detectMines()
                if event == 1:
                    zero_count += 1
                    if zero_count > 30:
                        return
                elif event >= 2:
                    targetDegree = 28
                    if event == 2:  # turnleft
                        targetDegree = -targetDegree
                    self.turnDegrees(targetDegree)
                    while self.finished == False:
                        self.wait(200)
                    self.finished = False
                    iter_nums = 400

                    while iter_nums > 0:
                        self.mGaitManager.setXAmplitude(0.8)
                        self.mGaitManager.step(int(self.mTimeStep2))  # 步态生成器生成一个步长的动作
                        self.myStep2()
                        iter_nums -= 1
                    self.finished = True
                    while self.finished == False:
                        self.wait(200)
                    self.turnDegrees(-targetDegree * 0.65)
                    while self.finished == False:
                        self.wait(200)
            # self.adjust(force=True)
            self.mGaitManager.start()
            self.mGaitManager.setXAmplitude(0.8)
            self.mGaitManager.step(int(self.mTimeStep2))  # 步态生成器生成一个步长的动作
            self.myStep2()
        
        
        
    def step(self):  # 实现机器人自动前进
    ###########################################################################
        self.myStep()  # 仿真一个步长，刷新传感器读数  
        cv.startWindowThread()
        cv.namedWindow("img")
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走  
        self.wait(1000)  # 等待200ms  
        self.mGaitManager.setBalanceEnable(True)
        self.mMotor[19].setPosition(0.1)
        
        #print(self.positionSensors[19].getValue())
        #self.isWalking = False  # 初始时机器人未进入行走状态  
        #self.maxStepPerGroup = 9
        while True:
           
            self.observation()
            self.adjust()
            self.decide()
            self.myStep()  # 仿真一个步长
            self.relax()
        self.wait(10000)
        #########################  test here  ##################################
        # cv.startWindowThread()
        #cv.namedWindow("img")
        # sum = 0
        # self.mGaitManager.start()  # 步态生成器进入行走状态  
        # self.mGaitManager.setXAmplitude(1) 
        
        # while True:
        # for _ in range(86):
            # self.observation()
            # # self.adjust()
            # self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            # self.myStep()  # 仿真一个步长
            # gyro3d = self.mGyro.getValues()
            # sum += gyro3d[2] - 512
            # print(sum)
        # self.wait(2000)
        # self.mMotor[2].setPosition(1.3)
        # self.mMotor[3].setPosition(-1.3)
        # self.mMotor[4].setPosition(-1.6)
        # self.mMotor[5].setPosition(1.6)
        # for i in range(55):
            # self.myStep()
        self.checkFallen_Enable = False
        self.mMotor[19].setPosition(0.94)
        self.mMotor[10].setPosition(-0.5)
        self.mMotor[11].setPosition(0.5)
        self.mMotor[12].setPosition(0.4)
        self.mMotor[13].setPosition(-0.4)
        self.mMotor[14].setPosition(0.2)
        self.mMotor[15].setPosition(-0.2)
        for i in range(25):
            self.myStep()
        self.mMotor[10].setPosition(-0.5)
        self.mMotor[11].setPosition(0.5)
        self.mMotor[12].setPosition(0.)
        self.mMotor[13].setPosition(-0.)
        self.mMotor[14].setPosition(0.)
        self.mMotor[15].setPosition(-0.)
        for i in range(25):
            self.myStep()
        #self.wait(5000)
        print(111)
        self.mMotor[5].setPosition(1.5)
        self.mMotor[4].setPosition(-1.5)
        self.mMotor[0].setPosition(-1.1)
        self.mMotor[1].setPosition(1.1)
        self.mMotor[2].setPosition(2.3)
        self.mMotor[3].setPosition(-2.25)
        for i in range(12):
            self.myStep()
        print(222)
        #self.wait(5000)
        self.mMotor[10].setPosition(-1.2)
        self.mMotor[11].setPosition(1.2)
        self.mMotor[12].setPosition(0.)
        self.mMotor[13].setPosition(-0.)
        self.mMotor[14].setPosition(0.36)
        self.mMotor[15].setPosition(-0.36)
        self.mMotor[0].setPosition(-1.55)
        self.mMotor[1].setPosition(1.55)
        for i in range(7):
            self.myStep()
        self.mMotor[14].setPosition(-0.22)
        self.mMotor[15].setPosition(0.22)
        for i in range(5):
            self.myStep()
        print(333)
        self.mMotor[10].setPosition(0.1)
        self.mMotor[11].setPosition(-0.1)
        self.mMotor[12].setPosition(2.25)
        self.mMotor[13].setPosition(-2.25)
        for i in range(15):
            self.myStep()
        print(444)
        #self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走  
        self.wait(1000)  # 等待200ms 
        ##############################################################
        print(self.positionSensors[2].getValue(),self.positionSensors[15].getValue())
        self.mMotor[10].setPosition(-0.5)
        self.mMotor[11].setPosition(0.5)
        self.mMotor[12].setPosition(0.4)
        self.mMotor[13].setPosition(-0.4)
        self.mMotor[14].setPosition(0.2)
        self.mMotor[15].setPosition(-0.2)
        for i in range(30):
            self.myStep()
        self.mMotor[10].setPosition(-0.5)
        self.mMotor[11].setPosition(0.5)
        self.mMotor[12].setPosition(0.)
        self.mMotor[13].setPosition(-0.)
        self.mMotor[14].setPosition(0.)
        self.mMotor[15].setPosition(-0.)
        for i in range(30):
            self.myStep()
        self.mMotor[10].setPosition(-0.)
        self.mMotor[11].setPosition(0.)
        self.mMotor[5].setPosition(1.5)
        self.mMotor[4].setPosition(-1.5)
        self.mMotor[3].setPosition(-0.9)
        self.mMotor[17].setPosition(0.2)
        self.mMotor[16].setPosition(0.2)
        for i in range(20):
            self.myStep()
        self.squatDown(0.3, -2)
        self.squatDown(0., -2)
        
        self.mMotor[12].setPosition(2.24)
        for i in range(12):
            self.myStep()
        self.mMotor[10].setPosition(-1.76)
        for i in range(12):
            self.myStep()
        self.mMotor[12].setPosition(0.8)
        self.mMotor[11].setPosition(-0.3)
        self.mMotor[16].setPosition(-0.12)
        self.mMotor[0].setPosition(-1.8)
        for i in range(20):
            self.myStep()
        
        self.mMotor[11].setPosition(-0.5)
        self.mMotor[14].setPosition(-0.6)
        self.mMotor[15].setPosition(-0.19)
        for i in range(55):
            self.myStep()
        
        #self.mMotor[10].setPosition(-1.2)
        #self.mMotor[12].setPosition(1.1)
        self.mMotor[3].setPosition(-0.5)
        self.mMotor[0].setPosition(-1.5)
        for i in range(20):
            self.myStep()
        # TODO
        self.mMotor[15].setPosition(-0.5)
        self.mMotor[14].setPosition(0.19)
        self.mMotor[11].setPosition(-0.4)
        self.mMotor[17].setPosition(0.02)
        self.mMotor[0].setPosition(0.8)
        self.mMotor[2].setPosition(-0.2)
        self.mMotor[8].setPosition(0.19)
        self.mMotor[1].setPosition(-0.6)
        for i in range(2):
            self.myStep()
        self.mMotor[15].setPosition(0.8)
        for i in range(3):
            self.myStep()
        self.mMotor[11].setPosition(-0.5)
        self.mMotor[12].setPosition(0.6)
        self.mMotor[10].setPosition(-0.84)
        for i in range(10):
            self.myStep()
        self.mMotor[13].setPosition(-1.25)
        for i in range(8):
            self.myStep()
            
        # right foot on ground
        self.mMotor[16].setPosition(-0.2)
        self.mMotor[0].setPosition(0.6)
        self.mMotor[2].setPosition(0.1)
        #self.mMotor[3].setPosition(0.3)
        self.mMotor[1].setPosition(-2.1)
        self.mMotor[10].setPosition(-0.92)
        for i in range(16):
            self.myStep()
            
        # self.mMotor[10].setPosition(-1.5)
        # #self.mMotor[11].setPosition(0.15)
        # for i in range(12):
            # self.myStep()
            
        # step over
        self.mMotor[8].setPosition(0.36)
        for i in range(16):
            self.myStep()
        self.mMotor[9].setPosition(-0.99)
        for i in range(12):
            self.myStep()
        self.mMotor[13].setPosition(-2.24)
        self.mMotor[11].setPosition(0.49)
        self.mMotor[10].setPosition(-1.)
        for i in range(16):
            self.myStep()
        self.squatDown(-2, 0.2)
        self.mMotor[13].setPosition(-0.0)
        self.mMotor[11].setPosition(0.4)
        self.mMotor[15].setPosition(0.0)
        for i in range(16):
            self.myStep()
        
        self.mMotor[0].setPosition(-0.845)
        self.mMotor[2].setPosition(-0.31)
        self.mMotor[4].setPosition(0.51)
        self.mMotor[1].setPosition(0.72)
        self.mMotor[3].setPosition(0.3)
        self.mMotor[5].setPosition(-0.52)
        for i in range(12):
            self.myStep()
            
        self.mMotor[8].setPosition(0.)
        self.mMotor[12].setPosition(0.)
        self.mMotor[15].setPosition(0.)
        self.mMotor[14].setPosition(0.)
        self.mMotor[16].setPosition(0.)
        self.mMotor[17].setPosition(0.)
        self.mMotor[10].setPosition(0.)
        self.mMotor[11].setPosition(0.)
        self.mMotor[9].setPosition(0.)
        for i in range(16):
            self.myStep()
            
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走  
        print("next")
        self.wait(5000)

    
    def slopeDown(self):
        # #
        # self.mMotor[12].setPosition(1.0)  # 右膝
        # self.mMotor[17].setPosition(0.2)  # 左足
        # self.mMotor[16].setPosition(0.5)  # 右足
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[12].setPosition(2.0)  # 右膝
        # self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(-0.)  # 右膝
        # # self.mMotor[12].setPosition(-0.02)#右膝
        # self.mMotor[14].setPosition(0.)  # 右脚踝
        # self.mMotor[14].setPosition(-0.22)  # 右脚踝
        # self.mMotor[16].setPosition(0.)  # 右足
        # self.mMotor[16].setPosition(-0.25)  # 右足
        # for _ in range(5):
        #     self.myStep()
        # self.mMotor[13].setPosition(0.)  # 左膝
        # self.mMotor[13].setPosition(-1.25)  # 左膝
        # self.mMotor[15].setPosition(-0.)  # 左脚踝
        # self.mMotor[15].setPosition(-0.85)  # 左脚踝
        # self.mMotor[1].setPosition(-0.3)  # 左肩
        # self.mMotor[3].setPosition(-1.0)  # 左臂
        # self.mMotor[5].setPosition(1.5)  # 左肘
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[17].setPosition(-0.)  # 左足
        # for _ in range(70):
        #     self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        # for _ in range(20):
        #     self.myStep()

        # 
        # for _ in range(20):
        #     self.myStep()
        self.mMotor[12].setPosition(0.97)  # 右膝
        self.mMotor[13].setPosition(-0.97)  # 左膝
        for _ in range(20):
            self.myStep()
        self.mMotor[12].setPosition(1.0)  # 右膝
        self.mMotor[17].setPosition(0.2)  # 左足
        self.mMotor[16].setPosition(0.5)  # 右足
        for _ in range(20):
            self.myStep()
        self.mMotor[12].setPosition(1.4)  # 右膝
        self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        for _ in range(10):
            self.myStep()
        self.mMotor[12].setPosition(-0.)  # 右膝
        self.mMotor[12].setPosition(0.16)  # 右膝
        self.mMotor[14].setPosition(0.)  # 右脚踝
        self.mMotor[14].setPosition(-0.14)  # 右脚踝
        self.mMotor[16].setPosition(0.)  # 右足
        self.mMotor[16].setPosition(-0.17)  # 右足
        for _ in range(15):
            self.myStep()
        self.mMotor[13].setPosition(0.)  # 左膝
        self.mMotor[13].setPosition(-1.21)  # 左膝
        self.mMotor[15].setPosition(-0.)  # 左脚踝
        self.mMotor[15].setPosition(-0.85)  # 左脚踝
        # self.mMotor[1].setPosition(-0.6)  # 左肩
        self.mMotor[3].setPosition(0.5)  # 左臂
        self.mMotor[2].setPosition(1.0)  # you臂
        self.mMotor[4].setPosition(-1.5)  # you肘
        self.mMotor[5].setPosition(1.5)  # 左肘

        for _ in range(10):
            self.myStep()
        self.mMotor[17].setPosition(-0.)  # 左足

        for _ in range(55):
            self.myStep()

        self.mMotionManager.playPage(2)  # 执行动作组9号动作，初始化站立姿势
        for _ in range(20):
            self.myStep()
        self.mMotionManager.playPage(9)
        self.wait(200)
    
    
    def flipOverThreshold(self):
        self.mMotor[19].setPosition(0.94)
        self.mMotor[10].setPosition(-0.5)
        self.mMotor[11].setPosition(0.5)
        self.mMotor[12].setPosition(0.4)
        self.mMotor[13].setPosition(-0.4)
        self.mMotor[14].setPosition(0.2)
        self.mMotor[15].setPosition(-0.2)
        for i in range(25):
            self.myStep()
        self.mMotor[10].setPosition(-0.5)
        self.mMotor[11].setPosition(0.5)
        self.mMotor[12].setPosition(0.)
        self.mMotor[13].setPosition(-0.)
        self.mMotor[14].setPosition(0.)
        self.mMotor[15].setPosition(-0.)
        for i in range(25):
            self.myStep()
        self.mMotor[5].setPosition(1.5)
        self.mMotor[4].setPosition(-1.5)
        self.mMotor[0].setPosition(-1.1)
        self.mMotor[1].setPosition(1.1)
        self.mMotor[2].setPosition(2.3)
        self.mMotor[3].setPosition(-2.25)
        for i in range(12):
            self.myStep()
        self.mMotor[10].setPosition(-1.2)
        self.mMotor[11].setPosition(1.2)
        self.mMotor[12].setPosition(0.)
        self.mMotor[13].setPosition(-0.)
        self.mMotor[14].setPosition(0.36)
        self.mMotor[15].setPosition(-0.36)
        self.mMotor[0].setPosition(-1.55)
        self.mMotor[1].setPosition(1.55)
        for i in range(7):
            self.myStep()
        self.mMotor[14].setPosition(-0.22)
        self.mMotor[15].setPosition(0.22)
        for i in range(5):
            self.myStep()
        self.mMotor[10].setPosition(0.1)
        self.mMotor[11].setPosition(-0.1)
        self.mMotor[12].setPosition(2.25)
        self.mMotor[13].setPosition(-2.25)
        for i in range(15):
            self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        self.wait(300)  # 等待200ms
        
            
    def turnDegrees(self, degree=360.):
        self.finished = False
        print("turn " + str(degree))

        turnAmptitude = 0.5
        if degree > 0:
            turnAmptitude = -turnAmptitude
        else:
            degree = -degree
        if degree < 20:
            turnAmptitude = turnAmptitude * degree / 18

            degree = 20
        iter_nums = degree / 360 * 1900
        self.mGaitManager.setXAmplitude(0)

        while iter_nums > 0:
            self.mGaitManager.setAAmplitude(turnAmptitude)
            self.mGaitManager.step(int(self.mTimeStep2))  # 步态生成器生成一个步长的动作
            self.myStep2()
            iter_nums -= 1
        self.finished = True
        self.mGaitManager.setAAmplitude(0)    
   
   
   
   
    def rotate(self, angle):
        """
  
        :return:
        """
        print("Rotating:"+str(angle))
        self.falldownMark = False
        target = self.yawIntegrater + angle * self.valuePer2PI / 360
        targetRange = [target - self.valuePer2PI / 36, target + self.valuePer2PI / 36]
        self.mGaitManager.start()  # 步态生成器进入行走状态
        self.mGaitManager.setYAmplitude(0)
        self.mGaitManager.setXAmplitude(0)
        self.checkGyro_Enable = False
        while True:
            if self.yawIntegrater > targetRange[1]:
                if self.yawIntegrater - targetRange[1] > targetRange[0] + self.valuePer2PI - self.yawIntegrater:
                    self.mGaitManager.setAAmplitude(0.5)  # 左转
                else:
                    self.mGaitManager.setAAmplitude(-0.5)  # 右转，积分器减小
            elif self.yawIntegrater < targetRange[0]:
                if targetRange[0] - self.yawIntegrater < self.valuePer2PI - targetRange[1] + self.yawIntegrater:
                    self.mGaitManager.setAAmplitude(0.5)  # 左转
                else:
                    self.mGaitManager.setAAmplitude(-0.5)  # 右转，积分器减小
            else:
                self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                self.wait(200)  # 等待200ms
                self.checkGyro_Enable = True
                self.mGaitManager.stop()
                print("Rotate Finished")
                return
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()
           
            
    def stepUp(self):
        # 
        # self.mMotor[12].setPosition(1.0)  # 右膝
        # self.mMotor[17].setPosition(0.2)  # 左脚踝x轴
        # self.mMotor[16].setPosition(0.5)  # 右脚踝x轴
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[12].setPosition(2.0)  # 右膝
        # self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(-0.)  # 右膝
        # self.mMotor[16].setPosition(-0.06)  # 右足
        # self.mMotor[12].setPosition(0.3)  # 右膝
        # self.mMotor[14].setPosition(0.)  # 右脚踝
        # self.mMotor[14].setPosition(-0.6)  # 右脚踝
        # self.mMotor[16].setPosition(-0.)  # 右足
        # self.mMotor[16].setPosition(-0.02)  # 右足
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[17].setPosition(-0.)  # 左足
        # self.mMotor[13].setPosition(0.)  # 左膝
        # for _ in range(15):
        #     self.myStep()
        # self.mMotor[15].setPosition(-1.39)  # 左脚踝
        # self.mMotor[14].setPosition(-0.5)  # 右脚踝
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(1.1)  # 右膝
        # for _ in range(15):
        #     self.myStep()
        # self.mMotor[14].setPosition(0.2)  # 右脚踝
        # self.mMotor[11].setPosition(0.9)  # 左胯骨x轴
        # for _ in range(5):
        #     self.myStep()
        # self.mMotor[0].setPosition(0.3)  # 右肩
        # self.mMotor[2].setPosition(1.0)  # 右臂
        # self.mMotor[4].setPosition(-1.5)  # 右肘
        # self.mMotor[17].setPosition(-0.4)  # 左脚踝x轴
        # self.mMotor[16].setPosition(-0.2)  # 右脚踝x轴
        # for _ in range(30):
        #     self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[13].setPosition(-1.0)  # 左膝
        # self.mMotor[17].setPosition(-0.5)  # 左脚踝x轴
        # self.mMotor[16].setPosition(-0.2)  # 右脚踝x轴
        # for _ in range(30):
        #     self.myStep()
        # self.mMotor[13].setPosition(-2.0)  # 左膝
        # self.mMotor[11].setPosition(1.68)  # 左胯骨x轴
        # for _ in range(30):
        #     self.myStep()
        # self.mMotor[13].setPosition(0.)  # 左膝
        # self.mMotor[13].setPosition(-1.0)  # 左膝
        # for _ in range(10):
        #     self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        # for _ in range(20):
        #     self.myStep()

        
        # self.mMotor[12].setPosition(1.0)  # 右膝
        # self.mMotor[17].setPosition(0.2)  # 左脚踝x轴
        # self.mMotor[16].setPosition(0.5)  # 右脚踝x轴
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[12].setPosition(2.0)  # 右膝
        # self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(-0.)  # 右膝
        # self.mMotor[16].setPosition(-0.06)  # 右足
        # self.mMotor[12].setPosition(0.3)  # 右膝
        # self.mMotor[14].setPosition(0.)  # 右脚踝
        # self.mMotor[14].setPosition(-0.6)  # 右脚踝
        # self.mMotor[16].setPosition(-0.)  # 右足
        # self.mMotor[16].setPosition(-0.)  # 右足
        #
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[17].setPosition(-0.)  # 左足
        # self.mMotor[13].setPosition(0.)  # 左膝
        # for _ in range(15):
        #     self.myStep()
        # self.mMotor[15].setPosition(-1.39)  # 左脚踝
        # self.mMotor[14].setPosition(-0.49)  # 右脚踝
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(1.1)  # 右膝
        #
        # for _ in range(5):
        #     self.myStep()
        # self.mMotor[14].setPosition(-0.1)  # 右脚踝
        # self.mMotor[11].setPosition(0.9)  # 左胯骨x轴
        # for _ in range(5):
        #     self.myStep()
        # self.mMotor[0].setPosition(0.3)  # 右肩
        # self.mMotor[2].setPosition(1.0)  # 右臂
        # self.mMotor[4].setPosition(-1.5)  # 右肘
        # self.mMotor[17].setPosition(-0.4)  # 左脚踝x轴
        # self.mMotor[16].setPosition(-0.2)  # 右脚踝x轴
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[13].setPosition(-0.4)  # 左膝
        # for _ in range(30):
        #     self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        # for _ in range(20):
        #     self.myStep()

        self.mMotor[12].setPosition(1.0)  # 右膝
        self.mMotor[17].setPosition(0.2)  # 左脚踝x轴
        self.mMotor[16].setPosition(0.5)  # 右脚踝x轴
        for _ in range(20):
            self.myStep()
        self.mMotor[12].setPosition(2.0)  # 右膝
        self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        for _ in range(10):
            self.myStep()
        self.mMotor[12].setPosition(-0.)  # 右膝
        self.mMotor[16].setPosition(-0.06)  # 右足
        self.mMotor[12].setPosition(0.3)  # 右膝
        self.mMotor[14].setPosition(0.)  # 右脚踝
        self.mMotor[14].setPosition(-0.6)  # 右脚踝
        self.mMotor[16].setPosition(-0.)  # 右足
        self.mMotor[16].setPosition(-0.)  # 右足
        for _ in range(20):
            self.myStep()
        self.mMotor[17].setPosition(-0.)  # 左足
        self.mMotor[13].setPosition(0.)  # 左膝
        for _ in range(15):
            self.myStep()
        self.mMotor[15].setPosition(-1.39)  # 左脚踝
        self.mMotor[14].setPosition(-0.49)  # 右脚踝
        for _ in range(10):
            self.myStep()
        self.mMotor[12].setPosition(1.1)  # 右膝
        for _ in range(5):
            self.myStep()
        self.mMotor[14].setPosition(-0.1)  # 右脚踝
        self.mMotor[11].setPosition(0.9)  # 左胯骨x轴
        for _ in range(5):
            self.myStep()
        self.mMotor[0].setPosition(0.3)  # 右肩
        self.mMotor[2].setPosition(1.0)  # 右臂
        self.mMotor[4].setPosition(-1.5)  # 右肘
        self.mMotor[17].setPosition(-0.4)  # 左脚踝x轴
        self.mMotor[16].setPosition(-0.2)  # 右脚踝x轴
        for _ in range(20):
            self.myStep()
        self.mMotor[13].setPosition(-1.0)  # 左膝
        self.mMotor[11].setPosition(1.5)  # 左胯骨x轴
        for _ in range(30):
            self.myStep()
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        self.wait(400)
        
        
    def stepDown(self):
      
        # self.mMotor[12].setPosition(1.0)  # 右膝
        # self.mMotor[17].setPosition(0.2)  # 左足
        # self.mMotor[16].setPosition(0.5)  # 右足
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[12].setPosition(2.0)  # 右膝
        # self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(-0.)  # 右膝
        # self.mMotor[12].setPosition(0.3)  # 右膝
        # self.mMotor[14].setPosition(0.)  # 右脚踝
        # self.mMotor[14].setPosition(-0.1)  # 右脚踝
        # self.mMotor[16].setPosition(0.)  # 右足
        # self.mMotor[16].setPosition(-0.25)  # 右足
        # for _ in range(15):
        #     self.myStep()
        # self.mMotor[13].setPosition(0.)  # 左膝
        # self.mMotor[13].setPosition(-1.22)  # 左膝
        # self.mMotor[15].setPosition(-0.)  # 左脚踝
        # self.mMotor[15].setPosition(-0.85)  # 左脚踝
        # self.mMotor[1].setPosition(-0.3)  # 左肩
        # self.mMotor[3].setPosition(-1.0)  # 左臂
        # self.mMotor[5].setPosition(1.5)  # 左肘
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[17].setPosition(-0.)  # 左足
        # for _ in range(50):
        #     self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        # for _ in range(20):
        #     self.myStep()

    
        # self.mMotor[12].setPosition(1.0)  # 右膝
        # self.mMotor[17].setPosition(0.2)  # 左足
        # self.mMotor[16].setPosition(0.5)  # 右足
        # for _ in range(20):
        #     self.myStep()
        # self.mMotor[12].setPosition(2.0)  # 右膝
        # self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[12].setPosition(-0.)  # 右膝
        # self.mMotor[12].setPosition(0.3)  # 右膝
        # self.mMotor[14].setPosition(0.)  # 右脚踝
        # self.mMotor[14].setPosition(-0.1)  # 右脚踝
        # self.mMotor[16].setPosition(0.)  # 右足
        # self.mMotor[16].setPosition(-0.25)  # 右足
        # for _ in range(15):
        #     self.myStep()
        # self.mMotor[13].setPosition(0.)  # 左膝
        # self.mMotor[13].setPosition(-1.22)  # 左膝
        # self.mMotor[15].setPosition(-0.)  # 左脚踝
        # self.mMotor[15].setPosition(-0.85)  # 左脚踝
        # self.mMotor[1].setPosition(-0.3)  # 左肩
        # self.mMotor[3].setPosition(-1.0)  # 左臂
        # self.mMotor[5].setPosition(1.5)  # 左肘
        # for _ in range(10):
        #     self.myStep()
        # self.mMotor[17].setPosition(-0.)  # 左足
        # for _ in range(50):
        #     self.myStep()
        # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        # for _ in range(20):
        #     self.myStep()

  
        self.mMotor[12].setPosition(1.0)  # 右膝
        self.mMotor[17].setPosition(0.2)  # 左足
        self.mMotor[16].setPosition(0.5)  # 右足
        for _ in range(20):
            self.myStep()
        self.mMotor[12].setPosition(2.0)  # 右膝
        self.mMotor[10].setPosition(-1.77)  # 右胯骨x轴
        for _ in range(10):
            self.myStep()
        self.mMotor[12].setPosition(-0.)  # 右膝
        self.mMotor[12].setPosition(0.3)  # 右膝
        self.mMotor[14].setPosition(0.)  # 右脚踝
        self.mMotor[14].setPosition(-0.1)  # 右脚踝
        self.mMotor[16].setPosition(0.)  # 右足
        self.mMotor[16].setPosition(-0.25)  # 右足
        for _ in range(15):
            self.myStep()
        self.mMotor[13].setPosition(0.)  # 左膝
        self.mMotor[13].setPosition(-1.22)  # 左膝
        self.mMotor[15].setPosition(-0.)  # 左脚踝
        self.mMotor[15].setPosition(-0.85)  # 左脚踝
        self.mMotor[1].setPosition(-0.3)  # 左肩
        self.mMotor[3].setPosition(-1.0)  # 左臂
        self.mMotor[5].setPosition(1.5)  # 左肘
        for _ in range(10):
            self.myStep()
        self.mMotor[17].setPosition(-0.)  # 左足
        for _ in range(50):
            self.myStep()
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势
        self.wait(420)

            
            
  
if __name__ == '__main__':
    print(sys.executable)
    print ("Python Version {}".format(str(sys.version).replace('\n', '')))
    walk = Walk()  # 初始化Walk类
    walk.step()  # 运行控制器

