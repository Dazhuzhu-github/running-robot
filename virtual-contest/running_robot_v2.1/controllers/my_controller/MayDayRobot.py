#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
from controller import Robot,Camera,Gyro,LED,Accelerometer,Motor,PositionSensor
import os
import sys

import cv2 as cv
import numpy as np 
import math
import imutils
import platform

# 功能：将webots的路径添加至包的搜索路径
# 作者：webots
print(os.environ.get("WEBOTS_HOME"))
libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op', 'libraries',
                           'python38')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
for i in sys.path:
    print(i)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

debug = True

# 功能：根据系统选择合适的路径前缀
# 作者：lrh
# 备注：需要根据实际情况做修改
print(platform.system())
if(platform.system()=='Windows'):
    workspace_root = 'D:/running_robot_xx'
elif platform.system() =='Darwin':
    workspace_root = '/Volumes/Data/running_robot_xx'
else:
    workspace_root = "/mnt/hgfs/D/running_robot_xx"

# 字典功能：李镕合的颜色字典
# 作者：lrh
color_dictionary = {
'lan_yellow':((24,40,40),(30,255,255)),
'keng_green':((60,70,35),(71,255,255)),
'lei_gray':((30,3,83),(112,86,205)),
'heng_blue':((113,235,100),(122,255,165)),
'lei_black':((88,83,0),(132,255,20)),
'door_floor':((0,0,206),(180,45,255)),
'door_color':((11,69,20),(29,219,84)),
'ball_wood':((11,23,89),(24,255,255)),
'ball':((78,72,24),(109,228,150)),
'ball_target':((5,162,0),(18,255,110)),
'step_blue':((101,87,186),(109,164,212))
}

# 函数功能：获取最大轮廓面积
# 作者：tsinghua
def getAreaSumContour(contours):
    contour_area_sum = 0
    for c in contours:  # 历遍所有轮廓
        contour_area_sum += math.fabs(cv.contourArea(c))  # 计算轮廓面积
    return contour_area_sum  # 返回最大的面积

# 函数功能：获取最大轮廓边缘点集
# 作者：tsinghua
def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    #area_max_contour = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]])
    # [[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]]
    # c[0] = [[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]
    # c[0][0][0][n] = [0, 0]
    
    for c in contours:  # 历遍所有轮廓
        contour_area_temp = math.fabs(cv.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 10:  #只有在面积大于25时，最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour = c
    return area_max_contour, contour_area_max  # 返回最大的轮廓

# 函数功能：和自己的HSV工具对接，方便快速设置默认阈值
# 作者：lrh
def debug_def_hsv(colorRange):
    dattxt = open(workspace_root+"/def.txt","w")
    if dattxt is not None:
        dattxt.write(str(colorRange[0][0]))
        dattxt.write('\n')
        dattxt.write(str(colorRange[0][1]))
        dattxt.write('\n')
        dattxt.write(str(colorRange[0][2]))
        dattxt.write('\n')
        dattxt.write(str(colorRange[1][0]))
        dattxt.write('\n')
        dattxt.write(str(colorRange[1][1]))
        dattxt.write('\n')
        dattxt.write(str(colorRange[1][2]))
        dattxt.write('\n')

class Walk():
    def __init__(self):
        self.robot = Robot()  # 初始化Robot类以控制机器人
        self.mTimeStep = int(self.robot.getBasicTimeStep())  # 获取当前每一个仿真步所仿真时间mTimeStep
        self.HeadLed = self.robot.getLED('HeadLed')  # 获取头部LED灯
        self.EyeLed = self.robot.getLED('EyeLed')  # 获取眼部LED灯
        self.HeadLed.set(0xff0000)  # 点亮头部LED灯并设置一个颜色
        self.EyeLed.set(0xa0a0ff)  # 点亮眼部LED灯并设置一个颜色
        self.mAccelerometer = self.robot.getAccelerometer('Accelerometer')  # 获取加速度传感器
        self.mAccelerometer.enable(self.mTimeStep)  # 激活传感器，并以mTimeStep为周期更新数值
        self.fup = 0
        self.fdown = 0   # 定义两个类变量，用于之后判断机器人是否摔倒  
        
        self.mGyro = self.robot.getGyro('Gyro')  # 获取陀螺仪
        self.mGyro.enable(self.mTimeStep)  # 激活陀螺仪，并以mTimeStep为周期更新数值
        
        self.camera = self.robot.getCamera("Camera")
        self.camera.enable(self.mTimeStep)

        self.cam_height = self.camera.getHeight()
        self.cam_width = self.camera.getWidth()
        print("Camera resolution:("+str(self.cam_width)+","+str(self.cam_height)+")")

        self.positionSensors = []  # 初始化关节角度传感器
        self.mMotors = []
        self.minMotorPositions = []
        self.maxMotorPositions = []
        self.positionSensorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                                    'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
                                    'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
                                    'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
                                    'FootR', 'FootL', 'Neck', 'Head')  # 初始化各传感器名

        # 获取各传感器并激活，以mTimeStep为周期更新数值
        for i in range(0, len(self.positionSensorNames)):
            self.mMotors.append(self.robot.getMotor(self.positionSensorNames[i]))
            self.minMotorPositions.append(self.mMotors[i].getMinPosition())
            self.maxMotorPositions.append(self.mMotors[i].getMaxPosition())
            self.positionSensors.append(self.robot.getPositionSensor(self.positionSensorNames[i] + 'S'))
            self.positionSensors[i].enable(self.mTimeStep)

        self.mKeyboard = self.robot.getKeyboard()  # 初始化键盘读入类
        self.mKeyboard.enable(self.mTimeStep)  # 以mTimeStep为周期从键盘读取

        self.mMotionManager = RobotisOp2MotionManager(self.robot)  # 初始化机器人动作组控制器
        self.mGaitManager = RobotisOp2GaitManager(self.robot, "config.ini")  # 初始化机器人步态控制器

    # 函数功能：步长仿真
    # 作者：webots
    def myStep(self):
        ret = self.robot.step(self.mTimeStep)
        if ret == -1:
            exit(0)

    # 函数功能：延迟
    # 作者：webots
    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms / 1000.0
        while s + startTime >= self.robot.getTime():
            self.myStep()
    
    # 函数功能：寻找画面中的色块
    # 作者：lrh
    def findColor(self,colorRange):
        if self.hsv is not None:
            mask = cv.inRange(self.hsv,colorRange[0],colorRange[1])
            mask = cv.erode(mask,None,iterations=2)
            mask = cv.dilate(mask,None,iterations=2)
            for x in range(0, self.cam_width,1):
                isBreak = False
                for y in range(0, self.cam_height,1):
                    if mask[y,x] != 0:
                        return True
        return False
    
    # 函数功能：根据颜色走在（比较纯色的）地面上
    # 作者：lrh
    # 感谢：tsinghua
    def goOnColor(self,colorRange,colorRange_next,hinder_color=None,persent_of_next=20):
        if debug is True:
            if hinder_color is not None:
                debug_def_hsv(colorRange)
            else:
                debug_def_hsv(colorRange) 
        if self.isOnTask == True: # 判断是否进入下一关
            mask = cv.inRange(self.hsv,colorRange_next[0],colorRange_next[1])
            opened=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
            closed=cv.morphologyEx(opened,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
            ( contours, hierarchy) = cv.findContours(closed,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
            area_sum = getAreaSumContour(contours)
            persent = round(100*area_sum/(self.cam_width * self.cam_height),2)
            if persent > persent_of_next:
                self.mGaitManager.setXAmplitude(1.0)
                self.mGaitManager.setAAmplitude(0.0)
                return False
        isLeiOnCentre = False
        mask = cv.inRange(self.hsv,colorRange[0],colorRange[1])
        opened=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
        closed=cv.morphologyEx(opened,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
        ( contours, hierarchy) = cv.findContours(closed,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        area_sum = getAreaSumContour(contours)
        persent = round(100*area_sum/(self.cam_width * self.cam_height),2)
        if persent < 10:
            self.mGaitManager.setXAmplitude(0.0)
            self.mGaitManager.setAAmplitude(0.2)
            return True
        areaMaxContour,area_max=getAreaMaxContour(contours)
        if areaMaxContour is not None:# 最大识别面的轮廓点集
            bottom_left = areaMaxContour[0][0] # 点集的第一个点
            bottom_right = areaMaxContour[0][0]
            quanzhong = 0.6
            for c in areaMaxContour:#遍历点集，c[0]即[x,y]
                if c[0][0] + quanzhong * (self.cam_height - c[0][1]) < bottom_left[0] + quanzhong* (self.cam_height - bottom_left[1]):
                    bottom_left = c[0]
                if c[0][0] + quanzhong * c[0][1] > bottom_right[0] + quanzhong * bottom_right[1]:
                    bottom_right = c[0]
            angle_bottom = - math.atan((bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0])) * 180.0 / math.pi
            self.last_A = 0
            bottom_center_x = int((bottom_right[0] + bottom_left[0]) / 2)
            bottom_center_y = int((bottom_right[1] + bottom_left[1]) / 2)
            if hinder_color is not None: # 是否有需要避让的设置
                mask = cv.inRange(self.hsv,hinder_color[0],hinder_color[1])
                mask = cv.erode(mask,None,iterations=1)
                mask = cv.dilate(mask,None,iterations=1)
                cnts = cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                if len(cnts)>0:
                    x,y,r = 0,0,0
                    for i in cnts:
                        ((X,Y),R) = cv.minEnclosingCircle(i)
                        if Y > y:
                            x,y,r = X,Y,R
                    r_range = 6
                    if bottom_center_x - r_range*r< x <bottom_center_x + r_range*r:
                        isLeiOnCentre = True
                    else:
                        pass
            if debug is True:
                mask = cv.cvtColor(closed,cv.COLOR_GRAY2BGR)#debug
                mask = cv.drawContours(mask,areaMaxContour,-1,(0,0,255),1)#debug
                mask = cv.circle(mask,(bottom_center_x,bottom_center_y),3,(0,0,255),1)#debug
                mask = cv.circle(mask,(bottom_right[0],bottom_right[1]),3,(255,0,0),1)#debug
                mask = cv.circle(mask,(bottom_left[0],bottom_left[1]),3,(255,0,0),1)#debug
                mask = cv.line(mask,(bottom_right[0],bottom_right[1]),(bottom_left[0],bottom_left[1]),(200,50,200),2)#debug
                cv.imwrite(workspace_root+'/temp_image/raw.png',mask)#debug
        else:
            self.last_A = 0
            angle_bottom = 0
            bottom_center_x = 0.5 * self.cam_width
            bottom_center_y = 0
        self.isOnTask = True
        self.mGaitManager.setXAmplitude(1.0)
        if bottom_center_y < 0.5 * self.cam_height:
            self.last_A = 0
        elif bottom_center_y >= 0.5 * self.cam_height:
            if angle_bottom < -3:
                self.last_A = -0.5
            elif angle_bottom > 3 :
                self.last_A = 0.5
            elif -3 <= angle_bottom <= 3:
                if bottom_center_x < 0.45 * self.cam_width:
                    self.last_A = 0.5
                elif bottom_center_x > 0.55 * self.cam_width:
                    self.last_A = -0.5
                else:
                    self.last_A = -(bottom_center_x / self.cam_width - 0.5) * 10 # -0.5 ~ +0.5
                    if isLeiOnCentre is True:
                        p=[]
                        for P in areaMaxContour:
                            if P[0][1] == int(y):
                                p.append(P[0][0])
                        p.sort()
                        if x - min(p) < 60 :
                            self.last_A = -0.5
                        else:
                            self.last_A = 0.3
        self.mGaitManager.setAAmplitude(self.last_A)
        return True
    
    # 函数功能：翻跟头之前对准栏杆
    # 作者：lrh
    def align_bar(self,colorRange):
        mask = cv.inRange(self.hsv,colorRange[0],colorRange[1])
        opened=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
        closed=cv.morphologyEx(opened,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
        lines = cv.HoughLinesP(closed,1,np.pi/180,30,minLineLength=30,maxLineGap=10)
        set_x = (0.0)
        ret = False
        self.last_A = (0.0)
        if lines is not None and len(lines)>0:
            lines = lines[:,0,:]
            mask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
            angle_lis = []
            yc_lis = []
            for x,y,X,Y in lines:
                angle_lis.append( math.atan2((Y-y),(X-x))*180/math.pi )
                yc_lis.append((y+Y)/2)
                if debug is True:
                    cv.line(mask,(x,y),(X,Y),(255,0,0),1)
            for i in angle_lis:
                if math.fabs(i) <= 11:
                    pass
                    # self.last_A =(0.0)
                elif i > 11:
                    self.last_A = (-0.2)
                    break
                elif i < -11:
                    self.last_A = (0.2)
                    break
            if self.last_A == 0.0:# x 位移判断
                for i in yc_lis:
                    if i < self.cam_height - 20:
                        set_x = (0.5)
                        break
                if set_x == 0.0:
                    if self.cam_motPos > -0.10:
                        self.cam_motPos = self.cam_motPos - 0.05
                    else:
                        ret = True
            if debug is True:
                # print(angle_lis)
                cv.imwrite(workspace_root+'/temp_image/raw.png',mask)#debug
        elif self.cam_motPos != 0.0:
            ret = True
            if debug is True:
                print("No line find but motor is not 0")
        else:
            if debug is True:
                set_x = 0.5
                print("No line find")
        self.mMotors[19].setPosition(self.cam_motPos)
        self.mGaitManager.setXAmplitude(set_x)
        self.mGaitManager.setAAmplitude(self.last_A)
        return ret
    
    # 函数功能：找门
    # 作者：lrh
    def findoor(self,colorRange):
        if debug is True:
            debug_def_hsv(colorRange)
        set_x = 0
        ret = False
        mask = cv.inRange(self.hsv,colorRange[0],colorRange[1])
        opened=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
        closed=cv.morphologyEx(opened,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
        self.cam_motPos = self.maxMotorPositions[19]
        ( contours, hierarchy) = cv.findContours(closed,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        area_sum = getAreaSumContour(contours)
        persent = round(100*area_sum/(self.cam_width * self.cam_height),2)
        if area_sum < 3:
            self.mGaitManager.setXAmplitude(0.0)
            self.mGaitManager.setAAmplitude(0.3)
            self.mMotors[19].setPosition(self.cam_motPos)
            return False
        else:
            mask = cv.cvtColor(closed,cv.COLOR_GRAY2BGR)
            cnts = cv.findContours(closed.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts)>0:
                X,Y,R=0,0,0
                for i in cnts:
                    (x,y),r = cv.minEnclosingCircle(i)
                    if r > R:
                        X,Y,R=x,y,r
                mask = cv.circle(mask,(int(x),int(y)),int(r),(0,0,255),1)
                cv.imwrite(workspace_root+'/temp_image/raw.png',mask)
                offset = x - self.cam_width//2
                if offset > 5:
                    self.last_A = -0.3
                elif offset < -5:
                    self.last_A = 0.3
                else:
                    self.last_A = 0
                set_x = 1
            else:
                print("None")
        areaMaxContour,area_max=getAreaMaxContour(contours)
        if persent > 11:
            ret = True
            self.cam_motPos = 0.5
        self.mMotors[19].setPosition(self.cam_motPos)
        self.mGaitManager.setXAmplitude(set_x)
        self.mGaitManager.setAAmplitude(self.last_A)
        return ret
    
    # 函数功能：翻跟头
    # 作者：徐达
    def fan_gen_tou(self):
        self.myStep()
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        self.mGaitManager.stop()
        self.wait(1500)  # 等待200ms
        self.setvalue = np.zeros(20)

        self.setvalue[0]=1.5
        self.setvalue[1]=-1.5
        
        self.setvalue[2]=-0.6
        self.setvalue[3]=0.6
        
        self.setvalue[4]=-1.2
        self.setvalue[5]=1.2
        
        self.setvalue[10]=-0.5
        self.setvalue[11]=0.5
        
        self.setvalue[12]=1
        self.setvalue[13]=-1
        
        self.setvalue[14]=0.5
        self.setvalue[15]=-0.5
        self.mMotors[0].setPosition(self.setvalue[0])
        self.mMotors[1].setPosition(self.setvalue[1])
        self.mMotors[2].setPosition(self.setvalue[2])
        self.mMotors[3].setPosition(self.setvalue[3])
        self.mMotors[4].setPosition(self.setvalue[4])
        self.mMotors[5].setPosition(self.setvalue[5])
        self.mMotors[10].setPosition(self.setvalue[10])
        self.mMotors[11].setPosition(self.setvalue[11])
        self.mMotors[12].setPosition(self.setvalue[12])
        self.mMotors[13].setPosition(self.setvalue[13])
        self.mMotors[14].setPosition(self.setvalue[14])
        self.mMotors[15].setPosition(self.setvalue[15])
        self.wait(200)

        self.setvalue[10]=-0.7
        self.setvalue[11]=0.7

        self.setvalue[12]=0.5
        self.setvalue[13]=-0.5
        
        self.setvalue[14]=0.2
        self.setvalue[15]=-0.2
        self.mMotors[10].setPosition(self.setvalue[10])
        self.mMotors[11].setPosition(self.setvalue[11])
        self.mMotors[12].setPosition(self.setvalue[12])
        self.mMotors[13].setPosition(self.setvalue[13])
        self.mMotors[14].setPosition(self.setvalue[14])
        self.mMotors[15].setPosition(self.setvalue[15])
        self.wait(200)

        self.setvalue[12]=0
        self.setvalue[13]=-0
        
        self.setvalue[14]=-0.2
        self.setvalue[15]=0.2
        self.mMotors[10].setPosition(self.setvalue[10])
        self.mMotors[11].setPosition(self.setvalue[11])
        self.mMotors[12].setPosition(self.setvalue[12])
        self.mMotors[13].setPosition(self.setvalue[13])
        self.mMotors[14].setPosition(self.setvalue[14])
        self.mMotors[15].setPosition(self.setvalue[15])
        self.wait(200); 
        
        self.setvalue[10]=-1
        self.setvalue[11]=1
        self.setvalue[14]=-0.4
        self.setvalue[15]=0.4
        self.mMotors[10].setPosition(self.setvalue[10])
        self.mMotors[11].setPosition(self.setvalue[11])
        self.mMotors[14].setPosition(self.setvalue[14])
        self.mMotors[15].setPosition(self.setvalue[15])
        self.wait(500)
        
        self.setvalue[0]=2.5
        self.setvalue[1]=-2.5
        
        self.setvalue[4]=1
        self.setvalue[5]=-1
        
        self.mMotors[0].setPosition(self.setvalue[0])
        self.mMotors[1].setPosition(self.setvalue[1])
        self.mMotors[4].setPosition(self.setvalue[4])
        self.mMotors[5].setPosition(self.setvalue[5])
        self.wait(500)
        
        self.setvalue[10]=-1.3
        self.setvalue[11]=1.3
        self.setvalue[14]=0.4
        self.setvalue[15]=-0.4
        self.mMotors[10].setPosition(self.setvalue[10]) 
        self.mMotors[11].setPosition(self.setvalue[11])
        self.mMotors[14].setPosition(self.setvalue[14])
        self.mMotors[15].setPosition(self.setvalue[15])
        self.wait(500)

        self.setvalue[10]=-1.77
        self.setvalue[11]=-0.5
        self.setvalue[13]=-2.2
        self.mMotors[10].setPosition(self.setvalue[10])
        self.mMotors[11].setPosition(self.setvalue[11])
        self.mMotors[13].setPosition(self.setvalue[13])
        self.wait(400)
        
        self.setvalue[10]=0.45
        self.setvalue[12]=2.2    
        self.mMotors[10].setPosition(self.setvalue[10])
        self.mMotors[12].setPosition(self.setvalue[12])
        self.wait(800)

        self.wait(1)                                        #动作完成

        for i in range(150):
            self.checkIfFallen()                                   #等待站起来
            self.myStep()
        self.mGaitManager.start()
        self.mMotionManager.playPage(9)
        
    # 函数功能：找球
    # 作者：lrh
    def findBall(self,colorRange):
        if debug is True:
            debug_def_hsv(colorRange)
        mask = cv.inRange(self.hsv,colorRange[0],colorRange[1])
        cnts = cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        ret = False
        x_rate = 1.0
        off_y = -50
        x,y,r=0,0,0
        offset = 0
        if len(cnts) > 0:
            x,y,r = 0,0,0
            for i in cnts:
                (X,Y),R=cv.minEnclosingCircle(i)
                if R>r:
                    x,y,r=X,Y,R
            if (x,y,r) != (0,0,0):
                offset = x-self.cam_width//2
                off_y = y - self.cam_height//2

        if offset < -3:
            self.last_A = 0.5
        elif offset > 3:
            self.last_A = -0.5
        else :
            self.last_A = 0

        if off_y > 5:
            self.cam_motPos = self.cam_motPos - 0.01
            if self.cam_motPos <= self.minMotorPositions[19]:
                self.cam_motPos = self.minMotorPositions[19]
                ret = True
        self.mMotors[19].setPosition(self.cam_motPos)
        self.mGaitManager.setXAmplitude(x_rate)
        self.mGaitManager.setAAmplitude(self.last_A)
        return ret 

    def kick_ball(self,colorRange):
        if debug is True:
            debug_def_hsv(colorRange)
        mask = cv.inRange(self.hsv,colorRange[0],colorRange[1])
        cnts = cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            x,y,r = 0,0,0
            for i in cnts:
                (X,Y),R=cv.minEnclosingCircle(i)
                if R>r:
                    x,y,r=X,Y,R
            if (x,y,r) != (0,0,0):
                offset = x-self.cam_width//2
                if offset > 2:
                    self.last_A = -0.3
                    self.mGaitManager.setAAmplitude(self.last_A)
                    self.las_motion = 13
                elif offset < 2:
                    self.last_A = 0.3
                    self.mGaitManager.setAAmplitude(self.last_A)
                    self.las_motion = 12
                else:
                    for i in range(1,20):
                        self.mGaitManager.setAAmplitude(self.last_A)
                        self.mGaitManager.step(self.mTimeStep)
                        self.myStep()
                    self.mGaitManager.stop()
                    self.wait(200)
                    return self.las_motion
        self.cam_motPos = 0.3
        self.mGaitManager.setXAmplitude(0)
        self.mGaitManager.setYAmplitude(0)
        self.mMotors[19].setPosition(self.cam_motPos)


    # 函数功能：主函数
    # 作者：lrh
    def run(self):
        print("-------Walk example of ROBOTIS OP2-------")
        print("This example illustrates Gait Manager")
        print("Press the space bar to start/stop walking")
        print("Use the arrow keys to move the robot while walking")
        self.myStep()  # 仿真一个步长，刷新传感器读数
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        self.wait(200)  # 等待200ms
        self.mGaitManager.setBalanceEnable(True)
        self.isWalking = False  # 初始时机器人未进入行走状态
        self.cam_motPos = 0.0
        self.door_isFace = 0
        self.last_A = 0
        stateMachine = "Start" 
        debug_partly = False
        if debug_partly is True:#debug
            stateMachine = "Ball_Find"
            self.mGaitManager.start()
            self.isWalking = True
            self.wait(200)
            self.cam_motPos = 0.8
            self.mMotors[19].setPosition(self.cam_motPos)
        self.isOnTask = False
        while True:
            self.checkIfFallen()
            raw = np.frombuffer(self.camera.getImage(),np.uint8).reshape((self.cam_height,self.cam_width,4))
            if debug is True:
                cv.imwrite(workspace_root+'/temp_image/cam.png',raw)#debug
            gauss = cv.GaussianBlur(raw,(3,3),0)
            self.hsv = cv.cvtColor(gauss,cv.COLOR_BGR2HSV)
            if stateMachine == "Start":
                ans = self.findColor(color_dictionary['lan_yellow'])
                if ans == True:
                    stateMachine = "Bond"
                    print(stateMachine)
                    debug_def_hsv(color_dictionary['lan_yellow'])
            elif stateMachine == "Bond":
                ans = self.findColor(color_dictionary['lan_yellow'])
                if ans == False:
                    self.mGaitManager.start()
                    self.isWalking = True
                    self.wait(200)
                    self.mGaitManager.setXAmplitude(1.0)
                    self.mGaitManager.setAAmplitude(0.0)
                    stateMachine = "Go"
                    print(stateMachine)
                    startTime = self.robot.getTime()
            elif stateMachine == "Go": 
                if self.robot.getTime() - startTime > 1.5: # 1.5s直接往前走
                    stateMachine = "Block"
                    print(stateMachine)
            elif stateMachine == "Block":
                ans = self.goOnColor(color_dictionary['keng_green'],color_dictionary['lei_gray'])
                if ans == False:
                    stateMachine = "Lei"
                    print(stateMachine)
                    self.isOnTask = False
            elif stateMachine == "Lei":
                ans = self.goOnColor(color_dictionary['lei_gray'],color_dictionary['heng_blue'],hinder_color=color_dictionary['lei_black'],persent_of_next=10)
                if ans == False:
                    stateMachine = "heng"
                    print(stateMachine)
                    debug_def_hsv(color_dictionary['heng_blue'])
                    self.cam_motPos = 0.0
                    self.isOnTask = False
            elif stateMachine == "heng":
                ans = self.align_bar(color_dictionary['heng_blue'])
                if ans == True:
                    self.fan_gen_tou()
                    stateMachine = "Door"
                    print(stateMachine)
            elif stateMachine == "Door":
                ans = self.findoor(color_dictionary['keng_green'])
                if ans == True:
                    stateMachine = "Bridge_pre"
                    print(stateMachine)
                    startTime = self.robot.getTime()
            elif stateMachine == "Bridge_pre":
                self.cam_motPos = 0.9
                self.mMotors[19].setPosition(self.cam_motPos)
                self.mGaitManager.setXAmplitude(1.0)
                self.mGaitManager.setAAmplitude(0.0)
                if self.robot.getTime() - startTime > 12: # 5s直接往前走
                    stateMachine = "Bridge"
                    print(stateMachine)
            elif stateMachine == "Bridge":
                self.cam_motPos = 0.4
                self.mMotors[19].setPosition(self.cam_motPos)
                ans = self.goOnColor(color_dictionary['keng_green'],color_dictionary['ball_wood'])
                if ans == False:
                    self.isOnTask = False
                    stateMachine = "Ball"
                    print(stateMachine)
                    startTime = self.robot.getTime()
            elif stateMachine == "Ball":
                ans = self.goOnColor(color_dictionary['ball_wood'],color_dictionary['heng_blue'])
                self.cam_motPos = 0.8
                self.mMotors[19].setPosition(self.cam_motPos)
                if self.robot.getTime() - startTime > 25: # 25s直接往前走
                    self.cam_motPos = 0.8
                    stateMachine = "Ball_Find"
                    print(stateMachine)
            elif stateMachine == "Ball_Find":
                ans = self.findBall(color_dictionary['ball'])
                if ans == True:
                    stateMachine = "Kick"
                    print(stateMachine)
            elif stateMachine == "Kick":
                ans = self.kick_ball(color_dictionary['ball_target'])
                if ans is not None:
                    self.mMotionManager.playPage(ans)
                    self.wait(1000)
                    self.mGaitManager.start()
                    self.wait(200)
                    stateMachine = "find_step"
                    print(stateMachine)
            elif stateMachine == "find_step":
                ans = self.goOnColor(color_dictionary['ball_wood'],color_dictionary['step_blue'])
                if ans == False:
                    stateMachine = "align_step"
                    print(stateMachine)
            elif stateMachine == "align_step":
                ans = self.align_bar(color_dictionary['step_blue'])
                if ans == True:
                    stateMachine = "STOP"
                    print(stateMachine)
            elif stateMachine == "STOP":
                self.mGaitManager.stop()
                self.isWalking = False
                self.wait(200)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
    # 函数功能：摔倒检测与自动复位
    # 作者：webots
    def checkIfFallen(self):
        acc_tolerance = 60.0
        acc_step = 100  # 计数器上限
        acc = self.mAccelerometer.getValues()  # 通过加速度传感器获取三轴的对应值
        if acc[1] < 512.0 - acc_tolerance :  # 面朝下倒地时y轴的值会变小
            self.fup += 1  # 计数器加1
        else :
            self.fup = 0  # 计数器清零
        if acc[1] > 512.0 + acc_tolerance : # 背朝下倒地时y轴的值会变大
            self.fdown += 1 # 计数器加 1
        else :
            self.fdown = 0 # 计数器清零
        
        if self.fup > acc_step :   # 计数器超过100，即倒地时间超过100个仿真步长
            self.mMotionManager.playPage(10) # 执行面朝下倒地起身动作
            self.mMotionManager.playPage(9) # 恢复准备行走姿势
            self.fup = 0 # 计数器清零
        elif self.fdown > acc_step :
            self.mMotionManager.playPage(11) # 执行背朝下倒地起身动作
            self.mMotionManager.playPage(9) # 恢复准备行走姿势
            self.fdown = 0 # 计数器清零

if __name__ == '__main__':
    walk = Walk()  # 初始化Walk类
    walk.run()  # 运行控制器
