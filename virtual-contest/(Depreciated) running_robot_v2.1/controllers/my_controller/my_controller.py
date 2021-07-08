"""my_controller controller."""

# # You may need to import some classes of the controller module. Ex:
# #  from controller import Robot, Motor, DistanceSensor
# from controller import Robot

# # create the Robot instance.
# robot = Robot()

# # get the time step of the current world.
# timestep = int(robot.getBasicTimeStep())

# # You should insert a getDevice-like function in order to get the
# # instance of a device of the robot. Something like:
# #  motor = robot.getDevice('motorname')
# #  ds = robot.getDevice('dsname')
# #  ds.enable(timestep)

# # Main loop:
# # - perform simulation steps until Webots is stopping the controller
# while robot.step(timestep) != -1:
#     # Read the sensors:
#     # Enter here functions to read sensor data, like:
#     #  val = ds.getValue()

#     # Process sensor data here.

#     # Enter here functions to send actuator commands, like:
#     #  motor.setPosition(10.0)
#     pass

# # Enter here exit cleanup code.
from controller import Robot
import os
import sys

libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op', 'libraries', 'python38')

libraryPath = libraryPath.replace('/', os.sep)

sys.path.append(libraryPath)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

class Walk():
    def __init__(self):
        self.robot = Robot();# 初始化 Robot 类以控制机器人
        self.mTimeStep = int(self.robot.getBasicTimeStep()) # 获取当前每一个仿真步所仿真时间 mTimeStep
        self.HeadLed = self.robot.getLED('HeadLed') # 获取头部 LED 灯
        self.EyeLed = self.robot.getLED('EyeLed') # 获取眼部 LED 灯
        self.HeadLed.set(0xff0000) # 点亮头部 LED 灯并设置一个颜色
        self.EyeLed.set(0xa0a0ff) # 点亮眼部 LED 灯并设置一个颜色
        self.mAccelerometer = self.robot.getAccelerometer('Accelerometer') # 获取加速度传感器
        self.mAccelerometer.enable(self.mTimeStep) # 激活传感器， 并以 mTimeStep 为周期更新数值
        self.fup = 0
        self.fdown = 0 # 定义两个类变量， 用于之后判断机器人是否摔倒

        self.mGyro = self.robot.getGyro('Gyro') # 获取陀螺仪
        self.mGyro.enable(self.mTimeStep) # 激活陀螺仪， 并以 mTimeStep 为周期更新数值
        self.positionSensors = [] # 初始化关节角度传感器
        self.positionSensorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
        'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
        'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
        'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
        'FootR', 'FootL', 'Neck', 'Head') # 初始化各传感器名

        # 获取各传感器并激活， 以 mTimeStep 为周期更新数值
        for i in range(0, len(self.positionSensorNames)):
            self.positionSensors.append(self.robot.getPositionSensor(self.positionSensorNames[i] + 'S'))
            self.positionSensors[i].enable(self.mTimeStep)

        self.mKeyboard = self.robot.getKeyboard() # 初始化键盘读入类
        self.mKeyboard.enable(self.mTimeStep) # 以 mTimeStep 为周期从键盘读取

        self.mMotionManager = RobotisOp2MotionManager(self.robot) # 初始化机器人动作组控制器
        self.mGaitManager = RobotisOp2GaitManager(self.robot, " Head ") # 初始化机器人步态控制器

    def myStep(self):
        ret = self.robot.step(self.mTimeStep)
        if ret == -1:
            exit(0)

    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms / 1000.0
        while (s + startTime >= self.robot.getTime()):
            self.myStep()

    def run(self):
        print("-------Walk example of ROBOTIS OP2 python-------")
        print("-------Walk example of ROBOTIS OP2 python-------")
        print("This example illustrates Gait Manager")
        print("Press the space bar to start/stop walking")
        print("Use the arrow keys to move the robot while walking")
        self.myStep() # 仿真一个步长， 刷新传感器读数

        self.mMotionManager.playPage(9) # 执行动作组 9 号动作， 初始化站立姿势， 准备行走
        self.wait(200) # 等待 200ms

        self.isWalking = False # 初始时机器人未进入行走状态

        while True:
            self.checkIfFallen() # 判断是否摔倒
            self.mGaitManager.setXAmplitude(0.0) # 前进为 0
            self.mGaitManager.setAAmplitude(0.0) # 转体为 0
            key = 0 # 初始键盘读入默认为 0
            key = self.mKeyboard.getKey() # 从键盘读取输入
            if key == 32: # 如果读取到空格， 则改变行走状态
                if (self.isWalking): # 如果当前机器人正在走路， 则使机器人停止
                    self.mGaitManager.stop()
                    self.isWalking = False
                    self.wait(200)
                else: # 如果机器人当前停止， 则开始走路
                    self.mGaitManager.start()
                    self.isWalking = True
                    self.wait(200)
            elif key == 315: # 如果读取到‘↑’， 则前进
                self.mGaitManager.setXAmplitude(1.0)
            elif key == 317: # 如果读取到‘↓’， 则后退
                self.mGaitManager.setXAmplitude(-1.0)
            elif key == 316: # 如果读取到‘←’， 则左转
                self.mGaitManager.setAAmplitude(-0.5)
            elif key == 314: # 如果读取到‘→’， 则右转
                self.mGaitManager.setAAmplitude(0.5)
            self.mGaitManager.step(self.mTimeStep) # 步态生成器生成一个步长的动作
            self.myStep() # 仿真一个步长

    def checkIfFallen(self):
        acc_tolerance = 60.0
        acc_step = 100 # 计数器上限
        acc = self.mAccelerometer.getValues() # 通过加速度传感器获取三轴的对应值
        if acc[1] < 512.0 - acc_tolerance : # 面朝下倒地时 y 轴的值会变小
            self.fup += 1 # 计数器加 1
        else :
            self.fup = 0 # 计数器清零
        if acc[1] > 512.0 + acc_tolerance : # 背朝下倒地时 y 轴的值会变大
            self.fdown += 1 # 计数器加 1
        else :
            self.fdown = 0 # 计数器清零

        if self.fup > acc_step : # 计数器超出上限， 即倒地时间超过 acc_step 个仿真步长
            self.mMotionManager.playPage(10) # 执行面朝下倒地起身动作
            self.mMotionManager.playPage(9) # 恢复准备行走姿势
            self.fup = 0 # 计数器清零
        elif self.fdown > acc_step :
            self.mMotionManager.playPage(11) # 执行背朝下倒地起身动作
            self.mMotionManager.playPage(9) # 恢复准备行走姿势
            self.fdown = 0 # 计数器清零

if __name__ == '__main__':
    walk = Walk()
    walk.run()