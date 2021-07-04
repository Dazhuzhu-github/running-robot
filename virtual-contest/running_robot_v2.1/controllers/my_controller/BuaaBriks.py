#coding:utf-8
import rospy
import time
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray, Float64

import std_msgs.msg
import geometry_msgs.msg
from bodyhub.msg import *
from bodyhub.srv import *
from sensor_msgs.msg import*
import cv2
from TrajectoryPlan import *
import numpy as np
from cv_bridge import CvBridge

bridge = CvBridge() 
walkingState = 0
NodeControlId = 6
n=0
global cv_image

def waitForWalkingDone():
    while walkingState != 0:
        pass
def walkingStateCallback(data):
    global walkingState
    walkingState = data.data
def SetBodyhubStatus(id, status):
    # wait a service of "MediumSize/BodyHub/StateJump",eg the service is exist
    rospy.wait_for_service('MediumSize/BodyHub/StateJump')
    client = rospy.ServiceProxy('MediumSize/BodyHub/StateJump', SrvState)# SrvState is the dataform
    client(id, status)
def walkAstep(index):
    global walkingState
    SetBodyhubStatus(NodeControlId, 'reset')
    SetBodyhubStatus(NodeControlId, 'setStatus')
    SetBodyhubStatus(NodeControlId, 'walking')

    if walkingState == 0:
            walkingState = 1

            walkSend = Float64MultiArray()
            walkSend.data = walkList[index]
            walk_pub.publish(walkSend)       #perform move command

            print walkListName[index]
            waitForWalkingDone()


def SendTrajectory(pub, trajectoryPoint):
    for m in range(len(trajectoryPoint[0])):
        jointPosition = []
        for n in range(len(trajectoryPoint)):
            jointPosition.append(trajectoryPoint[n][m].y)

        pub.publish(positions=jointPosition, mainControlID=NodeControlId)
def GetBodyhubStatus():
    rospy.wait_for_service('MediumSize/BodyHub/GetStatus')
    client = rospy.ServiceProxy('MediumSize/BodyHub/GetStatus', SrvString)
    response = client('get')
    return response.data
def WaitTrajectoryExecOver():
    while GetBodyhubStatus() != 'pause':
        pass

def pickBrik(poseL):
    SetBodyhubStatus(NodeControlId, 'reset')
    SetBodyhubStatus(NodeControlId, 'setStatus')
    
    tpObject.setInterval(1000.0)
    tpObject.planningBegin(poseL[0], poseL[1])

    tpObject.setInterval(1500.0)
    for poseIndex in range(2, len(poseL)):
        trajectoryPoint = tpObject.planning(poseL[poseIndex])
        SendTrajectory(jointPositionPub, trajectoryPoint)
        WaitTrajectoryExecOver()

    trajectoryPoint = tpObject.planningEnd()
    SendTrajectory(jointPositionPub, trajectoryPoint)
    WaitTrajectoryExecOver()




def StateCallback(data):
    global n
    n=n+1
    if n>20:
        n=0
        cv_image = bridge.compressed_imgmsg_to_cv2(data, desired_encoding="rgb8")
        cv2.namedWindow("Image window",0)
        cv2.resizeWindow("Image window", 640, 480)
        cv2.moveWindow("Image window",200,200)
        cv2.imshow("Image window", cv_image)   
        cv2.waitKey(1000) 
        #cv2.destroyAllWindows()    
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        #lower_back = np.array([100, 43, 46])#red
        #upper_back = np.array([124, 255, 255])
        lower_back = np.array([100, 200, 200])#goodred
        upper_back = np.array([124, 255, 255])
        frame = cv2.inRange(hsv, lower_back, upper_back)
        frame = cv2.medianBlur(frame, 7)
        cv2.imshow('image',frame)
    # print data


if __name__ == '__main__':
    
   

    #行走控制初始化
    index=0
    walk_pub = rospy.Publisher("gaitCommand", Float64MultiArray, queue_size=1)# 行走指令发布器
    rospy.Subscriber("/MediumSize/BodyHub/WalkingStatus",Float64, walkingStateCallback, queue_size=3)#get status of walking
    
    rospy.init_node('BuaaBriks')#创建节点名
    time.sleep(0.05)

    walkList= [
        [0.1, 0, 0],
        [-0.1,0, 0],
        [0, 0.05, 0],
        [0, -0.013, 0],
        [0.076, 0, 0]
    ]
    walkListName=['前进','后退','左移','右移','前进']


    #动作控制
    # 蹲下
    poseList1 = [
        [0,0,40,-85,-45,0,          0,0,-40,85,45,0,          0,-75,-10,      0,75,10,          0,0,   0,0],
        [0,0,40,-105,-55,0,         0,0,-40,105,55,0,         0,-75,-10,      0,75,10,          0,0,   0,0],
        [0,0,40,-125,-75,0,         0,0,-40,125,75,0,         0,-75,-10,      0,75,10,          0,0,   0,0],
        [0,0,40,-145,-95,0,         0,0,-40,145,95,0,         0,-75,-10,      0,75,10,          0,0,   0,0],
        # [0,0,40,-145,-95,0,         0,0,-40,145,95,0,         0,-75,-10,      0,75,10,          0,0,   0,0]
    ]
    # # 右手伸手抓 
    # poseList2 = [
    #     [0,0,40,-145,-90,0,         0,0,-40,145,90,0,         -90,0,-10,        0,75,10,          0,0,    0,0],   #手横着举起来
    #     [0,0,40,-145,-90,0,         0,0,-40,145,90,0,         -90,-71,-10,      0,75,10,          0,0,    0,0],   #手向前
    #     # [0,0,40,-145,-90,0,         0,0,-40,145,90,0,         -30,-85,-10,      0,75,10,          0,0,    0,0],
    #     [0,0,80,-190,-95,0,         0,0,-80,190,95,0,         -30,-71,-10,      0,75,10,          100,0,   0,0],   #手向下,手张开
    #     [0,0,100,-190,-95,0,        0,0,-100,190,95,0,        -30,-71,-10,      0,75,10,          100,0,   0,0],   #谦卑跪下
    #     [0,0,100,-190,-95,0,        0,0,-100,190,95,0,        -30,-71,-10,      0,75,10,          50,0,   0,0],
    #     [0,0,100,-190,-95,0,        0,0,-100,190,95,0,        -30,-71,-10,      0,75,10,          50,0,   0,0]
    # ]
    # # 左手伸手抓
    # poseList4 = [
    #     [0,0,40,-145,-90,0,         0,0,-40,145,90,0,         0,-75,-10,        -90,0,10,          0,0,    0,0],   #手横着举起来
    #     [0,0,40,-145,-90,0,         0,0,-40,145,90,0,         0,-75,-10,        -90,71,10,         0,0,    0,0],   #手向前
    #     # [0,0,40,-145,-90,0,         0,0,-40,145,90,0,          0,-75,-10,      0,75,10,          0,0,    0,0],
    #     [0,0,80,-190,-95,0,         0,0,-80,190,95,0,         0,-75,-10,        -40,71,10,         0,-90,   0,0],   #手向下,手张开
    #     [0,0,100,-190,-95,0,        0,0,-100,190,95,0,        0,-75,-10,        -40,71,10,         0,-90,   0,0],   #谦卑跪下
    #     [0,0,100,-190,-95,0,        0,0,-100,190,95,0,        0,-75,-10,        -40,71,10,         0,-50,   0,0],
    #     [0,0,100,-190,-95,0,        0,0,-100,190,95,0,        0,-75,-10,        -40,71,10,         0,-50,   0,0]
    # ]
    
    # 双手捧
    poseList2 = [
        [0,0,40,-145,-95,0,         0,0,-40,145,95,0,         0,-75,-10,      0,75,10,          0,0,   0,0],
        [0,0,80,-185,-95,0,         0,0,-80,185,95,0,         0,-75,-10,      0,75,10,          0,0,   0,0],
        [0,0,98,-190,-95,0,        0,0,-98,190,95,0,         0,-75,-10,      0,75,10,          0,0,   0,0],
        [0,0,98,-190,-95,0,        0,0,-98,190,95,0,         -43,-75,-10,      -43,75,10,          80,-80,   0,0],
        [0,0,98,-190,-95,0,        0,0,-98,190,95,0,         -43,-107,-10,     -43,107,10,          80,-80,   0,0],
        # [0,0,98,-190,-95,0,        0,0,-98,190,95,0,         -44,-106,-10,     -44,106,10,          0,0,   0,0],
        [0,0,98,-190,-95,0,        0,0,-98,190,95,0,         -43,-107,-15,     -43,107,15,          30,-30,   0,0],
        [0,0,98,-190,-95,0,        0,0,-98,190,95,0,         -43,-107,-15,     -43,107,15,          30,-30,   0,0],
        
    ]
    # 站起来
    poseList3 = [
        [0,0,98,-190,-95,0,        0,0,-98,190,95,0,        -43,-107,-15,     -43,107,15,          30,-30,   0,0],
        [0,0,80,-180,-90,0,         0,0,-80,180,90,0,       -43,-107,-15,     -43,107,15,          30,-30,  0,0],        
        [0,0,40,-145,-90,0,         0,0,-40,145,90,0,       -43,-107,-15,     -43,107,15,          30,-30,   0,0],
        [0,0,30,-80,-40,0,          0,0,-30,80,40,0,        -43,-107,-15,     -43,107,15,        30,-30,   0,0],
        [0,0,30,-60,-30,0,          0,0,-30,60,30,0,        -43,-107,-15,     -43,107,15,         30,-30,   0,0]
    ]


    jointPositionPub = rospy.Publisher('MediumSize/BodyHub/MotoPosition', JointControlPoint, queue_size=500)#动作指令发布器
    tpObject = TrajectoryPlanning(22,10.0)#初始化动作规划器


    #读取图像
    # rospy.Subscriber("/sim/camera/UVC/rgbImage",Image, StateCallback)#get status of walking
    rospy.Subscriber("/sim/camera/UVC/rgbImage/compressed",CompressedImage, StateCallback)#get status of walking
    # rospy.spin()

    while not rospy.is_shutdown():
        walkAstep(2)
        walkAstep(0)
        walkAstep(0)
        walkAstep(0)
        walkAstep(4)
        pickBrik(poseList1)
        pickBrik(poseList2)
        pickBrik(poseList3)
        # pickBrik(poseList6)
        

        #  poseList()