import cv2
import numpy
import numpy as np
from ultralytics import YOLO
from sort import *
import math
import cvzone



model=YOLO('../Cars counter/yolov8l.pt')

path="../Cars counter/168811 (1080p).mp4"
vid=cv2.VideoCapture(path)

graphic=cv2.imread("../Cars counter/graphics count the cars.png")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

line1=[665,60,901,60]
line2=[(420,135),(587,135)]
line3=[(383,719),(383,880)]
line4=[(531,897),(532,1034)]
line5=[(1062,1023),(1296,1023)]
line6=[(1330,950),(1540,950)]
line7=[(1476,301),(1476,90)]
line8=[(1300,159),(1300,20)]

count1=[]
count2=[]
count3=[]
count4=[]
count5=[]
count6=[]
count7=[]
count8=[]

def drawlines():

    cv2.line(img, (line1[0],line1[1]), (line1[2],line1[3]) ,(229, 195, 81), 5)
    cv2.line(img, line2[0], line2[1], (229, 195, 81), 5)
    cv2.line(img, line3[0], line3[1], (229, 195, 81), 5)
    cv2.line(img, line4[0], line4[1], (229, 195, 81), 5)
    cv2.line(img, line5[0], line5[1], (229, 195, 81), 5)
    cv2.line(img, line6[0], line6[1], (229, 195, 81), 5)

    #cv2.line(img, line7[0], line7[1], (229, 195, 81), 5)
    #cv2.line(img, line8[0], line8[1], (229, 195, 81), 5)


while True:
    success, img=vid.read()

    result = model(img, stream=True)
    detections = np.empty((0, 5))  # defining empty array of 0,5

    for r in result:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w=x2-x1
            h=y2-y1

            cls=int(box.cls[0])

            clsname=classNames[cls]
            print(classNames[cls])
            conf = math.ceil((box.conf[0] * 100)) / 100
            if clsname=="car" or clsname=="truck" or clsname=="motorbike" or clsname=="bus" and conf>0.3:

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                conf = math.ceil((box.conf[0] * 100)) / 100



                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))



        resulttracker = tracker.update(detections)
        drawlines()

        for result in resulttracker:  # taking co=ordinates values and tracking id from tracker for each person
            x1, y1, x2, y2, Id = result

            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            x2,y2=int(x2),int(y2)
            w=x2-x1
            h=y2-y1
            #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(25, 100, 255))
            #cv2.putText(img,"f'{Id})",(430,90),cv2.FONT_HERSHEY_PLAIN,5,(50,150,255),8)
            cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=1)

            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
            #cv2.circle(img, (x1, y1), 8, (255, 0, 127), cv2.FILLED)

            #cv2.polylines(img, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
            cv2.putText(img, "count the cars", (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)


            if line8[0][0] - 50< cx < line8[1][0] +50 and line8[0][1] < cy < line8[1][1]:
                #if count8.count(Id) == 0:
                count8.append(Id)
                cv2.line(img, (line8[0]), (line8[1]), (0, 255, 0),4)

            if line1[0] < cx < line1[2] and line1[1] - 20 < cy < line1[3] + 20:
                if count1.count(Id) == 0:
                    count1.append(Id)
                    cv2.line(img, (line1[0], line1[1]), (line1[2], line1[3]), (0, 255, 0))

            if line2[0][0] < cx < line2[1][0] and line2[0][1] - 20 < cy < line2[1][1] + 20:
                if count2.count(Id) == 0:
                    count2.append(Id)
                    cv2.line(img, (line2[0]), (line2[1]), (0, 255, 0),4)

            if (line3[0][0] - 30) < cx < (line3[1][0] +30) and (line3[0][1]) < cy < (line3[1][1]):
                if count3.count(Id) == 0:
                    count3.append(Id)
                    cv2.line(img, (line3[0]), (line3[1]), (0, 255, 0),4)

            if (line4[0][0] - 30) < cx < (line4[1][0] +30) and (line4[0][1]) < cy < (line4[1][1]):
                if count4.count(Id) == 0:
                    count4.append(Id)
                    cv2.line(img, (line4[0]), (line4[1]), (0, 255, 0),4)

            if line5[0][0] < cx < line5[1][0] and line5[0][1] - 20 < cy < line5[1][1] + 20:
                if count5.count(Id) == 0:
                    count5.append(Id)
                    cv2.line(img, (line5[0]), (line5[1]), (0, 255, 0),4)

            if line6[0][0] < cx < line6[1][0] and line6[0][1] - 20 < cy < line6[1][1] + 20:
                if count6.count(Id) == 0:
                    count6.append(Id)
                    cv2.line(img, (line6[0]), (line6[1]), (0, 255, 0),4)

            """
            if line7[0][0] - 40 < cx < line7[1][0] +40 and line7[0][1] < cy < line7[1][1]:
                if count7.count(Id) == 0:
                    count7.append(Id)
                    cv2.line(img, (line7[0]), (line7[1]), (0, 255, 0),4)

            if line8[0][0] - 50< cx < line8[1][0] +50 and line8[0][1] < cy < line8[1][1]:
                if count8.count(Id) == 0:
                    count8.append(Id)
                    cv2.line(img, (line8[0]), (line8[1]), (0, 255, 0),4)
            """
        drawlines()




#Putting Text
        cv2.putText(img, f'IN1-> {str(len(count1))}', (line1[0]-100,line1[1]), cv2.FONT_HERSHEY_COMPLEX,2, (0, 255, 0), 3)
        cv2.putText(img, f'OUT1 ^ {str(len(count2))}', line2[0], cv2.FONT_HERSHEY_COMPLEX,2, (50, 50, 255), 3)

        cv2.putText(img, f' {str(len(count3))} <-OUT2', line3[0], cv2.FONT_HERSHEY_COMPLEX,2, (50, 50, 255), 3)
        cv2.putText(img, f'IN2-> {str(len(count4))}', line4[0], cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

        cv2.putText(img, f'OUT3-> {str(len(count5))}', line5[0], cv2.FONT_HERSHEY_COMPLEX,2, (50, 50, 255), 3)
        cv2.putText(img, f'IN3-> {str(len(count6))}', line6[0], cv2.FONT_HERSHEY_COMPLEX,2, (0, 255, 0), 3)
        """"
        cv2.putText(img, f'OUT47-> {str(len(count7))}', line7[0], cv2.FONT_HERSHEY_COMPLEX,2, (50, 50, 255), 3)
        cv2.putText(img, f'{str(len(count8))}<-IN4', line8[0], cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        """

        img[5:5 + 400, 1250:1250 + 600] = graphic
        cv2.imshow("window", img)
        cv2.waitKey(1)

