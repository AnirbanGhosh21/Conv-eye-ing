import cv2
import numpy as np
import math
import csv

print("Initiating UI.....")

path = 'Keyboard/'
letter = []
with open(path + 'ui_index.csv', newline='') as f:
    reader = csv.reader(f)
    letter = list(reader)


def detectObj(img, cascade, scale, noOfNegh):
    imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objDet = cascade.detectMultiScale(imgGry, scale, noOfNegh)
    return objDet


def imgCrop(img, x1, y1, w1, h1):
    imgCrp = img[y1:y1+h1, x1:x1+w1]
    return imgCrp


def blob_process(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 42, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Test1", img)
    img = cv2.erode(img, None, iterations=2)
    #cv2.imshow("Test2", img)
    img = cv2.dilate(img, None, iterations=3)
    #cv2.imshow("Test3", img)
    img = cv2.medianBlur(img, 7)
    #cv2.imshow("Test4", img)
    keypoints = detector.detect(img)
    return keypoints


def vectorCalc(origin, pos):
    mag = int(math.sqrt((origin[0]-pos[0])**2 + (origin[1]-pos[1])**2))
    if (origin[0]-pos[0]) == 0:
        if (origin[1]-pos[1]) > 0:
            slope = 270
        else:
            slope = 90
    else:
        slope = math.degrees(math.atan((origin[1]-pos[1])/(origin[0]-pos[0])))
        slope = abs(slope)
        if pos[0] > origin[0]:
            if pos[1] < origin[1]:
                slope += 270
        elif pos[0] < origin[0]:
            if pos[1] < origin[1]:
                slope += 180
            else:
                slope += 90
    return mag, slope


def charPupil(b, m, s, Os, oldC):
    if m > thresholdMotion:
        if b < 6:
            if s > 135 or s < 225:
                c = letter[b][1].replace('.png', '')
                string = path + letter[b][1]
            elif s > 45 or s < 135:
                c = letter[b][2].replace('.png', '')
                string = path + letter[b][2]
            elif s > 315 or s < 45:
                c = letter[b][3].replace('.png', '')
                string = path + letter[b][3]
            elif s > 225 or s < 315:
                c = letter[b][4].replace('.png', '')
                string = path + letter[4]
        else:
            if s > 135 or s < 165:
                c = letter[b][1].replace('.png', '')
                string = path + letter[b][1]
            elif s > 45 or s < 135:
                c = letter[b][2].replace('.png', '')
                string = path + letter[b][2]
            elif s > 345 or s < 45:
                c = letter[b][3].replace('.png', '')
                string = path + letter[b][3]
            elif s > 245 or s < 295:
                c = letter[b][4].replace('.png', '')
                string = path + letter[b][4]
            elif s > 165 or s < 245:
                c = letter[b][5].replace('.png', '')
                string = path + letter[b][5]
            elif s > 295 or s < 345:
                c = letter[b][6].replace('.png', '')
                string = path + letter[b][6]
    else:
        string = Os
        c = oldC
    return c, string


def blockPupil(origin, pos, blk, oldString, oldChar):
    mag, slope = vectorCalc(origin, pos)
    overlaySt = oldString
    if mag < thresholdMotion:
        C = oldChar
        overlaySt = oldString
        blk = 0
    else:
        print("Motion Detected " + str(slope))
        C = '---'
        if blk == 0:
            if slope > 0 and slope < 60:
                if mag > thresholdMotion:
                    overlaySt = path + letter[7][2]
                    blk = 3
            elif slope > 60 and slope < 120:
                if mag > thresholdMotion:
                    overlaySt = path + letter[7][0]
                    blk = 2
            elif slope > 120 and slope < 180:
                if mag > thresholdMotion + 1:
                    overlaySt = path + letter[7][1]
                    blk = 1
            elif slope > 180 and slope < 240:
                if mag > thresholdMotion + 2:
                    overlaySt = path + letter[8][1]
                    blk = 4
            elif slope > 240 and slope < 300:
                if mag > thresholdMotion:
                    overlaySt = path + letter[8][0]
                    blk = 5
            elif slope > 300 and slope < 360:
                if mag > thresholdMotion:
                    overlaySt = path + letter[8][2]
                    blk = 6
        else:
            C, overlaySt = charPupil(blk, mag, slope, oldString, oldChar)

    return C, overlaySt, blk


print("Initiating Pupil Detector.....")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

thresholdMotion = 3

j = 0
i = 0
char = "---"
oldLet = char
block = 0
framePath = 'Frames/frm'

overlayStr = path + letter[0][0]

print("Initiating Clasifier.....")

face_cascade = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
overlay = cv2.imread(overlayStr)

print("Initiating Video Camera.....")

cap = cv2.VideoCapture(0)

while True:
    success, frm = cap.read()
    frm = cv2.flip(frm, 1)
    frm = cv2.resize(frm, (1024, 768), interpolation=cv2.INTER_AREA)
    faces = detectObj(frm, face_cascade, 1.2, 5)
    noOfFaces = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frm, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = imgCrop(frm, x, y, w, h)
        eyes = detectObj(face, eye_cascade, 1.2, 7)
        noOfEyes = len(eyes)
        if noOfEyes/noOfFaces == 2:
            for (ex, ey, ew, eh) in eyes:
                crpH = int(eh/3.5)
                crpW = int(ew/6)
                eye = imgCrop(face, ex+int(crpW/2), ey+crpH, ew-crpW, eh-crpH)
                keypoint = blob_process(eye, detector)
                if len(keypoint) > 0:
                    j += 1
                    if j == 1:
                        origin = [keypoint[0].pt[0], keypoint[0].pt[1]]

                if i == 0 or len(keypoint) == 0:
                    char = oldLet
                    overlaySt = path + letter[0][0]
                    overlay = cv2.imread(overlayStr)
                else:
                    if i % 5 == 0:
                        pupilPos = [keypoint[0].pt[0], keypoint[0].pt[1]]
                        char, overlayStr, block = blockPupil(
                            origin, pupilPos, block, overlayStr, char)
                        overlay = cv2.imread(overlayStr)
                        oldLet = char
                eye = cv2.drawKeypoints(
                    eye, keypoint, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.rectangle(face, (ex+int(crpW/2), ey+crpH),
                              (ex+ew-crpW, ey+eh-crpH), (0, 225, 255), 2)
        i += 1
    added_image = cv2.addWeighted(frm, 0.4, overlay, 1, 0)
    added_image = cv2.putText(added_image, char, (512, 720),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    added_image = cv2.putText(added_image, str(
        10-int((i) % 10)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Video Output", added_image)
    cv2.imwrite(framePath + str(i) + '.png', added_image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
