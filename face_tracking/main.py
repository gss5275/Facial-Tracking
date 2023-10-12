import math

import cv2
import dlib
import cv2 as cv
import numpy as np
from math import hypot

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('video.mp4')




num = 3

while True:

    ret, img_frame = cap.read()

    img_frame = cv2.resize(img_frame, (0, 0), fx=num/10, fy=num/10)
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    #얼굴들 찾기
    dets = detector(img_gray, 1)


    #=======계산
    #눈
    def get_gaze_ratio(eye_points, facial_landmarks, _gray, _frame):
        eye_region = np.array([facial_landmarks[eye_points[0]],
                               facial_landmarks[eye_points[1]],
                               facial_landmarks[eye_points[2]],
                               facial_landmarks[eye_points[3]],
                               facial_landmarks[eye_points[4]],
                               facial_landmarks[eye_points[5]]], np.int32)


        cv2.polylines(_frame, [eye_region], True, (0, 255, 255), 1)

        height, width, _ = _frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 1)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(_gray, _gray, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        padding = 4

        gray_eye = eye[min_y-padding: max_y+padding, min_x-padding: max_x+padding]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        # 눈동자의 흰부분 계산으로 보는 방향 추정
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)


        # left, right side white 가 10 미만이면 눈 감은것으로 인식
        if left_side_white < 5 or right_side_white < 5:
            _gaze_ratio = 1
        else:
            _gaze_ratio = left_side_white / right_side_white


        return _gaze_ratio


    #회전
    def get_head_angle_ratio(head_points, facial_landmarks):
        # 코의 가로선 표시
        nose_left_point = facial_landmarks[head_points[1]]
        nose_right_point = facial_landmarks[head_points[2]]
        nose_center_point = facial_landmarks[head_points[0]]

        # 오른쪽 기준선과 왼쪽 기준선 길이 계산
        nose_line_len1 = hypot(nose_left_point[0] - nose_center_point[0], nose_left_point[1] - nose_center_point[1])
        nose_line_len2 = hypot(nose_right_point[0] - nose_center_point[0], nose_right_point[1] - nose_center_point[1])

        if nose_line_len1 > nose_line_len2:
            _direction_ratio = (nose_line_len1 / nose_line_len2)
        else:
            _direction_ratio = (nose_line_len2 / nose_line_len1)*(-1)

        return _direction_ratio




    #가장 큰 얼굴 필터링
    com_li = []
    for face in dets:
        com_li.append(face.top()-face.bottom())
    try:
        face = dets[com_li.index(max(com_li))]


    #==실제 작동부
        #얼굴에서 68개 점 찾기
        shape = predictor(img_frame, face)

        #점 리스트
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])
        list_points = np.array(list_points)



        #정보 추출
        #회전(정면)
        cv2.line(img_frame, list_points[0],list_points[16], (0, 255, 255), 1)
        degree = (-1)*(math.atan2(list_points[0][0]-list_points[16][0],list_points[0][1]-list_points[16][1]))
        print(degree)


        #회전(수직)---[중심, 좌, 우],왼쪽:-, 오른쪽:+
        direction_ratio_lr = get_head_angle_ratio([30, 31, 35], list_points)



        # 회전(좌우)---위:작아짐, 아래: 커짐
        direction_ratio_ud = get_head_angle_ratio([27, 33, 8], list_points)
        #print(direction_ratio_ud)

        #눈 (왼쪽 커짐, 오른쪽 작아짐)
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], list_points, img_gray, img_frame)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], list_points, img_gray, img_frame)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2
        #print("right_left: ",gaze_ratio)

        #입




















        #점찍기 및 얼굴 사각형
        for i, pt in enumerate(list_points):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        #cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),(0, 0, 255), 3)

    except:
        pass

    img_frame = cv2.resize(img_frame, (0, 0), fx=10/num, fy=10/num)
    cv.imshow('result', img_frame)
    
    
    
    #esc=종료, 번호=필터링
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()


#입력값 : 이미지, 화질, 출력값의 기본값
#출력값 : 회전(3개축), 눈(좌우, 좌우 각각 높이 차), 입(모든 좌표), 코(코 위치), 얼굴 중앙