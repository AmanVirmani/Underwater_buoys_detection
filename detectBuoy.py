import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture("./data/detectbuoy.avi")
    if not cap.isOpened():
        print("Error opening video stream or file")

    roi = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (3, 3))

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 90, param1=150, param2=35, minRadius=0, maxRadius=40)

            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                for (x, y, r) in circles:
                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(output, (x-5, y-5), (x+5, y+5), (0, 128, 255), -1)
                    roi.append(frame[y-r:y+r, x-r:x+r, :])
                    #cv2.imshow('roi', roi[-1])
                    #cv2.waitKey(0)
            #cv2.imshow("output", np.hstack([frame, output]))
            #cv2.waitKey(0)
            # break
        else:
            break
    for d, img in enumerate(roi):
        cv2.imwrite('./data/train/img_{:05d}.jpg'.format(d), img)


if __name__ == '__main__':
    main()
