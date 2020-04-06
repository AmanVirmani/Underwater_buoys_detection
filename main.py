from GMM_EM import GMM_EM, load_data
import cv2
import numpy as np
import argparse
import colors
from scipy.stats import multivariate_normal


def get_trained_gmm():
    green_data = load_data(args["train"])
    yellow_data = load_data(args["train"])
    orange_data = load_data(args["train"])

    green_gmm = GMM_EM(green_data, 3, max_itr=1000)
    yellow_gmm = GMM_EM(yellow_data, 3, max_itr=1000)
    orange_gmm = GMM_EM(orange_data, 3, max_itr=1000)

    return green_gmm.train(), yellow_gmm.train(), orange_gmm.train()


def saveVideo(images, output='./output.avi'):
    h, w = images[0].shape[:2]
    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))
    for img in images:
        out.write(img)
    out.release()


def detect_buoys(gmms, video):
    cap = cv2.VideoCapture(video)
    images = []
    if not cap.isOpened():
        print("error reading video file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output = segment_buoys(gmms, frame)
            images.append(output)
            cv2.imshow('out', output)
            cv2.waitKey(0)
        else:
            break
    saveVideo(images)

def segment_buoys(gmms, frame):
    output = frame.copy()
    color = [colors.green, colors.yellow, colors.orange]
    for i, gmm in enumerate(gmms):
        test = frame.reshape((np.prod(frame.shape[0:-1]), frame.shape[-1]))
        l, ch = test.shape
        prob = np.zeros((l, len(gmm[0])))
        likelihood = np.zeros((l, 1))

        for j in range(len(gmm[0])):
            prob[:, j] = gmm[2][j]*multivariate_normal.pdf(test/255, gmm[0][j], gmm[1][j])

        likelihood = prob.sum(1)

        probabilities = np.reshape(likelihood, frame.shape[:2])
        probabilities = probabilities*255
        probabilities[probabilities < 200] = 0
        circles = cv2.HoughCircles(probabilities.astype(np.uint8), cv2.HOUGH_GRADIENT, 1.5, 90, param1=150, param2=35,
                                   minRadius=0, maxRadius=40)
        if circles is None:
            print("no buoy detected")
            continue
        for blob in circles[0, :]:
            output = cv2.circle(output, (blob[0], blob[1]), blob[2], color[i], 2)
    return output


def main(args):
    gmms = get_trained_gmm()
    detect_buoys(gmms, args["test"])


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-train", "--train", required=False, help="Input training images", default='./data/train/yellow', type=str)
    ap.add_argument("-test", "--test", required=False, help="Test video", default='./data/detectbuoy.avi', type=str)
    args = vars(ap.parse_args())

    main(args)