import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import argparse
import colors

class GMM_EM:

    def __init__(self, data, clusters, color="test",  max_itr=400, eps=1e-8):
        self.train_data = data
        self.clusters = clusters
        self.color = self.color2pixel(color)
        self.max_itr = max_itr
        self.eps = eps
        self.weights = np.ones(self.clusters) / self.clusters
        self.means = np.random.choice(data.flatten(), (self.clusters, data.shape[-1]))
        self.cov = np.array([make_spd_matrix(data.shape[-1]) for i in range(self.clusters)])

    def color2pixel(self, color):
        if color=="yellow":
            return colors.yellow
        elif color=="orange":
            return colors.orange
        elif color=="green":
            return colors.green
        else:
            print("color not recognized")
            return [0, 0, 0]

    def train(self):
        mle = []
        prev_mle = 0
        for step in range(self.max_itr):
            likelihood = []
            # Expectation step
            for j in range(self.clusters):
                likelihood.append(multivariate_normal.pdf(x=self.train_data, mean=self.means[j], cov=self.cov[j]))
            likelihood = np.array(likelihood)
            assert likelihood.shape == (self.clusters, len(self.train_data))

            b = []
            # Maximization step
            for j in range(self.clusters):
                # use the current values for the parameters to evaluate the posterior
                # probabilities of the (self.train_data to have been generanted by each gaussian
                b.append((likelihood[j] * self.weights[j]) / (np.sum([likelihood[i] * self.weights[i] for i in range(self.clusters)], axis=0)+self.eps))

                # update mean and variance
                self.means[j] = np.sum(b[j].reshape(len(self.train_data), 1) * self.train_data, axis=0) / (np.sum(b[j]+self.eps))
                self.cov[j] = np.dot((b[j].reshape(len(self.train_data), 1) * (self.train_data - self.means[j])).T, (self.train_data - self.means[j])) / (np.sum(b[j]+self.eps))

                # update the (self.weights
                self.weights[j] = np.mean(b[j])

                assert self.cov.shape == (self.clusters, self.train_data.shape[-1], self.train_data.shape[-1])
                assert self.means.shape == (self.clusters, self.train_data.shape[-1])
            mle.append(np.log(np.sum(np.ravel(b))))
            if np.abs(mle[-1] - prev_mle) < self.eps:
                print("GMM converged")
                break
            prev_mle = mle[-1]

        plt.plot(mle)
        plt.show()
        #np.save('params.npy', [self.means, self.cov, (self.weights])
        return self.means, self.cov, self.weights

    def segment_image(self, img):
        test = img.reshape((np.prod(img.shape[0:-1]), img.shape[-1]))
        l, ch = test.shape
        prob = np.zeros((l, self.clusters))
        likelihood = np.zeros((l, 1))

        for j in range(self.clusters):
            prob[:, j] = self.weights[j]*multivariate_normal.pdf(test/255, self.means[j], self.cov[j])

        likelihood = prob.sum(1)

        probabilities = np.reshape(likelihood, img.shape[:2])
        probabilities = probabilities*255
        probabilities[probabilities < 200] = 0
        circles = cv2.HoughCircles(probabilities.astype(np.uint8), cv2.HOUGH_GRADIENT, 1.5, 90, param1=150, param2=35,
                                   minRadius=0, maxRadius=40)
        if circles is None:
            print("no buoy detected")
            return
        for circle in circles[0, :]:
            output = cv2.circle(img.copy(), (circle[0], circle[1]), circle[2], self.color, 2)
        cv2.imshow('input', img)
        cv2.imshow('prob', output)
        cv2.waitKey(10)
        return circles[0, :]

    def test(self, test_dir):
        for img_path in os.listdir(test_dir):
            img = cv2.imread(test_dir+img_path)
            c = self.segment_image(img)

    def predict(self, video):
        cap = cv2.VideoCapture(video)
        images = []
        if not cap.isOpened():
            print("error reading video file")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                c = self.segment_image(frame)
            else:
                break


def detect_circle(gray_img):
    circles = cv2.HoughCircles(gray_img.astype(np.uint8), cv2.HOUGH_GRADIENT, 1.5, 90, param1=150, param2=35, minRadius=0, maxRadius=40)
    out = np.zeros(gray_img.shape)
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for circle in circles:
            out = cv2.circle(out, tuple(circle[0:2]), circle[2], [255], -1)
    return out


def load_data(dirpath):
    data = []
    for filename in os.listdir(dirpath):
        img = cv2.imread(os.path.join(dirpath,filename))
        img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_LINEAR)
        img = img[6:34, 6:34]
        data.append(img)
    data = np.array(data, dtype= np.float64)
    data = data/255
    data = data.reshape((np.prod(data.shape[0:-1]), data.shape[-1]))
    return data


def plot_hist(images):
    for img in images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.show()


def main(args):
    data = load_data(args["train"])
    gmm = GMM_EM(data, args["clusters"], max_itr=1000)
    means, cov, weights = gmm.train()
    #gmm.test('./data/test/yellow/')
    gmm.predict(args["test"])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--clusters", required=False, help="No. of clusters", default=3, type=int)
    ap.add_argument("-train", "--train", required=False, help="Input training images", default='./data/train/green', type=str)
    ap.add_argument("-test", "--test", required=False, help="Test video", default='./data/detectbuoy.avi', type=str)
    args = vars(ap.parse_args())

    main(args)
