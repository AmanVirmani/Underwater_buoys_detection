import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import argparse


def init_params(data, k):
    weights = np.ones((k)) / k
    mean = np.random.choice(data.flatten(), (k, data.shape[-1]))
    cov = []
    for i in range(k):
        cov.append(make_spd_matrix(data.shape[-1]))
    cov = np.array(cov)
    return weights, mean, cov


def GMM_EM(data, k=3):
    max_itr = 400
    eps = 1e-8

    weights, means, cov = init_params(data, k)

    for step in range(max_itr):
        likelihood = []
        # Expectation step
        for j in range(k):
            likelihood.append(multivariate_normal.pdf(x=data, mean=means[j], cov=cov[j]))
        likelihood = np.array(likelihood)
        assert likelihood.shape == (k, len(data))

        b = []
        # Maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian
            b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))

            # update mean and variance
            means[j] = np.sum(b[j].reshape(len(data), 1) * data, axis=0) / (np.sum(b[j]+eps))
            cov[j] = np.dot((b[j].reshape(len(data), 1) * (data - means[j])).T, (data - means[j])) / (np.sum(b[j]+eps))

            # update the weights
            weights[j] = np.mean(b[j])

            assert cov.shape == (k, data.shape[-1], data.shape[-1])
            assert means.shape == (k, data.shape[-1])

    #np.save('params.npy', [means, cov, weights])
    return means, cov, weights


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


def predict(video, mean, var, weights):
    cap = cv2.VideoCapture(video)
    images = []
    if not cap.isOpened():
        print("error reading video file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            test = frame.reshape((np.prod(frame.shape[0:-1]), frame.shape[-1]))
            l, ch = test.shape
            k = len(mean)
            prob = np.zeros((l, k))
            likelihood = np.zeros((l, 1))

            for j in range(k):
                prob[:, j] = weights[j]*multivariate_normal.pdf(test/255, mean[j], var[j])

            likelihood = prob.sum(1)

            probabilities = np.reshape(likelihood, frame.shape[:2])
            probabilities = probabilities*255
            probabilities[probabilities < 200] = 0
            output = np.zeros_like(frame)
            output = probabilities
            final = np.hstack((frame[:,:,1], output))
            cv2.imshow('input', frame)
            cv2.imshow('prob', output)
            cv2.waitKey(0)
        else:
            break



def main(args):
    data = load_data(args["train"])
    means, cov, weights = GMM_EM(data, 3)
    predict(args["test"], means, cov, weights)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-train", "--train", required=False, help="Input training images", default='./data/train/yellow', type=str)
    ap.add_argument("-test", "--test", required=False, help="Test video", default='./data/detectbuoy.avi', type=str)
    args = vars(ap.parse_args())

    main(args)
