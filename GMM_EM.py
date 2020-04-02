import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix


def init_params(X, k):
    weights = np.ones((k)) / k
    mean = np.random.choice(X.flatten(), (k, X.shape[-1]))
    cov = []
    for i in range(k):
      cov.append(make_spd_matrix(X.shape[-1]))
    cov = np.array(cov)
    return weights, mean, cov


def pdf(data, mean: float, variance: float):
    # A normal continuous random variable.
    s1 = 1/(np.sqrt(2*np.pi*variance))
    s2 = np.exp(-(np.square(data - mean)/(2*variance)))
    return s1 * s2


def GMM_EM(data, k=3):
    bound = 0.001
    max_itr = 1000
    eps = 1e-8

    weights, means, cov = init_params(data, k)

    for step in range(125):
        for img in data:
            likelihood = []
            # Expectation step
            for j in range(k):
              likelihood.append(multivariate_normal.pdf(x=img, mean=means[j], cov=cov[j]))
            likelihood = np.array(likelihood)
            #assert likelihood.shape == (k, img.shape)

            b = []
            # Maximization step
            for j in range(k):
                # use the current values for the parameters to evaluate the posterior
                # probabilities of the data to have been generanted by each gaussian
                b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))

                h, w = img.shape[:2]
                # updage mean and variance
                means[j] = np.sum(b[j].reshape(w*h,1) * img.flatten(), axis=0) / (np.sum(b[j]+eps))
                cov[j] = np.dot((b[j].reshape(w*h,1) * (img - means[j])).T, (img - means[j])) / (np.sum(b[j])+eps)

                # update the weights
                weights[j] = np.mean(b[j])

                assert cov.shape == (k, img.shape[2], img.shape[2])
                assert means.shape == (k, img.shape[2])
            
            
            # old
#            likelihood = []
#            for j in range(k):
#                likelihood.append(pdf(line, mean[j], np.sqrt(var[j])))
#            likelihood = np.array(likelihood)
#            b = []
#
#            for j in range(k):
#                b.append((likelihood[j]*weights[j])/(np.sum([likelihood[i]*weights[i] for i in range(k)],axis=0)+eps))
#
#                mean[j] = np.sum(b[j]*line) / (np.sum(b[j]+eps))
#                var[j] = np.sum(b[j] * np.square(line - mean[j])) / (np.sum(b[j]+eps))
#                weights[j] = np.mean(b[j])
#
    np.save('params.npy', [means, cov, weights])
    return means, cov, weights

def load_data():
    data = []
    dirpath = "data/train/green/"
    for filename in os.listdir(dirpath):
        img = cv2.imread(os.path.join(dirpath,filename))
        img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_LINEAR)
        img = img[6:34, 6:34]

        # # plotting histogram
        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # plt.plot(hist)
        # plt.show()


        data.append(img)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        #print("image shape is: ", end=" ")
        #print(img.shape)
    return np.array(data)

def predict(video, mean, var, weights):
    cap = cv2.VideoCapture(video)
    images = []
    if not cap.isOpened():
        print("error reading video file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            test = frame.copy()
            h, w, ch = test.shape#[:2]
            #green = test[:, :, 1]
            k = len(mean)
            prob = np.zeros((w*h*ch, k))
            likelihood = np.zeros((w*h*ch, 1))

            for j in range(k):
                prob[:, j] = weights[j]*pdf(test.flatten(), mean[j], var[j])

            likelihood = prob.sum(1)

            probabilities = np.reshape(likelihood, (h, w, ch))
            probabilities[probabilities < np.max(probabilities)/2] = 0
            probabilities[probabilities > np.max(probabilities)/2] = 255
            output = np.zeros_like(test)
            output = probabilities
            final = np.hstack((test, output))
            cv2.imshow('prob', output)
            cv2.waitKey(0)
        else:
            break



def main():
    data = load_data()
    print(data.shape)
    means, cov, weights = GMM_EM(data,3)
    print(means)
    print(cov)
    print(weights)
    #predict('./data/detectbuoy.avi', mean, var, weights)


if __name__ == '__main__':
   main()
