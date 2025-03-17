import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

MAX_EPOCHS_FACTOR = 500
DISTRIBUTION = "uniform"


# Function to generate random patterns and labels
def generate_random_data(num_samples, num_features, distribution=DISTRIBUTION):
    # X = np.random.randint(0, 2, (num_samples, num_features))  # Random binary patterns
    if distribution == "uniform":
        X = np.random.rand(num_samples, num_features)
    elif distribution == "gaussian":
        X = np.random.randn(num_samples, num_features)
    y = np.random.choice([-1, 1], num_samples)  # Random binary labels (-1 or 1)
    return X, y


# Perceptron model
class Perceptron:
    def __init__(self, num_features, learning_rate=0.00005, max_epochs=1000):
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_epochs = num_features * MAX_EPOCHS_FACTOR  # max_epochs

    def predict(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else -1

    def fit(self, X, y):
        for epoch in range(self.max_epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                if target != prediction:
                    self.weights += self.learning_rate * xi * target
                    # sign constrained
                    self.weights[self.weights < 0] = 0
                    self.bias += self.learning_rate * target
                    errors += 1
            # Stop if no errors in epoch (converged)
            if errors == 0:
                # print(f"Converged after {epoch + 1} epochs")
                break
        else:
            # print("Reached maximum epochs without full convergence")
            pass

    def score(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        return accuracy


class TrainPerceptron:
    def __init__(self, num_samples, num_features):
        self.num_samples = num_samples
        self.num_features = num_features
        self.perceptron = Perceptron(
            num_features=num_features, learning_rate=0.1, max_epochs=100
        )

    def train(self):
        X, y = generate_random_data(self.num_samples, self.num_features)
        self.perceptron.fit(X, y)
        return self.perceptron.score(X, y)


def single_trial(num_samples, num_features):
    train_perceptron = TrainPerceptron(
        num_samples=num_samples, num_features=num_features
    )
    accuracy = train_perceptron.train()
    return accuracy == 1


def get_success_rate(num_samples=20, num_features=10, repeat=15):
    # Run trials in parallel using joblib

    results = Parallel(n_jobs=15)(
        delayed(single_trial)(num_samples, num_features) for _ in range(repeat)
    )

    results = np.array(results)
    return (np.mean(results), np.std(results) / np.sqrt(repeat), results.tolist())


# plot capacity


success_rate = defaultdict(list)
experi = {}
# Create color map
cmap = plt.cm.viridis  # or any other colormap
N_values = [1000]
colors = {N: cmap(i / len(N_values)) for i, N in enumerate(N_values)}
capacity = []
for N in N_values:
    experi[N] = {}
    print("------------------------")

    if N == 10:
        min_N, max_N, step = 8, 15, 1

    elif N < 200:
        min_N, max_N, step = int(N * 0.8), int(N * 1.2) + 1, int(N * 0.05)
    elif N < 500:
        min_N, max_N, step = int(N * 0.8), int(N * 1.1) + 1, int(N * 0.05)
    else:
        min_N, max_N, step = int(N * 0.8), int(N * 1.1) + 1, int(N * 0.05)
    for P in range(min_N, max_N, step):
        ave_success_rate, std_success_rate, exp = get_success_rate(
            num_samples=P, num_features=N
        )
        experi[N][P] = exp
        print("N = {}, P = {}, success rate = {}".format(N, P, ave_success_rate))

        success_rate[N].append((P, ave_success_rate, std_success_rate))
    with open(
        "perceptron_sign_constrained_experi_max_iter_factor_{}_{}_N_1000.pkl".format(
            MAX_EPOCHS_FACTOR, DISTRIBUTION
        ),
        "wb",
    ) as f:
        pickle.dump(experi, f)

    sample = 100
    psedoC = []
    for i in range(sample):
        psedoX = []
        psedoY = []

        for P in experi[N]:
            psedoX += [P / N] * len(experi[N][P])
            psedoY += random.choices(experi[N][P], k=len(experi[N][P]))
        if min(psedoY) == max(psedoY):
            continue
        clf = LogisticRegression().fit(
            np.array(psedoX).reshape(-1, 1), np.array(psedoY)
        )
        newX = np.linspace(min(experi[N].keys()) / N, max(experi[N].keys()) / N, 20)
        if np.abs(-clf.intercept_[0] / clf.coef_[0][0]) > 20:
            continue
        psedoC.append(-clf.intercept_[0] / clf.coef_[0][0])

        probs = clf.predict_proba(newX.reshape(-1, 1))

        # plt.plot(
        #     newX,
        #     probs[:, 1],
        #     alpha=0.15,
        #     linewidth=0.5,
        #     color=colors[N],
        # )

    capacity.append([N, np.nanmean(np.array(psedoC)), np.nanstd(np.array(psedoC))])

    plt.errorbar(
        np.array(list(experi[N].keys())) / N,
        [np.mean(np.array(experi[N][P])) for P in experi[N]],
        yerr=[
            np.std(np.array(experi[N][P])) / np.sqrt(len(experi[N][P]))
            for P in experi[N]
        ],
        label="N=" + str(N),
        linewidth=1,
        color=colors[N],
    )
    print(capacity)
    with open(
        "perceptron_sign_constrained_capacity_{}_N_1000.pkl".format(DISTRIBUTION), "wb"
    ) as f:
        pickle.dump(capacity, f)

    # plt.errorbar(
    #     [np.array(psedoC).mean()],
    #     [0.5],
    #     xerr=np.array(psedoC).std(),
    #     fmt="o",
    #     linewidth=2,
    #     capsize=2,
    #     color=colors[N],
    # )
print(success_rate)
