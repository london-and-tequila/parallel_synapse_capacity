import datetime
import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_data_torch(
    nDimension=10,
    nSample=100,
    randomSeed=2,
    device=torch.device("cpu"),
    distribution: Literal["uniform", "gaussian"] = "gaussian",
):
    """
    Create training data for the parallel synapse model trained with gradient descent algorithm

    Inputs:
        nDimension: input dimension
        nSample: number of samples
        randomSeed: random seed
        device: cpu or gpu
        distribution: distribution of the data, either "uniform" or "gaussian"
    Outputs:
        data: torch tensor of shape (nSample, nDimension)
            data is sampled from uniform distribution between 0 and 1
        label: torch tensor of shape (nSample, 1)
            label is -1 for the first half of the samples and 1 for the second half
    """
    torch.manual_seed(randomSeed)
    if distribution == "uniform":
        data = torch.rand(nSample, nDimension, device=device)
    elif distribution == "gaussian":
        data = torch.randn(nSample, nDimension, device=device)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    label = torch.ones((nSample, 1), device=device).ravel()
    label[: int(nSample / 2)] = -1
    return data, label


def add_noise(data, label, noise_type="uniform", noise_size=0.1, noise_repeat=5):
    """
    Add noise to the data,
    if uniform noise, noise in U(-noise_size, noise_size)
    if gaussian noise, noise in N(0, noise_size)
    """
    noisy_data_all = []
    label_all = []
    for i in range(noise_repeat):
        if noise_type == "uniform":
            noise = torch.rand_like(data) * noise_size
            noise = noise - 0.5 * noise_size

        elif noise_type == "gaussian":
            noise = torch.randn_like(data) * noise_size
        else:
            raise ValueError("Invalid noise type")
        # Ensure noise is within [0, 1] range by reflecting values outside this range
        noisy_data = data + noise
        noisy_data = torch.where(noisy_data < 0, -noisy_data, noisy_data)
        noisy_data = torch.where(noisy_data > 1, 2 - noisy_data, noisy_data)
        noisy_data_all.append(noisy_data)
        label_all.append(label)
    noisy_data_all = torch.cat(noisy_data_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    return noisy_data_all, label_all


def load_model(path):
    """
    Loading model from a pickle file

    Inputs:
        path: path to the model file
    Outputs:
        model: torch model
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(obj, file):
    """
    Saving model to a pickle file
    """
    with open(file, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def hinge_loss(actv, label, theta, margin):
    """
    Compute hinge loss from the activation of the neuron and the label of the samples

    Inputs:
        actv: torch tensor of shape (nSample, 1)
            activation of the parallel synapses
        label: torch tensor of shape (nSample, 1)
            label of the samples
        theta: torch tensor of shape (1, 1)
            threshold
        margin: float
            margin
    Outputs:
        loss: torch tensor of shape (1, 1)
            hinge loss
    """
    return (
        torch.maximum(torch.zeros_like(actv), margin - (actv - theta) * label)
    ).sum()


def custom_pdf(x, a=0.0, b=0.5):
    """
    cdf of a custom distribution, more threshold towards 0 or 1
    """
    return b * np.abs(x - 0.5) + a


def inverse_cdf(p):
    """
    inverse cdf of a custom distribution
    """
    x = np.linspace(0, 1, 1000)
    cdf = np.cumsum(custom_pdf(x)) / np.sum(custom_pdf(x))
    return np.interp(p, cdf, x)
    # np.interp(p, cdf, x) is the inverse of cdf(p)


def plot_trial(trial, model, path, repeat, t):
    """
    plot the training history, parameter distributions of a trial
    """
    accList = [acc.cpu() for acc in trial.acc_history]
    lossList = [loss.cpu() for loss in trial.loss_history]
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(2, 3, 1)
    plt.plot(trial.time, accList)
    plt.title(
        "acc. ={:.6f}, ".format(trial.acc.detach().cpu())
        + str(repeat)
        + " * "
        + str(trial.Nepoch)
        + " epoch"
    )
    plt.xlabel("time cost (s)")
    plt.grid()
    plt.subplot(2, 3, 2)
    plt.plot(trial.time, lossList)
    plt.title("loss\n " + "time: " + str(datetime.timedelta(seconds=t)))
    plt.xlabel("time cost (s)")
    plt.grid()
    plt.subplot(2, 3, 4)
    plt.hist(model.thres.detach().cpu().numpy().ravel(), bins=30)
    plt.title("thres hist")
    plt.subplot(2, 3, 5)
    plt.hist(model.slope.detach().cpu().numpy().ravel(), bins=30)
    plt.title("slope hist")
    plt.subplot(2, 3, 6)
    plt.hist(model.ampli.detach().cpu().numpy().ravel(), bins=30)
    plt.title(
        "ampli hist, {:.3f} of ampli < {:.3f}".format(
            (model.ampli.detach().cpu().numpy().ravel() < trial.minAmpli).mean(),
            trial.minAmpli,
        )
    )
    plt.tight_layout()
    plt.savefig(path + ".png")
    plt.close(fig)
    # plt.show()
