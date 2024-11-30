import argparse
import os
import time
from queue import deque

import numpy as np
import torch

from utils_parallel_syn_gradient import *

ACCURACY_THRESHOLD = 0.999999


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ParallelSyn(torch.nn.Module):
    """
    Self-defined parallel synapse model class

    Attributes:
                N: int
                        input dimension
                M: int
                        number of parallel synapses
                seed: int
                        random seed
                device: torch.device
                        cpu or gpu
                ampli: torch.nn.Parameter of shape (N, M)
                        amplitude of the parallel synapses, initialized as random numbers between 0 and 1
                slope: torch.nn.Parameter of shape (N, M)
                        slope of the parallel synapses, initialized as random numbers between 0 and 500
                thres: torch.nn.Parameter of shape (N, M)
                        threshold of the parallel synapses, initialized as random numbers between 0 and 1
                theta: torch.nn.Parameter of shape (1)
                        global threshold, initialized as a random number between 0 and 1
                actv: torch tensor of shape (nSample, 1)
                        activation of the neuron, computed by the forward function
    """

    def __init__(self, params):
        super().__init__()
        for k in params:
            setattr(self, k, params[k])
        torch.manual_seed(self.seed)
        self.ampli = torch.nn.Parameter(
            torch.rand(self.N, self.M, device=params["device"])
        )
        self.slope = torch.nn.Parameter(
            torch.rand(self.N, self.M, device=params["device"]) * 500
        )
        self.thres = torch.nn.Parameter(
            torch.rand(self.N, self.M, device=params["device"])
        )
        self.theta = torch.nn.Parameter(torch.rand(1, device=params["device"]))
        self.shuffle_count = torch.nn.Parameter(
            torch.zeros((self.N, self.M), device=params["device"])
        )

    def forward(self, data):
        """
        Compute the activation of the neuron given the input data
        Inputs:
                data: torch tensor of shape (nSample, N)
                        input data
        Outputs:
                actv: torch tensor of shape (nSample, 1)
        """
        self.actv = data.unsqueeze(2) - self.thres.unsqueeze(0)
        self.actv = self.actv * self.slope.unsqueeze(0)
        self.actv = torch.tanh(self.actv)
        self.actv = self.ampli.pow(2).unsqueeze(0) * self.actv
        self.actv = self.actv.mean(dim=(1, 2))
        return self.actv


class TrainParallelSyn:
    """
    Self-defined training class for the parallel synapse model

    Attributes:
                margin: float
                        margin of the hinge loss
                threslr: float
                        learning rate for the threshold
                adamlr: float
                        learning rate for the other parameters
                minAmpli: float
                        minimum amplitude of the parallel synapses,
                        if the amplitude is smaller than this value, it will be set to this value
                Nepoch: int
                        number of epochs to train on all the data
                P: int
                        number of samples
                maxRecord: int
                        maximum number of records
                downSample: int
                        downsample rate, only record the loss and accuracy every downSample epochs
                thresPool: torch tensor of shape (NthresPool, 1)
                        threshold pool, used to reset the threshold of the parallel synapses when the amplitude is too small
                NthresPool: int
                        number of threshold pool
                loss: torch tensor of shape (1, 1)
                        loss
                acc: torch tensor of shape (1, 1)
                        accuracy
                optim: torch optimizer
                        optimizer, using Adam
                loss_history: deque of length maxRecord
                        loss history
                acc_history: deque of length maxRecord
                        accuracy history
                time: deque of length maxRecord
                        epoch history
                distribution: str
                        distribution of the data, either "uniform" or "gaussian"
    """

    def __init__(self, params):
        for k in params:
            setattr(self, k, params[k])
        self.loss_history = deque()
        self.acc_history = deque()
        self.time = deque()

    def lossFunc(self, model, label):
        self.loss = hinge_loss(model.actv, label, model.theta, self.margin)

    def accu(self, model, label):
        """
        Compute accuracy from the activation of the neuron and the label of the samples
        """
        n_samples = label.shape[0]
        self.acc = (torch.sign(model.actv - model.theta) == label).sum() / n_samples

    def train(self, model, label, inputX, t1):
        # set up optimizer

        self.optim = torch.optim.Adam(
            [
                {"params": model.ampli},
                {"params": model.slope},
                {"params": model.theta},
                {"params": model.thres, "lr": self.threslr},
            ],
            lr=self.adamlr,
        )

        # set up threshold pool, a pool of random thresholds, sample more thresholds from closer to 0 or 1
        if self.distribution == "uniform":
            self.thresPool = torch.tensor(
                inverse_cdf(np.random.uniform(size=(self.NthresPool, 1))),
                device=model.device,
                dtype=torch.float32,
            ).float()
        elif self.distribution == "gaussian":
            self.thresPool = torch.tensor(
                np.random.normal(size=(self.NthresPool, 1)),
                device=model.device,
                dtype=torch.float32,
            ).float()

        for k in range(self.Nepoch):
            if self.shuffle:
                self.shuffle_invalid(model)

            with torch.no_grad():
                # clamp the slope to be non-negative
                model.slope.clamp_min_(0)

            model.forward(inputX)
            self.lossFunc(model, label)
            self.loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            model.forward(inputX)
            self.accu(model, label)

            # record the loss and accuracy every downSample epochs
            if (k % self.downSample) == 5:
                if len(self.acc_history) > self.maxRecord * self.downSample:
                    self.acc_history.popleft()
                    self.loss_history.popleft()
                    self.time.popleft()
                self.acc_history.append(self.acc.detach())
                self.loss_history.append(self.loss.detach())
                self.time.append(time.time() - t1)

            if self.acc > ACCURACY_THRESHOLD:
                print("accuracy reached 1")
                break

    def shuffle_invalid(self, model):
        """
        For those parallel synapses with amplitude smaller than minAmpli,
        reset the threshold to a random threshold from the threshold pool,
        and set the amplitude to minAmpli
        """
        with torch.no_grad():
            mask = model.ampli < self.minAmpli
            # only shuffle those synapses with shuffle_count less than shuffle_limit
            if self.shuffle_limit != -1:
                mask = mask & (model.shuffle_count < self.shuffle_limit)

                model.shuffle_count[mask] += 1
            model.thres[mask] = self.thresPool[
                torch.randint(self.NthresPool, (mask.sum(),))
            ].ravel()
            model.ampli[mask] = self.minAmpli


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("N", type=int, help="N")
    parser.add_argument("M", type=int, help="M")
    parser.add_argument("P", type=int, help="P")
    parser.add_argument("seed", type=int, help="seed")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="repeat")
    parser.add_argument(
        "--shuffle_limit", type=int, default=10000, help="shuffle limit"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model_params = {
        "N": args.N,  # input dimension
        "M": args.M,  # parallel synapse number
        "seed": args.seed,
        "device": device,
        "distribution": "uniform",  # affects initialization of threshold
    }

    train_params = {
        "margin": 0.1,  # only applied when 'loss' is hinge
        "threslr": 1e-6,
        "adamlr": 0.003,
        "minAmpli": 1e-1,
        "Nepoch": 160000,
        "P": args.P,
        "maxRecord": 400,
        "downSample": 100,
        "NthresPool": int(args.P / 2),
        "distribution": "uniform",  # affects threshold resetting
        "shuffle": args.shuffle,
        "shuffle_limit": args.shuffle_limit,
        "N": args.N,
        "M": args.M,
    }

    path = ""
    folder = "./N_" + str(model_params["N"])
    if model_params["distribution"] == "gaussian":
        folder += "_gaussian"
    if args.shuffle and args.shuffle_limit == -1:
        path += (
            "N_"
            + str(model_params["N"])
            + "_M_"
            + str(model_params["M"])
            + "_P_"
            + str(train_params["P"])
            + "_seed_"
            + str(model_params["seed"])
        )
    else:
        path += (
            "N_"
            + str(model_params["N"])
            + "_M_"
            + str(model_params["M"])
            + "_P_"
            + str(train_params["P"])
            + "_seed_"
            + str(model_params["seed"])
            + "_shuffle_"
            + str(train_params["shuffle"])
            + "_shuffle_limit_"
            + str(train_params["shuffle_limit"])
        )

    # create folder to save the model and load the model if it exists
    if os.path.isfile(folder + "/" + path + "_data") and os.path.isfile(
        folder + "/" + path
    ):
        print("loading existing model")
        data_ = load_model(folder + "/" + path + "_data")
        inputX, label = (
            data_[:, :-1].to(model_params["device"]),
            data_[:, -1].to(model_params["device"]),
        )
        model = ParallelSyn(model_params)
        model.to(model_params["device"])
        state_dict = torch.load(
            folder + "/" + path, map_location=model_params["device"]
        )
        model.load_state_dict(state_dict)

    # create new model if it does not exist
    else:
        print("creating new model")
        inputX, label = generate_data_torch(
            nDimension=model_params["N"],
            nSample=train_params["P"],
            randomSeed=model_params["seed"],
            device=model_params["device"],
            distribution=model_params["distribution"],
        )

        data_ = torch.hstack((inputX.cpu(), label.reshape(-1, 1).cpu()))
        save_model(data_, folder + "/" + path + "_data")

        model = ParallelSyn(model_params)
        model.to(model_params["device"])

    trial = TrainParallelSyn(train_params)
    t1 = time.time()
    count = 0
    for repeat in range(800):
        trial.train(model, label, inputX, t1)

        print(f"Repeat: {repeat}, accuracy: {trial.acc}")
        if trial.acc > ACCURACY_THRESHOLD:
            plot_trial(
                trial,
                model,
                folder + "_png" + "/" + path + "_true",
                repeat,
                time.time() - t1,
            )
            torch.save(model.state_dict(), folder + "/" + path)
            save_model(trial, folder + "/" + path + "_trial")
            break
        plot_trial(trial, model, folder + "_png" + "/" + path, repeat, time.time() - t1)
        torch.save(model.state_dict(), folder + "/" + path)
        save_model(trial, folder + "/" + path + "_trial")
