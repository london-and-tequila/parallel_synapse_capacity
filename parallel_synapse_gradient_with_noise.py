import argparse
import os
import time

import numpy as np
import torch

from parallel_synapse_gradient import ACCURACY_THRESHOLD, ParallelSyn, TrainParallelSyn
from utils_parallel_syn_gradient import (
    add_noise,
    generate_data_torch,
    inverse_cdf,
    load_model,
    plot_trial,
    save_model,
)


class TrainParallelSynWithNoise(TrainParallelSyn):
    """
    Apart from the regular training, we have three different testing accuracies:
    - accuracy: the accuracy of the model on the original data, without noise.
    - accuracy_noisy_train: the accuracy of the model on the noisy data, with noise, different from the training data.
    - accuracy_noisy_test: the accuracy of the model on the noisy data, with noise, same as the training data.
    """

    def __init__(self, train_params):
        super().__init__(train_params)

    def train(self, model: ParallelSyn, data: dict, t1: float):
        """

        Args:
            model (ParallelSyn)
            data (dict): {
                "input
            }
            t1 (float): _description_
        """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("N", type=int, help="N")
    parser.add_argument("M", type=int, help="M")
    parser.add_argument("P", type=int, help="P")
    parser.add_argument("seed", type=int, help="seed")
    parser.add_argument("noise_size", type=float, help="noise size")
    parser.add_argument("noise_repeat", type=int, help="noise repeat")

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
        "noise_size": args.noise_size,
        "noise_repeat": args.noise_repeat,
    }

    path = ""
    folder = "./N_" + str(model_params["N"]) + "_noise"
    if model_params["distribution"] == "gaussian":
        folder += "_gaussian"
    print(folder)
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

    # create folder to save the model and load the model if it exists
    # TODO: modify below for dict input
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

        # original data
        inputX, label = generate_data_torch(
            nDimension=model_params["N"],
            nSample=train_params["P"],
            randomSeed=model_params["seed"],
            device=model_params["device"],
            distribution=model_params["distribution"],
        )
        # noisy training data
        inputX_noisy_train, label_noisy_train = add_noise(
            inputX, label, noise_size=args.noise_size, noise_repeat=args.noise_repeat
        )
        # noisy testing data
        inputX_noisy_test, label_noisy_test = add_noise(
            inputX, label, noise_size=args.noise_size, noise_repeat=args.noise_repeat
        )

        path = ""
        path += (
            "N_"
            + str(model_params["N"])
            + "_M_"
            + str(model_params["M"])
            + "_P_"
            + str(train_params["P"])
            + "_seed_"
            + str(model_params["seed"])
            + "_noise_size_"
            + str(args.noise_size)
            + "_noise_repeat_"
            + str(args.noise_repeat)
        )

        # saving original data
        data_ = torch.hstack((inputX.cpu(), label.reshape(-1, 1).cpu()))
        save_model(data_, folder + "/" + path + "_data")

        # saving noisy training data
        data_ = torch.hstack(
            (inputX_noisy_train.cpu(), label_noisy_train.reshape(-1, 1).cpu())
        )
        save_model(data_, folder + "/" + path + "_noisy_train_data")

        # saving noisy testing data
        data_ = torch.hstack(
            (inputX_noisy_test.cpu(), label_noisy_test.reshape(-1, 1).cpu())
        )
        save_model(data_, folder + "/" + path + "_noisy_test_data")

        model = ParallelSyn(model_params)
        model.to(model_params["device"])

    trial = TrainParallelSynWithNoise(train_params)
    t1 = time.time()
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
        plot_trial(
            trial,
            model,
            folder + "_png" + "/" + path,
            repeat,
            time.time() - t1,
        )
        torch.save(model.state_dict(), folder + "/" + path)
        save_model(trial, folder + "/" + path + "_trial")
