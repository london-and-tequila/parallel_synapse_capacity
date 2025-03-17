import argparse
import datetime
import os
import time
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from parallel_synapse_gradient import ACCURACY_THRESHOLD, ParallelSyn, TrainParallelSyn
from utils_parallel_syn_gradient import (
    add_noise,
    generate_data_torch,
    inverse_cdf,
    load_model,
    save_model,
)


def plot_trial(acc_history_dict, loss_history, model, path, repeat, t):
    """
    plot the training history, parameter distributions of a trial
    """
    acc_original_list = [acc.cpu() for acc in acc_history_dict["original"]]
    acc_noisy_train_list = [acc.cpu() for acc in acc_history_dict["noisy_train"]]
    acc_noisy_test_list = [acc.cpu() for acc in acc_history_dict["noisy_test"]]
    loss_list = [loss.cpu() for loss in loss_history]
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(2, 3, 1)
    plt.plot(trial.time, acc_original_list, label="original")
    plt.plot(trial.time, acc_noisy_train_list, label="noisy train")
    plt.plot(trial.time, acc_noisy_test_list, label="noisy test")
    plt.legend()
    plt.title(
        "acc. ={:.4f}, ".format(acc_original_list[-1])
        + "acc. noisy train ={:.4f}, \n".format(acc_noisy_train_list[-1])
        + "acc. noisy test ={:.4f}, ".format(acc_noisy_test_list[-1])
        + "repeat "
        + str(repeat)
        + " * "
        + str(trial.Nepoch)
        + " epoch"
    )
    plt.xlabel("time cost (s)")
    plt.grid()
    plt.subplot(2, 3, 2)
    plt.plot(trial.time, loss_list)
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


class TrainParallelSynWithNoise(TrainParallelSyn):
    """
    Apart from the regular training, we have three different testing accuracies:
    - accuracy: the accuracy of the model on the original data, without noise.
    - accuracy_noisy_train: the accuracy of the model on the noisy data, with noise, different from the training data.
    - accuracy_noisy_test: the accuracy of the model on the noisy data, with noise, same as the training data.
    """

    def __init__(self, train_params):
        super().__init__(train_params)
        self.acc_noisy_train_history = deque()
        self.acc_noisy_test_history = deque()

    def accu_all(self, model, actv_dict):
        """
        Compute accuracy of the model on the original data, noisy training data, and noisy testing data

        Args:
            model (_type_): _description_
            actv_dict (_type_): _description_
        Returns:
            dict: accuracy of the model on the original data, noisy training data, and noisy testing data
        """
        acc_dict = defaultdict(float)
        for key, (actv, label) in actv_dict.items():
            n_samples = label.shape[0]
            acc_dict[key] = (torch.sign(actv - model.theta) == label).sum() / n_samples
        return acc_dict

    def train_with_noise(self, model: ParallelSyn, data: dict, t1: float):
        """

        Args:
            model (ParallelSyn)
            data (dict): {
                "inputX": torch.tensor,
                "label": torch.tensor,
                "inputX_noisy_train": torch.tensor,
                "label_noisy_train": torch.tensor,
                "inputX_noisy_test": torch.tensor,
                "label_noisy_test": torch.tensor,
            }
            t1 (float): _description_
        """
        # unpack data
        inputX = data["inputX"]
        label = data["label"]
        inputX_noisy_train = data["inputX_noisy_train"]
        label_noisy_train = data["label_noisy_train"]
        inputX_noisy_test = data["inputX_noisy_test"]
        label_noisy_test = data["label_noisy_test"]

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
        actv_dict = {}
        for k in range(self.Nepoch):
            self.shuffle_invalid(model)

            with torch.no_grad():
                # clamp the slope to be non-negative
                model.slope.clamp_min_(0)

            model.forward(inputX_noisy_train)
            self.lossFunc(model, label_noisy_train)

            self.loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            # test on original data
            actv = model.forward(inputX)

            # test on noisy training data
            actv_noisy_train = model.forward(inputX_noisy_train)

            # test on noisy testing data
            actv_noisy_test = model.forward(inputX_noisy_test)

            actv_dict["original"] = (actv, label)
            actv_dict["noisy_train"] = (actv_noisy_train, label_noisy_train)
            actv_dict["noisy_test"] = (actv_noisy_test, label_noisy_test)

            acc_dict = self.accu_all(model, actv_dict)

            # record the loss and accuracy every downSample epochs
            if (k % self.downSample) == 5:
                if len(self.acc_history) > self.maxRecord * self.downSample:
                    self.acc_history.popleft()
                    self.loss_history.popleft()
                    self.acc_noisy_train_history.popleft()
                    self.acc_noisy_test_history.popleft()
                    self.time.popleft()
                self.acc_history.append(acc_dict["original"])
                self.loss_history.append(self.loss.detach())
                self.acc_noisy_train_history.append(acc_dict["noisy_train"])
                self.acc_noisy_test_history.append(acc_dict["noisy_test"])

                self.time.append(time.time() - t1)

            if (
                acc_dict["original"] > ACCURACY_THRESHOLD
                and acc_dict["noisy_train"] > ACCURACY_THRESHOLD
                and acc_dict["noisy_test"] > ACCURACY_THRESHOLD
            ):
                print("accuracy all reached 1")
                break
        return acc_dict


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
        "shuffle_limit": -1,
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
    if os.path.isfile(folder + "/" + path + "_data") and os.path.isfile(
        folder + "/" + path
    ):
        print("loading existing model")
        data_ = load_model(folder + "/" + path + "_data")
        inputX, label = (
            data_[:, :-1].to(model_params["device"]),
            data_[:, -1].to(model_params["device"]),
        )

        data_ = load_model(folder + "/" + path + "_noisy_train_data")
        inputX_noisy_train, label_noisy_train = (
            data_[:, :-1].to(model_params["device"]),
            data_[:, -1].to(model_params["device"]),
        )

        data_ = load_model(folder + "/" + path + "_noisy_test_data")
        inputX_noisy_test, label_noisy_test = (
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

    data = {
        "inputX": inputX,
        "label": label,
        "inputX_noisy_train": inputX_noisy_train,
        "label_noisy_train": label_noisy_train,
        "inputX_noisy_test": inputX_noisy_test,
        "label_noisy_test": label_noisy_test,
    }

    trial = TrainParallelSynWithNoise(train_params)
    t1 = time.time()
    is_plotted_original = False
    is_plotted_noisy_train = False
    is_plotted_noisy_test = False
    for repeat in range(800):
        acc_dict = trial.train_with_noise(model, data, t1)
        print(
            f"Repeat: {repeat}, acc_original: {acc_dict['original']}, "
            + f"acc_noisy_train: {acc_dict['noisy_train']}, "
            + f"acc_noisy_test: {acc_dict['noisy_test']}"
        )
        acc_history_dict = {
            "original": trial.acc_history,
            "noisy_train": trial.acc_noisy_train_history,
            "noisy_test": trial.acc_noisy_test_history,
        }
        if acc_dict["original"] > ACCURACY_THRESHOLD and not is_plotted_original:
            plot_trial(
                acc_history_dict,
                trial.loss_history,
                model,
                folder + "_png" + "/" + path + "_original_true",
                repeat,
                time.time() - t1,
            )
            torch.save(model.state_dict(), folder + "/" + path)
            save_model(trial, folder + "/" + path + "_trial")
            is_plotted_original = True
        if acc_dict["noisy_train"] > ACCURACY_THRESHOLD and not is_plotted_noisy_train:
            plot_trial(
                acc_history_dict,
                trial.loss_history,
                model,
                folder + "_png" + "/" + path + "_noisy_train_true",
                repeat,
                time.time() - t1,
            )
            torch.save(model.state_dict(), folder + "/" + path)
            save_model(trial, folder + "/" + path + "_trial")
            is_plotted_noisy_train = True
        if acc_dict["noisy_test"] > ACCURACY_THRESHOLD and not is_plotted_noisy_test:
            plot_trial(
                acc_history_dict,
                trial.loss_history,
                model,
                folder + "_png" + "/" + path + "_noisy_test_true",
                repeat,
                time.time() - t1,
            )
            torch.save(model.state_dict(), folder + "/" + path)
            save_model(trial, folder + "/" + path + "_trial")
            is_plotted_noisy_test = True
        plot_trial(
            acc_history_dict,
            trial.loss_history,
            model,
            folder + "_png" + "/" + path,
            repeat,
            time.time() - t1,
        )
        torch.save(model.state_dict(), folder + "/" + path)
        save_model(trial, folder + "/" + path + "_trial")
