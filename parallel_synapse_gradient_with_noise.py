import argparse
import os
import time

import torch

from parallel_synapse_gradient import ACCURACY_THRESHOLD, ParallelSyn, TrainParallelSyn
from utils_parallel_syn_gradient import (
    add_noise,
    generate_data_torch,
    load_model,
    plot_trial,
    save_model,
)

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
        inputX, label = add_noise(
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

        data_ = torch.hstack((inputX.cpu(), label.reshape(-1, 1).cpu()))
        save_model(data_, folder + "/" + path + "_data")

        model = ParallelSyn(model_params)
        model.to(model_params["device"])

    trial = TrainParallelSyn(train_params)
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
