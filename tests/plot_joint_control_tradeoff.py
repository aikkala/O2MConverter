import Utils
import os
import pandas as pd
import numpy as np
from tests.envs import EnvFactory
import sys
import matplotlib.pyplot as pp

# Increase font size
import matplotlib
matplotlib.rcParams.update({'font.size': 22})


def get_tradeoff_error(run_folder):

    # Get alpha values (we assume they're the same for each run)
    alpha_folder = os.path.join(run_folder, "optimized_control")
    alphas = np.sort(os.listdir(alpha_folder))

    # Loop through different alpha values and read errors
    tradeoff_error = np.zeros((len(alphas), 2))
    for idx, alpha in enumerate(alphas):

        # Read joint error
        joint_error = pd.read_csv(os.path.join(alpha_folder, alpha, 'joint_error.csv'), delimiter='\n', header=None)

        # Read control error
        control_error = pd.read_csv(os.path.join(alpha_folder, alpha, 'control_error.csv'), delimiter='\n', header=None)

        # Save sum of joint error and mean of control error
        tradeoff_error[idx, 0] = control_error.mean().values[0]
        tradeoff_error[idx, 1] = joint_error.mean().values[0]

    return tradeoff_error


def main(model_names):

    # Plot the tradeoff errors
    fig, ax = pp.subplots(figsize=(18, 12))
    ax.set_ylabel('Average joint error (mean absolute difference)')
    ax.set_xlabel('Average control error (mean absolute difference)')

    # Load data for each model
    #tradeoff_error = {}
    handles = []
    markers = ['o', 's', 'P']
    alpha_idx = {"mobl_arms": 5, "gait10dof18musc": 7, "gait2392": 7}
    for model_idx, model_name in enumerate(model_names):
        env = EnvFactory.get(model_name)
        d = Utils.load_data(env.data_file)

        # Go through each test run
        tradeoff_error = []
        for run_idx in d["test_idxs"]:

            # Get run data
            run_data = d["data"][run_idx]

            # Get control / joint errors for different alpha values
            e = get_tradeoff_error(os.path.join(env.output_folder, run_data["run"]))
            tradeoff_error.append(e)

        # Use only test runs that have all alpha optimizations
        num_alpha = [e.shape[0] for e in tradeoff_error]
        for idx, n in enumerate(reversed(num_alpha)):
            if n < max(num_alpha):
                del tradeoff_error[-(idx+1)]

        # Convert tradeoff_error into a numpy array
        tradeoff_error = np.asarray(tradeoff_error)

        # Plot
        handles.append(ax.scatter(tradeoff_error[:, :, 0].mean(axis=0), tradeoff_error[:, :, 1].mean(axis=0), s=100,
                                  marker=markers[model_idx]))
        ax.scatter(tradeoff_error[:, alpha_idx[model_name], 0].mean(axis=0),
                   tradeoff_error[:, alpha_idx[model_name], 1].mean(axis=0),
                   s=250, marker='o', edgecolors='k', facecolors='none', linewidths=3)

    # Set legend
    pp.legend(handles, model_names)
    pp.xlim(left=0)

    # Save figure
    #ax.set_title('Trade-off between trajectory and muscle excitation accuracy')
    #os.makedirs(os.path.join(env.output_folder, '..', 'figures'), exist_ok=True)
    fig.savefig(os.path.join(env.output_folder, '..', '..', '..', f'joint_control_error_tradeoff.png'))
    pp.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
