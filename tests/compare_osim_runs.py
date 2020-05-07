import Utils
from tests.envs import EnvFactory
import os
import pickle
import matplotlib.pyplot as pp
import numpy as np


def main():

    # Go through test runs and compare states from osim to states from forward dynamics tool
    env = EnvFactory.get("mobl_arms")

    # Load states calculated with forward tool
    with open(env.data_file, 'rb') as f:
        _, data, _, test_idxs = pickle.load(f)

    # Go through only test data
    test_data = [data[idx] for idx in test_idxs]

    # Loop test data
    joint_error = []
    for run_idx in range(len(test_data)):

        # Check if there are osim states for this run
        osim_file = os.path.join(env.forward_dynamics_folder, test_data[run_idx]["run"], "FDS2_states.sto")
        if not os.path.isfile(osim_file):
            continue

        # Also check whether this run was successful in forward tool
        if not test_data[run_idx]["success"]:
            continue

        # Get osim states
        osim, _ = Utils.parse_sto_file(osim_file)

        state_names = list(osim)

        # Parse state names
        parsed_state_names = []
        for state_name in state_names:
            p = state_name.split("/")
            if p[-1] != "value":
                parsed_state_names.append(state_name)
            else:
                parsed_state_names.append(p[1])

        # Rename states
        osim.columns = parsed_state_names

        # Use only env.target_states; filter and reorder columns in osim
        osim = osim.filter(items=env.target_states)
        osim = osim[env.target_states]

        # Get forward dynamics states
        fds = test_data[run_idx]["states"]

        # Calculate joint error over simulation
        joint_error.append(fds - osim.values)

    # Concatenate joint errors into a numpy array
    errors = np.asarray(joint_error)

    # Plot each joint separately
    pp.figure(figsize=(10, 20))
    for idx in range(len(env.target_states)):

        ax = pp.subplot(len(env.target_states), 1, idx+1)

        # Calculate [25, 50, 75]th percentiles of absolute errors for each joint
        prc = np.percentile(abs(errors[:, :, idx]), [25, 50, 75], axis=0)

        # Plot
        ax.plot(np.arange(prc.shape[1]), prc[1, :])
        ax.fill_between(np.arange(prc.shape[1]), prc[0, :], prc[2, :], alpha=0.3)
        ax.set_ylabel('Absolute displacement')
        ax.set_xlabel(env.target_states[idx])

    pp.show()
    print("Average absolute joint error per timestep: {}".format(np.mean(abs(errors))))


main()