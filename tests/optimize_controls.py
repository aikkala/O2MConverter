from tests.envs import EnvFactory
import pickle
import random
import numpy as np
import scipy
import cma
import Utils
import mujoco_py
import os
import matplotlib.pyplot as pp
from scipy.signal import sosfiltfilt, butter
import sys

# Increase font size
import matplotlib
matplotlib.rcParams.update({'font.size': 22})


def preprocess_controls(controls):

    # Low-pass filter
    #sos = butter(4, 0.01, output='sos')
    #controls = sosfiltfilt(sos, controls, axis=0)

    # Clip to range [0, 1]
    controls = np.clip(controls, 0, 1)

    return controls


def main(model_name, alpha=1):
    # Choose a random test run and optimize the controls to match OpenSim trajectory

    # Make sure alpha is a float
    alpha = float(alpha)

    # Load test data
    env = EnvFactory.get(model_name)
    with open(env.data_file, 'rb') as f:
        params, data, train_idxs, test_idxs = pickle.load(f)

    # Initialise MuJoCo with the converted model
    model = mujoco_py.load_model_from_path(env.mujoco_model_file)

    # Initialise simulation
    sim = mujoco_py.MjSim(model)

    # Check muscle order
    Utils.check_muscle_order(model, data)

    # Get indices of target states
    target_state_indices = Utils.get_target_state_indices(model, env)

    # Get initial states
    initial_states = Utils.get_initial_states(model, env)

    # Set parameters
    Utils.set_parameters(model, params["parameters"], params["muscle_idxs"], params["joint_idxs"])

    # Go through all test runs
    #viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
    viewer = None
    for run_idx in test_idxs:

        # Get run data
        run_data = data[run_idx]

        # Parameterize controls as splines
        fs = 10
        T = run_data["controls"].shape[0]
        nmuscles = run_data["controls"].shape[1]
        xi = np.arange(0, T)
        xn = np.arange(T*(1/fs), T, T*(1/fs), dtype=int)
        xn = np.insert(xn, 0, 0)
        xn = np.append(xn, T-1)
        y = np.zeros(run_data["controls"][xn, :].flatten().shape)

        # Use CMA-ES to optimize
        sigma = 0.5
        niter = 1000
        opts = {"popsize": 32, "maxiter": niter, "CMA_diagonal": False}
        optimizer = cma.CMAEvolutionStrategy(y, sigma, opts)

        while not optimizer.stop():

            # Get candidate solutions
            solutions = optimizer.ask()

            # Loop through solutions
            fitness = np.zeros((len(solutions),))
            for solution_idx, solution in enumerate(solutions):

                # Create a new set of controls
                f_spline = scipy.interpolate.CubicSpline(xn, np.reshape(solution, (fs+1, nmuscles)))
                controls = preprocess_controls(run_data["controls"] - f_spline(xi))

                # Initialise sim
                Utils.initialise_simulation(sim, env.timestep, initial_states)

                # Run simulation
                qpos = Utils.run_simulation(sim, controls)

                # Calculate fitness as a combination of joint error and control error
                joint_error = np.sum(Utils.estimate_joint_error(run_data["states"], qpos[:, target_state_indices], error="squared_sum"))
                control_error = np.sum((run_data["controls"] - controls)**2)
                fitness[solution_idx] = alpha * joint_error + (1 - alpha) * control_error

            # Tell optimizer
            optimizer.tell(solutions, fitness)
            optimizer.disp()

        # Continue if optimization was ended prematurely
        if optimizer.countiter < niter:
            continue

        # Record a video with found parameters
        solution = optimizer.result.xfavorite

        # Set camera parameters
        cam_id = model._camera_name2id["for_testing"]
        model.cam_pos[cam_id, :] = env.camera_pos[:3]
        model.cam_quat[cam_id, :] = env.camera_pos[3:]

        # Create a new set of controls
        f_spline = scipy.interpolate.CubicSpline(xn, np.reshape(solution, (fs + 1, nmuscles)))
        controls = preprocess_controls(run_data["controls"] - f_spline(xi))

        # Initialise sim
        Utils.initialise_simulation(sim, env.timestep, initial_states)

        # Make sure output folder exists
        alpha_str = f"{alpha:.3f}"[2:]
        output_folder = os.path.join(env.output_folder, run_data["run"], "optimized_control", alpha_str)
        os.makedirs(output_folder, exist_ok=True)

        # Run and record video
        qpos = Utils.run_simulation(sim, controls, viewer=viewer,
                                    output_video_file=os.path.join(output_folder, "controls_optimized.mp4"))

        # Calculate joint mean absolute error and plot differences in joints
        timesteps = np.arange(env.timestep, (controls.shape[0]+1)*env.timestep, env.timestep)
        joint_error = Utils.estimate_joint_error(run_data["states"], qpos[:, target_state_indices],
                                                 plot=True, joint_names=env.target_states, timesteps=timesteps,
                                                 output_file=os.path.join(output_folder, "controls_optimized_parameters.png"),
                                                 error="MAE")

        # Calculate also mean absolute error in control values
        control_error = abs(run_data["controls"] - controls).mean(axis=0)

        # Save optimized controls and the error
        np.savetxt(os.path.join(output_folder, "controls.csv"), controls)
        np.savetxt(os.path.join(output_folder, "joint_error.csv"), joint_error)
        np.savetxt(os.path.join(output_folder, "control_error.csv"), control_error)

        # Plot original controls and optimized controls
        fig, axs = pp.subplots(2, 1, sharex=True, sharey=True, figsize=(20, 20))
        fig.suptitle('Original and optimized muscle excitations')
        axs[0].plot(timesteps, run_data["controls"])
        axs[1].plot(timesteps, controls)
        pp.ylabel('Muscle excitation')
        pp.xlabel('Time (seconds)')
        fig.savefig(os.path.join(output_folder, "controls_optimized.png"))
        pp.close(fig)

        # Do also a bar plot of "total muscle activation"
        fig, ax = pp.subplots(figsize=(25, 8))
        x = np.arange(nmuscles)
        pp.bar(x-0.125, np.sum(run_data["controls"], axis=0), width=0.25)
        pp.bar(x+0.125, np.sum(controls, axis=0), width=0.25)
        pp.legend(["Reference", "Optimized"])
        fig.suptitle("Total muscle excitation during trajectory")
        pp.ylabel('Muscle excitation')
        pp.xlabel('Muscle')
        fig.savefig(os.path.join(output_folder, "muscle_excitations.png"))
        pp.close(fig)


if __name__ == "__main__":
#    if len(sys.argv) < 3:
#        alpha = 1
#    else:
#        alpha = sys.argv[2]
    main(*sys.argv[1:])
