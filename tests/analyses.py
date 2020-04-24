import pickle
from tests.envs import EnvFactory
import mujoco_py
import numpy as np
import Utils
import os
import matplotlib.pyplot as pp
from timeit import default_timer as timer
import matplotlib
import tests.run_opensim_simulations


# Increase font size
matplotlib.rcParams.update({'font.size': 22})


def calculate_mujoco_durations(sim, data, initial_states, N):

    # Calculate simulation duration for each run

    run_times = np.zeros((N, len(data)))
    success = np.zeros((N, len(data)))
    for run_idx in range(len(data)):

        # Get data for this run
        controls = data[run_idx]["controls"]
        timestep = data[run_idx]["timestep"]

        for repeat_idx in range(N):

            # Initialise sim
            Utils.initialise_simulation(sim, timestep, initial_states)

            # Run simulation
            start = timer()
            qpos = Utils.run_simulation(sim, controls)
            end = timer()

            # Check if simulation failed
            if np.any(np.isnan(qpos.flatten())):
                run_success = 0
            else:
                run_success = 1

            run_times[repeat_idx, run_idx] = end-start
            success[repeat_idx, run_idx] = run_success

    return run_times, success


def estimate_run_speed(env, sim, data, train_idxs, test_idxs, initial_states, output_folder):

    # Repeat each simulation N times
    N = 2

    # Calculate OpenSim simulation speed for training and test runs
    all_idxs = train_idxs + test_idxs
    opensim_durations = tests.run_opensim_simulations.run_forward_tool(env, [data[idx]["run"] for idx in all_idxs], N)

    # Calculate MuJoCo simulation speed for all runs; repeat each run N times
    mujoco_durations, mujoco_success = calculate_mujoco_durations(sim, data, initial_states, N)

    # Check MuJoCo success rate
    print("Successful MuJoCo simulations ({} runs, {} repeats): {} %"
          .format(len(data), N, 100*mujoco_success.sum()/mujoco_success.size))

    # Compare total run time for training and test runs
    total_run_time_mujoco = mujoco_durations[:, all_idxs].mean(axis=0).mean()
    mujoco_sd = mujoco_durations[:, all_idxs].mean(axis=0).std()
    total_run_time_opensim = opensim_durations.mean(axis=0).mean()
    opensim_sd = opensim_durations.mean(axis=0).std()
    fig = pp.figure(figsize=(12, 20))
    pp.bar(np.arange(2), [total_run_time_opensim, total_run_time_mujoco], yerr=[opensim_sd, mujoco_sd])
    pp.xlabel(["OpenSim", "MuJoCo"])
    pp.ylabel("Seconds")
    pp.title("Average run time (over {} runs and {} repeats)\nOpenSim: {} seconds\nMuJoCo: {} seconds\nMuJoCo is {} times faster"
             .format(len(all_idxs), N, total_run_time_opensim, total_run_time_mujoco,
                     total_run_time_opensim/total_run_time_mujoco))
    fig.savefig(os.path.join(output_folder, 'run_time_comparison'))

    # Compare run time per evaluation
    #avg_run_time_per_eval = np.sum(run_times[idxs, :], axis=0) / np.sum(num_evals[idxs, :], axis=0)
    #pp.figure(figsize=(12, 8))
    #pp.bar(np.arange(2), avg_run_time_per_eval)
    #pp.xlabel(["OpenSim", "MuJoCo"])
    #pp.ylabel("Seconds")
    #pp.title("Average run time per evaluation for {} runs\nMuJoCo is {} times faster"
    #         .format(len(data), avg_run_time_per_eval[0]/avg_run_time_per_eval[1]))


def calculate_joint_errors(env, viewer, sim, data, target_state_indices, initial_states=None, condition=None):

    # Go through each run in test_data, plot joint errors and record a video
    errors = []
    for run_idx in range(len(data)):
        print(run_idx)

        # Get data for this run
        states = data[run_idx]["states"]
        controls = data[run_idx]["controls"]
        timestep = data[run_idx]["timestep"]
        run = data[run_idx]["run"]

        # Initialise sim
        Utils.initialise_simulation(sim, timestep, initial_states)

        # Run simulation
        qpos = Utils.run_simulation(
            sim, controls, viewer=viewer,
            output_video_file=os.path.join(env.output_folder, run, "{}.mp4".format(condition)))
#        viewer = mujoco_py.MjViewer(sim)
#        qpos = Utils.run_simulation(
#            sim, controls, viewer=viewer)

        # Get timesteps (we assume there's one set of controls per timestep)
        timesteps = np.arange(timestep, (controls.shape[0]+1)*timestep, timestep)

        # Calculate joint errors
        run_err = Utils.estimate_joint_error(states, qpos[:, target_state_indices], plot=True,
                                             joint_names=env.target_states, timesteps=timesteps,
                                             output_file=os.path.join(env.output_folder, run,
                                                                      "{}.png".format(condition)),
                                             error="MAE")
        errors.append(run_err)

    return errors


def analyse_errors(env, params, test_data, output_folder):

    # Open MuJoCo model and initialise a simulation
    model = mujoco_py.load_model_from_path(env.mujoco_model_file)
    sim = mujoco_py.MjSim(model)

    # Check muscle order
    Utils.check_muscle_order(model, test_data)

    # Get indices of target states
    target_state_indices = Utils.get_target_state_indices(model, env)

    # Get initial states
    initial_states = Utils.get_initial_states(model, env)

    return sim, initial_states

    # Update camera position
    cam_id = model._camera_name2id["for_testing"]
    model.cam_pos[cam_id, :] = env.camera_pos[:3]
    model.cam_quat[cam_id, :] = env.camera_pos[3:]

    # Run accuracy analysis first with default parameters
    viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
    #viewer = None
    errors_default = calculate_joint_errors(env, viewer, sim, test_data, target_state_indices, initial_states, "default_parameters")

    # Set parameters and calculate errors again
    Utils.set_parameters(model, params["parameters"], params["muscle_idxs"], params["joint_idxs"])
    errors_optimized = calculate_joint_errors(env, viewer, sim, test_data, target_state_indices, initial_states, "optimized_parameters")

    # Convert into numpy arrays
    errors_default = np.stack(errors_default, axis=0)
    errors_optimized = np.stack(errors_optimized, axis=0)

    # Do a bar plot of error (sum over all joints) before and after optimization
    #x = np.arange(len(test_data))
    #pp.figure()
    #pp.bar(x-0.125, errors_default.sum(axis=1), width=0.25)
    #pp.bar(x+0.125, errors_optimized.sum(axis=1), width=0.25)
    #pp.xticks(x)
    #pp.xlabel('Run')
    #pp.ylabel('Error')
    #pp.legend(["Default params", "Optimized params"])

    # Do a stacked bar plot of average joint errors before and after optimization
    fig1 = pp.figure(figsize=(10, 8))
    avgs_default = np.mean(errors_default, axis=0)
    std_default = np.std(errors_default, axis=0)
    avgs_optimized = np.mean(errors_optimized, axis=0)
    std_optimized = np.std(errors_optimized, axis=0)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    handles = []
    for joint_idx in range(errors_default.shape[1]):

        # Calculate bottom values
        bottom_default = np.sum(avgs_default[:joint_idx])
        bottom_optimized = np.sum(avgs_optimized[:joint_idx])

        # Do bar plots
        h = pp.bar(0, avgs_default[joint_idx], yerr=std_default[joint_idx], width=0.25, bottom=bottom_default, color=colors[joint_idx])
        pp.bar(0.5, avgs_optimized[joint_idx], yerr=std_optimized[joint_idx], width=0.25, bottom=bottom_optimized, color=colors[joint_idx])
        handles.append(h)

    # Set labels and such
    pp.xticks([0, 0.5], ["default params", "optimized params"])
    pp.ylabel('Mean absolute error')
    pp.legend(handles, env.target_states)
    fig1.savefig(os.path.join(output_folder, 'joint_errors_stacked_bar_plot'))

    # Do another bar plot but use separate bars for joints
    x = np.arange(len(avgs_default))
    fig2 = pp.figure(figsize=(20, 8))
    pp.bar(x-0.125, avgs_default, yerr=std_default, width=0.25)
    pp.bar(x+0.125, avgs_optimized, yerr=std_optimized, width=0.25)
    pp.xticks(x, env.target_states)
    pp.ylabel('Mean absolute error (in radians)')
    pp.legend(["Default params", "Optimized params"])
    fig2.savefig(os.path.join(output_folder, 'joint_errors_bar_plot'))

    return sim, target_state_indices, initial_states


def main(model_name):

    # Load test data
    env = EnvFactory.get(model_name)
    with open(env.data_file, 'rb') as f:
        params, data, train_idxs, test_idxs = pickle.load(f)

    # Create a folder for output figures if it doesn't exist
    output_folder = os.path.join(env.output_folder, '..', 'figures')
    os.makedirs(output_folder, exist_ok=True)

    # Get test data
    test_data = [data[idx] for idx in test_idxs]

    # Analyse errors
    sim, initial_states = analyse_errors(env, params, test_data, output_folder)

    # Record opensim videos
    #tests.run_opensim_simulations.run_forward_dynamics(env, [t["run"] for t in test_data], True)

    # Run speed analysis
    estimate_run_speed(env, sim, data, train_idxs, test_idxs, initial_states, output_folder)


if __name__ == "__main__":
    #main(sys.argv[1])
    main("mobl_arms")
    #main("leg6dof9musc")