import pickle
from tests.envs import EnvFactory
import mujoco_py
import numpy as np
import Utils
import os
import matplotlib.pyplot as pp
from timeit import default_timer as timer


def estimate_run_speed(env, sim, data, train_idxs, test_idxs, target_state_indices, initial_states):

    # Calculate mujoco simulation speed for all runs
    run_times = np.zeros((len(data), 2))
    num_evals = np.zeros((len(data), 2))
    success = np.zeros((len(data), 2))
    for run_idx in range(len(data)):

        # Get data for this run
        controls = data[run_idx]["controls"]
        timestep = data[run_idx]["timestep"]
        opensim_run_time = data[run_idx]["run_time"]
        opensim_num_evals = data[run_idx]["num_evals"]
        opensim_success = data[run_idx]["success"]

        # Initialise sim
        Utils.initialise_simulation(sim, timestep, initial_states)

        # Run simulation
        start = timer()
        qpos = Utils.run_simulation(sim, controls)
        end = timer()

        # Check simulation didn't fail
        if np.any(np.isnan(qpos.flatten())):
            mujoco_success = 0
        else:
            mujoco_success = 1

        run_times[run_idx, :] = np.array([opensim_run_time, end-start])
        num_evals[run_idx, :] = np.array([opensim_num_evals, controls.shape[0]])
        success[run_idx, :] = np.array([opensim_success, mujoco_success])

    # Compare success rate
    print("Successful OpenSim simulations: {} ({} %)".format(int(success[:, 0].sum()), 100*success[:, 0].sum()/len(data)))
    print("Successful MuJoCo simulations: {} ({} %)".format(int(success[:, 1].sum()), 100*success[:, 1].sum()/len(data)))
    print()

    # Get indices of runs that were successful in both simulators
    idxs = np.all(success, axis=1)

    # Compare total run time
    total_run_times = np.sum(run_times[idxs, :], axis=0)
    pp.figure()
    pp.bar(np.arange(2), total_run_times)
    pp.xlabel(["OpenSim", "MuJoCo"])
    pp.ylabel("Seconds")
    pp.title("Total run time for {} runs\nMuJoCo is {} times faster".format(len(data),
                                                                            total_run_times[0]/total_run_times[1]))

    # Compare run time per evaluation
    avg_run_time_per_eval = np.sum(run_times[idxs, :], axis=0) / np.sum(num_evals[idxs, :], axis=0)
    pp.figure()
    pp.bar(np.arange(2), avg_run_time_per_eval)
    pp.xlabel(["OpenSim", "MuJoCo"])
    pp.ylabel("Seconds")
    pp.title("Average run time per evaluation for {} runs\nMuJoCo is {} times faster"
             .format(len(data), avg_run_time_per_eval[0]/avg_run_time_per_eval[1]))


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
                                                                      "{}.png".format(condition)))
        errors.append(run_err)

    return errors


def main(model_name):

    # Load test data
    env = EnvFactory.get(model_name)
    with open(env.data_file, 'rb') as f:
        params, data, train_idxs, test_idxs = pickle.load(f)

    # Open MuJoCo model and initialise a simulation
    model = mujoco_py.load_model_from_path(env.mujoco_model_file)
    sim = mujoco_py.MjSim(model)

    # Check muscle order
    Utils.check_muscle_order(model, data)

    # Get indices of target states
    target_state_indices = Utils.get_target_state_indices(model, env)

    # Get initial states
    initial_states = Utils.get_initial_states(model, env)

    # Update camera position
    cam_id = model._camera_name2id["for_testing"]
    model.cam_pos[cam_id, :] = env.camera_pos[:3]
    model.cam_quat[cam_id, :] = env.camera_pos[3:]

    # Get test data
    test_data = [data[idx] for idx in test_idxs]

    # Run accuracy analysis first with default parameters
    viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
    errors_default = calculate_joint_errors(env, viewer, sim, test_data, target_state_indices, initial_states, "default_parameters")

    # Set parameters and calculate errors again
    Utils.set_parameters(model, params["parameters"], params["muscle_idxs"], params["joint_idxs"])
    errors_optimized = calculate_joint_errors(env, viewer, sim, test_data, target_state_indices, initial_states, "optimized_parameters")

    # Convert into numpy arrays
    errors_default = np.stack(errors_default, axis=0)
    errors_optimized = np.stack(errors_optimized, axis=0)

    # Do a bar plot of error (sum over all joints) before and after optimization
    x = np.arange(len(test_data))
    pp.figure()
    pp.bar(x-0.125, errors_default.sum(axis=1), width=0.25)
    pp.bar(x+0.125, errors_optimized.sum(axis=1), width=0.25)
    pp.xticks(x)
    pp.xlabel('Run')
    pp.ylabel('Error')
    pp.legend(["Default params", "Optimized params"])

    # Run speed analysis
    estimate_run_speed(env, sim, data, train_idxs, test_idxs, target_state_indices, initial_states)


if __name__ == "__main__":
    #main(sys.argv[1])
    main("mobl_arms")
    #main("leg6dof9musc")