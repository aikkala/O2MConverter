from tests.envs import EnvFactory
import mujoco_py
import os
import Utils
import pandas as pd
import random
import math
import numpy as np
import cma
import skvideo
import pickle
import matplotlib.pyplot as pp


def get_run_info(run_folder):
    output_file = os.path.join(run_folder, "output")
    if os.path.isfile(output_file):
        run_info = pd.read_csv(output_file, delimiter=", ", header=0, engine="python")
    return run_info


def is_unique(values):
    return (values - values[0] < 1e-6).all()


def collect_data_from_runs(env):

    # Get sub-folders
    runs = os.listdir(env.forward_dynamics_folder)

    # Go through all runs
    data = []
    for run in runs:

        # Get controls
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        controls, ctrl_hdr = Utils.parse_sto_file(os.path.join(run_folder, "controls.sto"))

        # Check timesteps
        #timesteps = np.diff(controls.index.values)
        #if not is_unique(timesteps):
        #    raise ValueError("Check timesteps in the control file of run {}".format(run_folder))
        #if timesteps[0] > 0.01:
        #    raise ValueError("This timestep might be too big")
        #timestep = timesteps[0]

        # Check if this run was successful
        #run_info = get_run_info(run_folder)

        # Don't process states if run was unsuccessful
        #if not run_info["success"][0]:
        #    success = 0

        #else:

        # Get states
        states, state_hdr = Utils.parse_sto_file(os.path.join(run_folder, "FDS_states.sto"))

        # Make sure OpenSim simulation was long enough (should be if it didn't fail, just double-checking)
        #if states.index[-1] < controls.index[-2]:
        #    raise ValueError("Did this run fail?\n  {}".format(run_folder))

        # We're interested only in a subset of states
        state_names = list(states)

        # Parse state names
        parsed_state_names = []
        for state_name in state_names:
            p = state_name.split("/")
            if p[-1] != "value":
                parsed_state_names.append(state_name)
            else:
                parsed_state_names.append(p[1])

        # Rename states
        states.columns = parsed_state_names

        # Check that initial states are correct (OpenSim forward tool seems to ignore initial states
        # of locked joints); if initial states do not match, then trajectories in mujoco will start
        # from incorrect states
        if env.initial_states is not None and "joints" in env.initial_states:
            for state_name in env.initial_states["joints"]:
                if abs(states[state_name][0] - env.initial_states["joints"][state_name]["qpos"]) > 1e-5:
                    raise ValueError('Initial states do not match')

        # Filter and reorder states
        states = states.filter(items=env.target_states)
        states = states[env.target_states]

        # Get number of evaluations (forward steps); note that the last timestep isn't simulated
        num_evals = len(states) - 1

        # Reindex states
        states = Utils.reindex_dataframe(states, np.arange(env.timestep, controls.index.values[-1]+2*env.timestep, env.timestep))

        # Get state values and state names
        state_values = states.values
        state_names = list(states)

        # Don't use this data if there were nan states
        if np.any(np.isnan(state_values)):
            success = 0
            state_values = []
            state_names = []
            num_evals = 0
        else:
            success = 1

        data.append({"states": state_values, "controls": controls.values,
                     "state_names": state_names, "muscle_names": list(controls),
                     "timestep": env.timestep, "success": success, "run": run, "num_evals": num_evals})

    return data


def do_optimization(env, data):

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

    #viewer = mujoco_py.MjViewer(sim)
    #controls = data[0]["controls"]
    #controls *= 0
    #timestep = data[0]["timestep"]
    #Utils.initialise_simulation(sim, timestep, initial_states)
    #Utils.run_simulation(sim, controls, viewer=viewer)

    # Go through training data once to calculate error with default parameters
    default_error = np.zeros((len(data),))
    for run_idx in range(len(data)):
        states = data[run_idx]["states"]
        controls = data[run_idx]["controls"]
        timestep = data[run_idx]["timestep"]

        # Initialise sim
        Utils.initialise_simulation(sim, timestep, initial_states)

        # Run simulation
        qpos = Utils.run_simulation(sim, controls)

        # Calculate joint errors
        default_error[run_idx] += np.sum(Utils.estimate_joint_error(states, qpos[:, target_state_indices]))

    # Get initial values for params
    sigma = 1
    niter = 500
    nmuscles = len(model.actuator_names)
    joint_idxs = list(set(range(len(model.joint_names))) - set(model.eq_obj1id[np.asarray(model.eq_active, dtype=bool)]))
    njoints = len(joint_idxs)
    params = [5] * nmuscles + [2] * (2*nmuscles + 2*njoints)

    # Initialise optimizer
    opts = {"popsize": 16, "maxiter": niter, "CMA_diagonal": True}
    optimizer = cma.CMAEvolutionStrategy(params, sigma, opts)
    nbatch = len(data)

    # Keep track of errors
    history = np.empty((niter,))
    history.fill(np.nan)

    # Initialise plots
    pp.ion()
    fig1 = pp.figure(1, figsize=(10, 5))
    fig1.gca().plot([0, len(data)], [1, 1], 'k--')
    bars = fig1.gca().bar(np.arange(len(data)), [0]*len(data))
    fig1.gca().axis([0, nbatch, 0, 1.2])

    fig2 = pp.figure(2, figsize=(10, 5))
    fig2.gca().plot([0, niter], [default_error.sum(), default_error.sum()], 'k--')
    line, = fig2.gca().plot(np.arange(niter), [0]*niter)
    fig2.gca().axis([0, niter, 0, 1.1*default_error.sum()])

    while not optimizer.stop():
        solutions = optimizer.ask()

        # Test solutions on a batch of runs
        batch_idxs = random.sample(list(np.arange(len(data))), nbatch)

        errors = np.zeros((len(batch_idxs), len(solutions)))
        params_cost = np.zeros((len(solutions)))
        for solution_idx, solution in enumerate(solutions):

            # Set parameters
            Utils.set_parameters(model, np.exp(solution), np.arange(nmuscles), joint_idxs)
            params_cost[solution_idx] = 1e-3 * np.sum(np.exp(solution))

            # Go through all simulations in batch
            try:
                for idx, run_idx in enumerate(batch_idxs):
                    states = data[run_idx]["states"]
                    controls = data[run_idx]["controls"]
                    timestep = data[run_idx]["timestep"]

                    # Initialise sim
                    Utils.initialise_simulation(sim, timestep, initial_states)

                    # Run simulation
                    qpos = Utils.run_simulation(sim, controls)

                    # Calculate joint errors
                    errors[idx, solution_idx] = np.sum(Utils.estimate_joint_error(states, qpos[:, target_state_indices]))

            except mujoco_py.builder.MujocoException:
                print("Failed for parameters")
                print(np.exp(solution))
                errors[:, solution_idx] = np.nan

        # Use sum of errors over runs as fitness, and calculate mean error for each run
        fitness = errors.sum(axis=0) + params_cost
        avg_run_error = np.nanmean(errors, axis=1)

        # Plot mean error per run as a percentage of default error
        prc = avg_run_error / default_error[batch_idxs]
        for rect, y in zip(bars, prc):
            rect.set_height(y)
        fig1.canvas.draw()
        fig1.canvas.flush_events()

        # Plot history of mean fitness
        history[optimizer.countiter] = np.nanmean(fitness)
        line.set_ydata(history)
        fig2.gca().axis([0, optimizer.countiter, 0, 1.1*max(history)])
        fig2.canvas.draw()
        fig2.canvas.flush_events()

        # Ignore failed runs / solutions
        valid_idxs = np.where(np.isfinite(fitness))[0]

        optimizer.tell(solutions, fitness)
        optimizer.logger.add()
        optimizer.disp()

    # Return found solution
    return {"parameters": np.exp(optimizer.result.xfavorite),
            "joint_idxs": joint_idxs, "muscle_idxs": np.arange(len(model.actuator_names)),
            "history": history}


def main(model_name):

    # Get env
    env = EnvFactory.get(model_name)

    # Collect states and controls from successful runs
    data = collect_data_from_runs(env)

    # Divide successful runs into training and testing sets
    success_idxs = []
    for run_idx in range(len(data)):
        if data[run_idx]["success"]:
            success_idxs.append(run_idx)

    # Use 80% of runs for training
    k = math.ceil(0.8*len(success_idxs))
    train_idxs = random.sample(success_idxs, k)
    test_idxs = list(set(success_idxs) - set(train_idxs))

    # Get training data
    #train_idxs = [0]
    train_set = [data[idx] for idx in train_idxs]

    # Do optimization with CMA-ES
    parameters = do_optimization(env, train_set)

    # Save parameters, data, and indices
    with open(env.data_file, 'wb') as f:
        pickle.dump([parameters, data, train_idxs, test_idxs], f)


if __name__ == "__main__":
    #main(sys.argv[1])
    main("mobl_arms")
    #main('leg6dof9musc')