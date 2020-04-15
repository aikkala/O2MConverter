import opensim
import os
import xmltodict
import sys
import multiprocessing
from timeit import default_timer as timer
from tests.envs import EnvFactory
import matplotlib.pyplot as pp


def run_forward_tool(setup_file, run_folder):

    # Initialise forward tool with the new setup file
    tool = opensim.ForwardTool(setup_file)

    # Run the tool
    start = timer()
    tool.run()
    end = timer()

    # Write output
    with open(os.path.join(run_folder, "output"), 'w') as f:
        f.write("success, run_time\n")
        f.write("1, {}\n".format(end-start))


def main(model_name):

    env = EnvFactory.get(model_name)

    # Get all runs
    runs = os.listdir(env.forward_dynamics_folder)

    # We need to modify the setup file for each run
    with open(env.opensim_setup_file) as f:
        text = f.read()
    setup = xmltodict.parse(text)

    # Loop through runs and simulate forward dynamics
    for run in runs:

        # Ignore this run if it's already been processed
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        if os.path.isfile(os.path.join(run_folder, "output")):
            continue
        #    pass

        # Edit model file, control file, output folder, and initial states
        setup["OpenSimDocument"]["ForwardTool"]["model_file"] = env.opensim_model_file
        setup["OpenSimDocument"]["ForwardTool"]["ControllerSet"]["objects"]["ControlSetController"]["controls_file"] = "controls.sto"
        setup["OpenSimDocument"]["ForwardTool"]["results_directory"] = "."
        setup["OpenSimDocument"]["ForwardTool"]["states_file"] = env.initial_states_file
        setup["OpenSimDocument"]["ForwardTool"]["minimum_integrator_step_size"] = 0.002
        setup["OpenSimDocument"]["ForwardTool"]["maximum_integrator_step_size"] = 0.002
        setup["OpenSimDocument"]["ForwardTool"]['solve_for_equilibrium_for_auxiliary_states'] = 'false'

        # Save the modified setup file
        modified_setup_xml = os.path.join(run_folder, "modified_setup_fd.xml")
        with open(modified_setup_xml, 'w') as f:
            f.write(xmltodict.unparse(setup, pretty=True, indent="  "))

        # Run forward tool as a separate process so we can kill it if it exceeds a time limit
        # (seems like the tool freezes/hangs if the simulation fails)
        p = multiprocessing.Process(target=run_forward_tool, args=(modified_setup_xml, run_folder))
        p.start()

        # Wait for 10 minutes, if the process is still running, kill it
        p.join(5*60)
        if p.is_alive():
            p.terminate()

            # Write output
            with open(os.path.join(run_folder, "output"), 'w') as f:
                f.write("success, run_time\n")
                f.write("0, 0\n")


if __name__ == "__main__":
    #main(sys.argv[1])
    main("mobl_arms")