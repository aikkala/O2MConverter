import Utils


class EnvFactory:

    class EnvTemplate:
        def __init__(self, opensim_setup_file, forward_dynamics_folder, mujoco_model_file, initial_states_file=None):
            self.opensim_setup_file = opensim_setup_file
            self.forward_dynamics_folder = forward_dynamics_folder
            self.mujoco_model_file = mujoco_model_file
            self.target_states = ["shoulder_elv", "shoulder1_r2", "shoulder_rot",
                                  "elbow_flexion", "pro_sup", "wrist_hand_r1", "wrist_hand_r3"]

            # Read initial states from a file if given
            self.initial_states = None
            if initial_states_file is not None:
                self.initial_states = {"joints": {}, "actuators": {}}
                states, hdr = Utils.parse_sto_file(initial_states_file)
                state_names = list(states)
                for state_name in state_names:
                    if state_name.endswith(".fiber_length"):
                        continue
                    elif state_name.endswith("_u"):
                        if state_name[:-2] not in self.initial_states["joints"]:
                            self.initial_states[state_name[:-2]] = {}
                        self.initial_states["joints"][state_name[:-2]]["qvel"] = states[state_name][0]
                    elif state_name.endswith(".activation"):
                        self.initial_states["actuators"][state_name[:-11]] = states[state_name][0]
                    else:
                        if state_name not in self.initial_states["joints"]:
                            self.initial_states["joints"][state_name] = {}
                        self.initial_states["joints"][state_name]["qpos"] = states[state_name][0]

    MoBL_ARMS = EnvTemplate(
        '/home/aleksi/Workspace/O2MConverter/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/setup_fd.xml',
        '/home/aleksi/Workspace/O2MConverter/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/forward_dynamics',
        '/home/aleksi/Workspace/O2MConverter/models/converted/MoBL_ARMS_model_for_mujoco_converted/MoBL_ARMS_model_for_mujoco_converted.xml',
        '/home/aleksi/Workspace/O2MConverter/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/initial_states.sto')

    @staticmethod
    def get(env_name):
        if env_name.lower() == "mobl_arms":
            return EnvFactory.MoBL_ARMS
        else:
            raise NotImplementedError
