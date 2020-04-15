import Utils
import numpy as np
from osim.env.osim import OsimEnv


class EnvFactory:

    class EnvTemplate:
        def __init__(self, opensim_setup_file, forward_dynamics_folder, mujoco_model_file, data_file,
                     output_folder, camera_pos, opensim_model_file, initial_states_file, target_states,
                     osim_mapping):
            self.opensim_setup_file = opensim_setup_file
            self.forward_dynamics_folder = forward_dynamics_folder
            self.mujoco_model_file = mujoco_model_file
            self.data_file = data_file
            self.output_folder = output_folder
            self.camera_pos = camera_pos
            self.opensim_model_file = opensim_model_file
            self.target_states = target_states
            self.osim_mapping = osim_mapping

            # Read initial states from a file if given
            self.initial_states_file = initial_states_file
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
        '/home/aleksi/Workspace/O2MConverter/tests/mobl_arms/forward_dynamics',
        '/home/aleksi/Workspace/O2MConverter/models/converted/MoBL_ARMS_model_for_mujoco_converted/MoBL_ARMS_model_for_mujoco_converted.xml',
        '/home/aleksi/Workspace/O2MConverter/tests/mobl_arms/data.pckl',
        '/home/aleksi/Workspace/O2MConverter/tests/mobl_arms/output',
        np.array([1.8, -0.1, 0.7, 0.5, 0.5, 0.5, 0.5]),
        '/home/aleksi/Workspace/O2MConverter/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/ModelFiles/MoBL_ARMS_model_for_opensim.osim',
        '/home/aleksi/Workspace/O2MConverter/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/initial_states.sto',
        ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup"],
        {"r_z": ("groundthorax", 0),
         "sternoclavicular_r2": ("sternoclavicular", 0),
         "sternoclavicular_r3": ("sternoclavicular", 1),
         "unrotscap_r3": ("unrotscap", 0),
         "unrotscap_r2": ("unrotscap", 1),
         "acromioclavicular_r2": ("acromioclavicular", 0),
         "acromioclavicular_r3": ("acromioclavicular", 1),
         "acromioclavicular_r1": ("acromioclavicular", 2),
         "unrothum_r1": ("unrothum", 0),
         "unrothum_r3": ("unrothum", 1),
         "unrothum_r2": ("unrothum", 2),
         "elv_angle": ("shoulder0", 0),
         "shoulder_elv": ("shoulder1", 0),
         "shoulder1_r2": ("shoulder1", 1),
         "shoulder_rot": ("shoulder2", 0),
         "elbow_flexion": ("elbow", 0),
         "pro_sup": ("radioulnar", 0),
         "deviation": ("radiocarpal", 0),
         "flexion": ("radiocarpal", 1),
         "wrist_hand_r1": ("wrist_hand", 0),
         "wrist_hand_r3": ("wrist_hand", 1)}
    )

#    leg6dof9musc = EnvTemplate(
#        '/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/setup_fd.xml',
#        '/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/forward_dynamics',
#        '/home/aleksi/Workspace/O2MConverter/models/converted/leg6dof9musc_for_testing_converted/leg6dof9musc_for_testing_converted.xml',
 ##       '/home/aleksi/Workspace/O2MConverter/models/converted/leg6dof9musc_for_testing_converted/test_data.pckl',
 #       np.array([1.8, -0.1, 0.7, 0.5, 0.5, 0.5, 0.5]),
 #       '/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/leg6dof9musc_for_testing.osim',
 #       '/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/initial_states.sto',
  #      ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
  #  )

    @staticmethod
    def get(env_name):
        if env_name.lower() == "mobl_arms":
            return EnvFactory.MoBL_ARMS
        elif env_name.lower() == "leg6dof9musc":
            return EnvFactory.leg6dof9musc
        else:
            raise NotImplementedError


class OsimWrapper(OsimEnv):

    def __init__(self, env, visualize=True, integrator_accuracy=5e-5, report=None):
        self.model_path = env.opensim_model_file
        self.env = env

        # Load model
        super(OsimWrapper, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy)

        # Set timestep
        self.osim_model.stepsize = 0.002

        # initialize state
        state = self.osim_model.model.initializeState()
        self.osim_model.set_state(state)

        # Get joint names
        self.joint_names = self.get_observation_names()

        # Get muscle names, this is the order control values should be in when input to step method
        self.muscle_names = [muscle.getName() for muscle in self.osim_model.muscleSet]
        #self.muscle_mapping = {muscle: self.muscle_names.index(muscle) for muscle in self.muscle_names}

        # Check if we want to save simulated states
        if report:
            bufsize = 0
            self.observations_file = open('%s-obs.csv' % (report,), 'w', bufsize)
            self.actions_file = open('%s-act.csv' % (report,), 'w', bufsize)
            self.get_headers()

    def reset(self, project=True, obs_as_dict=True):

        # initialize state
        state = self.osim_model.model.initializeState()

        # Get joint positions and velocities
        Q = state.getQ()
        QDot = state.getQDot()

        # Set joint positions
        for joint_idx, joint_name in enumerate(self.joint_names):
            Q[joint_idx] = self.env.initial_states["joints"][joint_name]["qpos"]
            QDot[joint_idx] = self.env.initial_states["joints"][joint_name]["qvel"]

        # How to set initial muscle activations? Are they even used since we overwrite them instantly?

        # Set joint positions and velocities
        state.setQ(Q)
        state.setU(QDot)
        self.osim_model.set_state(state)
        self.osim_model.model.equilibrateMuscles(self.osim_model.state)

        # Set time to zero
        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0
        self.t = 0

        self.osim_model.reset_manager()

        return self.get_observation()

    def load_model(self, model_path=None):
        super(OsimWrapper, self).load_model(model_path)

    def step(self, action):

        obs, reward, done, info = super(OsimWrapper, self).step(action, project=True, obs_as_dict=False)
        self.t += self.osim_model.stepsize

        return obs, reward, done, info

    def is_done(self):
        return False

    def get_observation_dict(self):

        # Return only joint positions
        states = self.get_state_desc()
        obs = {}
        for joint_name, mapping in self.env.osim_mapping.items():
            obs[joint_name] = states["joint_pos"][mapping[0]][mapping[1]]
#        obs = {"r_z": states["joint_pos"]["groundthorax"][0],
#               "sternoclavicular_r2": states["joint_pos"]["sternoclavicular"][0],
#               "sternoclavicular_r3": states["joint_pos"]["sternoclavicular"][1],
#               "unrotscap_r3": states["joint_pos"]["unrotscap"][0],
#               "unrotscap_r2": states["joint_pos"]["unrotscap"][1],
#               "acromioclavicular_r2": states["joint_pos"]["acromioclavicular"][0],
#               "acromioclavicular_r3": states["joint_pos"]["acromioclavicular"][1],
#               "acromioclavicular_r1": states["joint_pos"]["acromioclavicular"][2],
#               "unrothum_r1": states["joint_pos"]["unrothum"][0],
#               "unrothum_r3": states["joint_pos"]["unrothum"][1],
#               "unrothum_r2": states["joint_pos"]["unrothum"][2],
#               "elv_angle": states["joint_pos"]["shoulder0"][0],
#               "shoulder_elv": states["joint_pos"]["shoulder1"][0],
#               "shoulder1_r2": states["joint_pos"]["shoulder1"][1],
#               "shoulder_rot": states["joint_pos"]["shoulder2"][0],
#               "elbow_flexion": states["joint_pos"]["elbow"][0],
#               "pro_sup": states["joint_pos"]["radioulnar"][0],
#               "deviation": states["joint_pos"]["radiocarpal"][0],
#               "flexion": states["joint_pos"]["radiocarpal"][1],
#               "wrist_hand_r1": states["joint_pos"]["wrist_hand"][0],
#               "wrist_hand_r3": states["joint_pos"]["wrist_hand"][1]}
#
        return obs

    def get_observation(self):
        return np.fromiter(self.get_observation_dict().values(), dtype=float)

    def get_observation_names(self):
        return list(self.get_observation_dict().keys())

    def get_state_desc(self):
        d = super(OsimWrapper, self).get_state_desc()
        return d

    def get_reward(self):
        return 0
