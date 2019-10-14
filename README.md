OpenSim to MuJoCo XML converter. Work very much in progress.

Install
---

1. [Install MuJoCo](http://mujoco.org)
1. By default mujoco-py looks for MuJoCo and the license key in folder ~/.mujoco. If your installation is in a different path let mujoco-py know by setting the environment variables, e.g. 
    - `export MUJOCO_PY_MUJOCO_PATH=/home/aleksi/Workspace/mujoco200_linux`
    - `export MUJOCO_PY_MJKEY_PATH=/home/aleksi/Workspace/mujoco200_linux/bin`
    - `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aleksi/Workspace/mujoco200_linux/bin`
1. Install a mujoco-py requirement that will cause problems if it's missing `sudo apt install libosmesa6-dev`
1. I had a problem related to Nvidia graphics driver and OpenGL (conda environment creation failed when pip was trying to install mujoco-py, error message: "cannot find -lGL", i.e. libGL.so was missing). This was resolved with `sudo apt install libgl1-mesa-dev` 
1. Create a conda environment from the environment file O2MConverter.yml `conda env create --name O2MConverter --file=O2MConverter.yml`
