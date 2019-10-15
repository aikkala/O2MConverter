OpenSim to MuJoCo XML converter. Work very much in progress.

Install (for Ubuntu 18.04)
---

1. [Install MuJoCo](http://mujoco.org) and get the license
1. By default mujoco-py looks for MuJoCo and the license key in folder ~/.mujoco. If your installation is in a different path let mujoco-py know by setting the environment variables, e.g. 
    - `export MUJOCO_PY_MUJOCO_PATH=/home/aleksi/Workspace/mujoco200_linux`
    - `export MUJOCO_PY_MJKEY_PATH=/home/aleksi/Workspace/mujoco200_linux/bin/mjkey.txt`
    - `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aleksi/Workspace/mujoco200_linux/bin`
1. Install mujoco-py requirements that might cause problems if they're missing `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`
1. I had a problem with OpenGL (conda environment creation failed when pip was trying to install mujoco-py, error message: "cannot find -lGL", i.e. libGL.so was missing). This was resolved with `sudo apt install libgl1-mesa-dev`. Another fix for this issue, according to mujoco-py troubleshooting, is that you create the missing symbolic link directly: `sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so` 
1. Create a conda environment from the environment file O2MConverter.yml `conda env create --name O2MConverter --file=O2MConverter.yml`

Troubleshooting
---
1. mujoco-py viewer crashed with **GLEW initialization error: Missing GL version**
    - Install libglew-dev `sudo apt install libglew-dev` and set environment variable `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`
