# DTU HPC user guide

## Setup

This document describes step-by-step how to use the DTU HPC and its available GPUs to train deep neural networks for this project.
Here are the necessary steps:

1. Log in to DTU HPC using ssh (the same method you used to clone files from the databar to your local computer). It can be easier to use VS Code Remote for this to open and edit files on the DTU server more easily (see [VS Code tutorial](https://code.visualstudio.com/docs/remote/ssh-tutorial) about this).

```shell
ssh <your DTU username>@login.gbar.dtu.dk
```

2. Once logged in, generate an ssh key for Github, so you can clone the repository (https is only available using Git Credential Manager, but that seemed harder to get working than ssh). Find more information about the necessary procedure on the [relevant Github documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux). The steps from the doc are replicated here.

    a. Generate ssh key

    ```shell
    ssh-keygen -t ed25519 -C "<your github account email address>"
    ```

    b. Hit enter to select defaults for selecting the file where to save the key (you can find it later on path `~/.ssh/id_ed25519` and on `~/.ssh/id_ed25519.pub`)

    c. Add the generated public key to your Github account [here](https://github.com/settings/ssh/new). You can get the contents of your public ssh key file using the following command (the path is printed by `ssh-keygen` after generation):

    ```shell
    cat /path/to/ssh/key/<filename>.pub
    ```

3. Clone the repository inside deep-learning folder (this is necessary, because environment variables are defined like this)

    ```shell
    cd ~ # go to home 
    mkdir deep-learning # create dl folder in home
    cd deep-learning # move to dl folder
    git clone git@github.com:Lorl0rd/dtu-dl-p22.git
    # checkout to a given branch if needed (default is main)
    # git checkout dev_HPC
    ```

4. Go to the repository

    ```shell
    cd dtu-dl-p22
    ```

5. Copy data files from shared folder to the repository (because they are not version-controlled). This can take some time (in the order of 10-20 minutes).

```shell
cp /zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data . --verbose
```

6. Install uv (command copied from [uv docs](https://docs.astral.sh/uv/getting-started/installation/#installation-methods))

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

7. Sync python version and dependencies using uv. Take care to check if the correct CUDA-enabled pytorch versions are selected in `pyproject.toml`. You can change the file through VS Code Remote SSH connection. (This feature may be handy later to also edit other files in the project directory on DTU server.)

    ```shell
    uv sync
    ```

8. Add necessary shell environment variables to `.env-vars` (not always necessary, you can just accept the defaults on the repository).

9. Modify the bash script, that defines how to run the batch job on HPC cluster. You can leave the defaults, just take care of the following:
    a. Change the Python file to run (if necessary), e.g. check the following line at the end of the file

    ```bash
    # run training
    python <name of your script to run>.py
    ```
    
    b. Change used CPU memory (16 GB should be more than enough, but see the [HPC website]() for more information)

    ```bash
    #BSUB -R "rusage[mem=16GB]"
    ```

    c. Change wall time (if computations take longer than this duration to finish, the job will be interrupted). The GPUs have a lot of memory compared to our needs, so training will not take multiple hours with a reasonably sized network and number of epochs. Although take this as a guess.

    ```bash
    #BSUB -W 1:00
    ```

10. Add the (bash) shell script to the batch queue for running using the following command in the ssh terminal. Note that you can use the DTU server for simple computations, assigning a job is only necessary for heavy GPU calculations.

    ```shell
    bsub < jobscript.sh
    ```

11. You can check some details about your queued jobs using the following command

    ```shell
    bstat
    ```

## Important notes

The job only executes your python file, then clears variable data. This means, that you need to take care to save anything important, e.g. the trained network for later tests. You can use `pickle` for saving Python objects or `torch.save` to save model parameters to disk. More information can be found on [pytorch documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-and-loading-models).

## Useful resources

### Useful commands to specify batch job

**Important note**: Do not remove the comment in front of BSUB, it is required for the resource manager to correctly interpret the command.

```bash
### General options for selecting a GPU
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options --
```

## Resources

- [DTU HPC website](https://www.hpc.dtu.dk/?page_id=42)
- [available GPUs](https://www.hpc.dtu.dk/?page_id=2759)
- [HPC Guide 2023 Google document](https://docs.google.com/document/d/1bRtX87ZD7faG1b5CayN5LKUPcn75xmrWt6jFcuwoGMU/edit?tab=t.0)
