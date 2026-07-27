.. _container-deployment:

Container Deployment
====================

Docker is a tool that allows for the creation of containers, which are isolated environments that can
be used to run applications. They are useful for ensuring that an application can run on any machine
that has Docker installed, regardless of the host machine's operating system or installed libraries.

We include a Dockerfile and docker-compose.yaml file that can be used to build a Docker image that
contains Isaac Lab and all of its dependencies. This image can then be used to run Isaac Lab in a container.
The Dockerfile is based on the Isaac Sim image provided by NVIDIA, which includes the Omniverse
application launcher and the Isaac Sim application. The Dockerfile installs Isaac Lab and its dependencies
on top of this image.

Cloning the Repository
----------------------

Before building the container, clone the Isaac Lab repository (if not already done):

.. isaaclab-clone-commands::

Next Steps
----------

After cloning, you can choose the deployment workflow that fits your needs:

- :doc:`docker`

  - Learn how to build, configure, and run Isaac Lab in Docker containers.
  - Explains the repository's ``docker/`` setup, the ``container.py`` helper script, mounted volumes,
    image extensions (like ROS 2), and optional CloudXR streaming support.
  - Covers running pre-built Isaac Lab containers from NVIDIA NGC for headless training.

- :doc:`run_docker_example`

  - Learn how to run a development workflow inside the Isaac Lab Docker container.
  - Demonstrates building the container, entering it, executing a sample Python script (`log_time.py`),
    and retrieving logs using mounted volumes.
  - Highlights bind-mounted directories for live code editing and explains how to stop or remove the container
    while keeping the image and artifacts.

- :doc:`cluster`

  - Covers the one-time local prerequisites (``apptainer`` install, SSH setup)
    shared by every cluster deployment workflow.

- :doc:`uw_cluster`

  - Full deployment workflow for the University of Washington's Hyak/Klone and Tillicum
    SLURM clusters.
  - Describes the ``cluster_helpers.sh`` bash wrappers (``cluster_setup``,
    ``cluster_submit``, ``cluster_sweep``, ``cluster_collect``) which keep cluster infra
    off your working branch via a transient squash-merge of ``feature/uw-cluster``.
  - Includes multi-cluster dispatch, auto-resume on preemption, hyperparameter
    sweeps, Weights & Biases propagation, and troubleshooting.

.. toctree::
   :maxdepth: 1
   :hidden:

   docker
   run_docker_example
   cluster
   uw_cluster
