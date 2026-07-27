.. _deployment-cluster:


Cluster Guide
=============

Clusters are a great way to speed up training and evaluation of learning algorithms.
While the Isaac Lab Docker image can be used to run jobs on a cluster, many clusters only
support singularity images. This is because `singularity`_ is designed for
ease-of-use on shared multi-user systems and high performance computing (HPC) environments.
It does not require root privileges to run containers and can be used to run user-defined
containers.

Singularity is compatible with all Docker images. In this section, we describe how to
convert the Isaac Lab Docker image into a singularity image and use it to submit jobs to a cluster.

.. attention::

   This branch (``feature/uw-cluster``) targets the University of Washington's Hyak/Klone
   and Tillicum clusters. For the full deployment workflow — multi-cluster dispatch,
   auto-resume on preemption, hyperparameter sweeps, and the ``cluster_helpers.sh``
   bash wrappers — see :doc:`uw_cluster`.

   This page covers only the one-time local prerequisites (``apptainer`` and SSH)
   that the UW workflow builds on.

.. _cluster-setup-instructions:

Setup Instructions
------------------

In order to export the Docker Image to a singularity image, `apptainer`_ is required.
A detailed overview of the installation procedure for ``apptainer`` can be found in its
`documentation`_. For convenience, we summarize the steps here for a local installation:

.. code:: bash

    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt update
    sudo apt install -y apptainer

For simplicity, we recommend that an SSH connection is set up between the local
development machine and the cluster. Such a connection will simplify the file transfer and prevent
the user cluster password from being requested multiple times.

.. attention::
  The workflow has been tested with:

  - ``apptainer version 1.2.5-1.el7`` and ``docker version 24.0.7``
  - ``apptainer version 1.3.4`` and ``docker version 27.3.1``

  In the case of issues, please try to switch to those versions.


Next steps
----------

Once ``apptainer`` is installed locally and SSH is configured, follow :doc:`uw_cluster`
for cluster configuration, image push, and job submission.


.. _Singularity: https://docs.sylabs.io/guides/2.6/user-guide/index.html
.. _apptainer: https://apptainer.org/
.. _documentation: https://www.apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
