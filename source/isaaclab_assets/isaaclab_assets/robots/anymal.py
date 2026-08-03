# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0
* :obj:`ANYMAL_D_PACE_CFG`: The ANYmal-D robot with per-joint PACE actuator models

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg, PaceDCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

##
# Configuration - Actuators.
##

ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)
"""Configuration for ANYdrive 3.x with DC actuator model."""


ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    armature=0.001,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""


##
# Configuration - Articulation.
##

ANYMAL_B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-B/anymal_b.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-B robot using actuator-net."""


ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using actuator-net.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""


##
# Configuration - PACE Actuators (per-joint, identified for ANYmal-D).
##

ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=140.0,
    effort_limit=89.0,
    velocity_limit=8.5,
    stiffness={".*": 85.0},
    damping={".*": 0.6},
    dynamic_friction=0.0,
    encoder_bias=[0.0] * 12,
    max_delay=2,
)
"""Shared PACE base config for ANYdrive 3 actuators.

Per-joint variants below override ``armature``, ``viscous_friction``, ``friction`` and
``encoder_bias`` with values identified on the real ANYmal-D hardware.
"""

LF_HAA_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["LF_HAA"], armature=0.076, viscous_friction=4.9, friction=0.0054, encoder_bias=0.022, max_delay=2
)
LF_HFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["LF_HFE"], armature=0.076, viscous_friction=4.4, friction=0.021, encoder_bias=0.0057, max_delay=2
)
LF_KFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["LF_KFE"], armature=0.067, viscous_friction=5.2, friction=0.028, encoder_bias=-0.003, max_delay=2
)
RF_HAA_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["RF_HAA"], armature=0.074, viscous_friction=4.7, friction=0.0035, encoder_bias=0.011, max_delay=2
)
RF_HFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["RF_HFE"], armature=0.077, viscous_friction=4.3, friction=0.027, encoder_bias=-0.0072, max_delay=2
)
RF_KFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["RF_KFE"], armature=0.067, viscous_friction=5.3, friction=0.036, encoder_bias=0.0094, max_delay=2
)
LH_HAA_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["LH_HAA"], armature=0.089, viscous_friction=4.9, friction=0.0032, encoder_bias=-0.012, max_delay=2
)
LH_HFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["LH_HFE"], armature=0.051, viscous_friction=4.9, friction=0.024, encoder_bias=-0.0013, max_delay=2
)
LH_KFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["LH_KFE"], armature=0.064, viscous_friction=5.4, friction=0.040, encoder_bias=-0.0095, max_delay=2
)
RH_HAA_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["RH_HAA"], armature=0.079, viscous_friction=5.1, friction=0.0029, encoder_bias=-0.016, max_delay=2
)
RH_HFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["RH_HFE"], armature=0.039, viscous_friction=5.1, friction=0.013, encoder_bias=0.0043, max_delay=2
)
RH_KFE_ANYDRIVE_PACE_ACTUATOR_CFG = ANYDRIVE_PACE_ACTUATOR_CFG.replace(
    joint_names_expr=["RH_KFE"], armature=0.051, viscous_friction=5.5, friction=0.045, encoder_bias=0.0045, max_delay=2
)


ANYMAL_D_PACE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,
            ".*F_HFE": 0.4,
            ".*H_HFE": -0.4,
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
    actuators={
        "LF_HAA": LF_HAA_ANYDRIVE_PACE_ACTUATOR_CFG,
        "LF_HFE": LF_HFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "LF_KFE": LF_KFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "RF_HAA": RF_HAA_ANYDRIVE_PACE_ACTUATOR_CFG,
        "RF_HFE": RF_HFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "RF_KFE": RF_KFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "LH_HAA": LH_HAA_ANYDRIVE_PACE_ACTUATOR_CFG,
        "LH_HFE": LH_HFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "LH_KFE": LH_KFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "RH_HAA": RH_HAA_ANYDRIVE_PACE_ACTUATOR_CFG,
        "RH_HFE": RH_HFE_ANYDRIVE_PACE_ACTUATOR_CFG,
        "RH_KFE": RH_KFE_ANYDRIVE_PACE_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of the ANYmal-D robot using PACE per-joint actuator models.

Each leg joint uses a :class:`~isaaclab.actuators.PaceDCMotorCfg` parameterized with armature,
viscous and Coulomb friction, encoder bias and command delay identified on the real ANYmal-D
hardware. PACE is an analytic Lab-side actuator, so this config is intended for the PhysX
backend.
"""


##
# Configuration - Sensors.
##

ANYMAL_LIDAR_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(
    offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 1.0, 0.0))
)
"""Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
