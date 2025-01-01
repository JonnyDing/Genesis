import genesis as gs
from legged_gym.envs.base.base_config import BaseConfig


class Go2Cfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        # termination
        termination_if_roll_greater_than = 10  # degree
        termination_if_pitch_greater_than = 10
        simulate_action_latency = True
        dof_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

    class asset:
        file = "urdf/go2/urdf/go2.urdf"

    class terrain:
        file = "urdf/plane/plane.urdf"
        fixed = True

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 4.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x_range = [0.5, 0.5]  # min max [m/s]
            lin_vel_y_range = [0.0, 0.0]  # min max [m/s]
            ang_vel_range = [0.0, 0.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        base_init_pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # joint/link names
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control:
        # PD Drive parameters:
        stiffness = {
            "FL_hip_joint": 20.0,
            "FR_hip_joint": 20.0,
            "RL_hip_joint": 20.0,
            "RR_hip_joint": 20.0,
            "FL_thigh_joint": 20.0,
            "FR_thigh_joint": 20.0,
            "RL_thigh_joint": 20.0,
            "RR_thigh_joint": 20.0,
            "FL_calf_joint": 20.0,
            "FR_calf_joint": 20.0,
            "RL_calf_joint": 20.0,
            "RR_calf_joint": 20.0,
        }
        damping = {
            "FL_hip_joint": 0.5,
            "FR_hip_joint": 0.5,
            "RL_hip_joint": 0.5,
            "RR_hip_joint": 0.5,
            "FL_thigh_joint": 0.5,
            "FR_thigh_joint": 0.5,
            "RL_thigh_joint": 0.5,
            "RR_thigh_joint": 0.5,
            "FL_calf_joint": 0.5,
            "FR_calf_joint": 0.5,
            "RL_calf_joint": 0.5,
            "RR_calf_joint": 0.5,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class rewards:
        tracking_sigma = 0.25
        base_height_target = 0.3
        feet_height_target = 0.075

        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.2
            lin_vel_z = -1.0
            base_height = -50.0
            action_rate = -0.005
            similar_to_default = -0.1

        only_positive_rewards = (
            True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        )
        max_contact_force = 100.0  # forces above this value are penalized

    class sim:
        substeps = 2
        dt = 0.02

        class viewer:
            max_FPS = 0.5  # 0.05/dt
            camera_pos = (2.0, 0.0, 2.5)
            camera_lookat = (0.0, 0.0, 0.5)
            camera_fov = 40

        class rigid_params:
            constraint_solver = "Newton"
            enable_collision = True
            enable_joint_limit = True

        class vis_params:
            n_rendered_envs = 1


class Go2CfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        activation = "elu"
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "go2-walking"
        run_name = ""
        # load and resume
        resume = False
        resume_path = None
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        record_interval = -1
        resume_path = ""  # updated from load_run and chkpt
