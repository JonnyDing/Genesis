import torch
import genesis as gs


# from legged_gym.envs.go2.go2_train import Go2Cfg


class BaseTask:
    def __init__(self, cfg, sim_params, viewer_params: dict, vis_params: dict, rigid_params: dict, show_viewer, device):
        """Abstract environment base class.

        Args:
            cfg (object): Configuration object containing environment settings.
            sim_params (object): Simulation parameters.
            viewer_params (object): Viewer-specific parameters.
            vis_params (object): Visualization-specific parameters.
            rigid_params (object): Rigid body simulation-specific parameters.
            device (str): Device for computations ('cuda' or 'cpu').
        """
        self.cfg = cfg
        if isinstance(sim_params, dict):
            self.sim_params = sim_params
        if isinstance(viewer_params, dict):
            self.viewer_params = viewer_params
        if isinstance(vis_params, dict):
            self.vis_params = vis_params
        if isinstance(rigid_params, dict):
            self.rigid_params = rigid_params
            if self.rigid_params["constraint_solver"] == "Newton":
                self.constraint_solver = gs.constraint_solver.Newton
            else:
                self.constraint_solver = gs.constraint_solver.CG
        self.show_viewer = show_viewer
        self.device = torch.device(device)
        # Extract environment-specific parameters
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # Optimization flags for PyTorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.neg_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.pos_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float
            )
        else:
            self.privileged_obs_buf = None
        # Sim Option
        self.dt = self.sim_params["dt"]
        self.substeps = self.sim_params["substeps"]
        # Viewer Option
        self.max_FPS = int(self.viewer_params["max_FPS"] / self.dt)
        self.camera_pos = self.viewer_params["camera_pos"]
        self.camera_lookat = self.viewer_params["camera_lookat"]
        self.camera_fov = self.viewer_params["camera_fov"]
        # Vis Option
        self.n_rendered_envs = self.vis_params["n_rendered_envs"]
        # Rigid Option
        # self.constraint_solver = self.rigid_params["constraint_solver"]
        self.enable_collision = self.rigid_params["enable_collision"]
        self.enable_joint_limit = self.rigid_params["enable_joint_limit"]

        self.extras = {}

        # Initialize physics engine and scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.max_FPS,
                camera_pos=self.camera_pos,
                camera_lookat=self.camera_lookat,
                camera_fov=self.camera_fov,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=self.n_rendered_envs,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=self.constraint_solver,
                enable_collision=self.enable_collision,
                enable_joint_limit=self.enable_joint_limit,
            ),
            show_viewer=self.show_viewer,
        )
        self.robot = None  # Placeholder for robot initialization in subclasses

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError
