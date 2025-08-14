from typing import Any, Mapping, Type, Union

import copy

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
from skrl.utils import set_seed
import gymnasium as gym
import numpy as np
import torch
import rospy
from geometry_msgs.msg import PointStamped
from mag_msgs.msg import FieldStamped



class Runner:
    def __init__(self) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        :param env: Environment to train on
        :param cfg: Runner configuration
        """
        cfg_path = rospy.get_param("/skrl_controller/cfg_path")
        self._cfg = self.load_cfg_from_yaml(cfg_path)
        if not self._cfg:
            raise ValueError("Failed to load configuration from the specified YAML file")
        self._cfg["trainer"]["close_environment_at_exit"] = False
        self._cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
        self._cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

        observation_space = rospy.get_param("/skrl_controller/observation_space")
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space,), dtype=np.float32)
        action_space = rospy.get_param("/skrl_controller/action_space")
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_space,), dtype=np.float32)

        max_angle_divider =rospy.get_param("/skrl_controller/max_angle_divider_field")
        self.max_angle_field = torch.pi / max_angle_divider  # maximum angle for the catheter
        max_angle_divider_goal = rospy.get_param("/skrl_controller/max_angle_divider_goal")
        self.max_angle_goal = torch.pi / max_angle_divider_goal  # maximum angle for the

        self.goal_radius = rospy.get_param("/skrl_controller/goal_radius")  # radius for the goal position
        self.field_magnitude = rospy.get_param("/skrl_controller/field_magnitude")  # norm of the magnetic field
        self.max_goal_count = rospy.get_param("/skrl_controller/goal_count")

        self.goal = self.compute_goal_position()  # compute the goal position for the environment

        self.last_x = 0.0
        self.last_y = 0.0
        self.last_action = 0.0
        self.goal_count = 0
        self.goal_position_switches = 0  # Counter for goal position switches

        self.checkpoint_path = rospy.get_param("/skrl_controller/checkpoint_path")

        rospy.loginfo("observation_space: %s", observation_space)
        rospy.loginfo("action_space: %s", action_space)

        rospy.loginfo("max_angle_field: %s", rospy.get_param("/skrl_controller/max_angle_divider_field"))
        rospy.loginfo("max_angle_goal: %s", rospy.get_param("/skrl_controller/max_angle_divider_goal"))
        rospy.loginfo("goal_radius: %s", self.goal_radius)
        rospy.loginfo("field_magnitude: %s", self.field_magnitude)
        rospy.loginfo("checkpoint_path: %s", self.checkpoint_path)

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._cfg["agent"]["rewards_shaper"] = None  # FIXME: avoid 'dictionary changed size during iteration'

        self._models = self._generate_models(
                                             observation_space=observation_space, 
                                             action_space=action_space, 
                                             device="cuda", 
                                             cfg=copy.deepcopy(self._cfg))
        
        self._agent = self._generate_agent(
                                           observation_space=observation_space,
                                           action_space=action_space,
                                           device="cuda",
                                           cfg=copy.deepcopy(self._cfg),
                                           models=self._models)
        
        self._agent.load(self.checkpoint_path)
        self._agent.set_running_mode("eval")
        self.position_subscriber = rospy.Subscriber('hsv_tracker/position', PointStamped, self.position_callback)
        self.field_publisher = rospy.Publisher('skrl_controller/field', FieldStamped, queue_size=1)
        self.goal_publisher = rospy.Publisher('skrl_controller/goal', PointStamped, queue_size=1)

        self.publish_goal(self.goal)  # publish the goal position

    @property
    def agent(self) -> Agent:
        """Agent instance"""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file

        :param path: File path

        :return: Loaded configuration, or an empty dict if an error has occurred
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def position_callback(self, msg: PointStamped):
        """Callback function for position updates from the HSV tracker."""
        # Extract position data from the message
        x = msg.point.x
        y = msg.point.y

        x *= 1e2
        y *= 1e2
        x_goal = self.goal[0].item() * 1e2
        y_goal = self.goal[1].item() * 1e2
        obs = torch.tensor([x, self.last_x, x_goal, self.last_action], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        with torch.inference_mode():
            out = self.agent.act(states=obs, timestep=0, timesteps=0)
            actions = out[-1].get("mean_actions", out[0])
            field = self.compute_field_from_angles(actions)
        self.publish_field(field)  # Publish the computed magnetic field
        self.last_x = x
        self.last_y = y
        self.last_action = actions.item()  # Store the last action taken

        # if np.abs(x - x_goal) < (self.goal_radius *1e2):
        #     self.goal = self.compute_goal_position()  # Recompute the goal position
        if self.goal_count < self.max_goal_count:
            self.goal_count += 1
        else:
            self.goal_count = 0
            self.goal = self.compute_goal_position()
        self.publish_goal(self.goal)  # Publish the goal position

    def _component(self, name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component
        """
        component = None
        name = name.lower()
        # model
        if name == "shared":
            from skrl.utils.model_instantiators.torch import shared_model as component
        # memory
        elif name == "randommemory":
            from skrl.memories.torch import RandomMemory as component

        elif name in ["ppo", "ppo_default_config"]:
            from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

            component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO

        # trainer
        elif name == "sequentialtrainer":
            from skrl.trainers.torch import SequentialTrainer as component

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components

        :param cfg: A configuration dictionary

        :return: Updated dictionary
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_state_preprocessor",
            "noise",
            "smooth_regularization_noise",
        ]

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(
        self, observation_space: int, action_space: int, device: str, cfg: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param env: Wrapped environment
        :param cfg: A configuration dictionary

        :return: Model instances
        """
        device = device
        possible_agents = ["agent"]
        observation_spaces = {"agent": observation_space}
        action_spaces = {"agent": action_space}


        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
            # non-shared models
            roles = list(models_cfg.keys())
            if len(roles) != 2:
                raise ValueError(
                    "Runner currently only supports shared models, made up of exactly two models. "
                    "Set 'separate' field to True to create non-shared models for the given cfg"
                )
            # get shared model structure and parameters
            structure = []
            parameters = []
            for role in roles:
                # get instantiator function and remove 'class' key
                model_structure = models_cfg[role].get("class")
                if not model_structure:
                    raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                del models_cfg[role]["class"]
                structure.append(model_structure)
                parameters.append(self._process_cfg(models_cfg[role]))
            model_class = self._component("Shared")
            # print model source
            source = model_class(
                observation_space=observation_spaces[agent_id],
                action_space=action_spaces[agent_id],
                device=device,
                structure=structure,
                roles=roles,
                parameters=parameters,
                return_source=True,
            )
            # instantiate model
            models[agent_id][roles[0]] = model_class(
                observation_space=observation_spaces[agent_id],
                action_space=action_spaces[agent_id],
                device=device,
                structure=structure,
                roles=roles,
                parameters=parameters,
            )
            models[agent_id][roles[1]] = models[agent_id][roles[0]]

        # initialize lazy modules' parameters
        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role)

        return models

    def _generate_agent(
        self,
        observation_space: int,
        action_space: int,
        device: str,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """
        num_envs = 1
        device = device
        possible_agents = ["agent"]
        observation_spaces = {"agent": observation_space}
        action_spaces = {"agent": action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if not agent_class:
            raise ValueError(f"No 'class' field defined in 'agent' cfg")

        # check for memory configuration (backward compatibility)
        if not "memory" in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # single-agent configuration and instantiation
        if agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise", None):
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise", None):
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)
    
    def compute_field_from_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """Compute the magnetic field from angles."""
        angles = self.max_angle_field * torch.tanh(angles)
        x = self.field_magnitude * torch.sin(angles)
        y = self.field_magnitude * torch.cos(angles)
        z = torch.zeros_like(x)
        return torch.stack((x, y, z), dim=-1).squeeze(0).squeeze(0)
    
    def compute_goal_position(self, length_catheter: float = 0.0434) -> torch.Tensor:
        """Compute the goal position for the environment."""
        angles = torch.rand(1, device="cuda:0") *2*self.max_angle_goal -self.max_angle_goal
        x = length_catheter * torch.sin(angles)
        y = length_catheter * torch.cos(angles) - 0.0203
        z = torch.ones_like(x) * 0.0013
        return torch.stack((x, y, z), dim=-1).squeeze(0)
    
    def compute_goal_position2(self, length_catheter: float = 0.0434) -> torch.Tensor:
        """Compute the goal position for the environment."""
        if self.goal_position_switches % 2 == 0:
            angles = torch.tensor([0.5], dtype=torch.float32) * 2*self.max_angle_goal -self.max_angle_goal
        else:
            angles = torch.tensor([0.25], dtype=torch.float32) * 2*self.max_angle_goal -self.max_angle_goal
        self.goal_position_switches += 1
        x = length_catheter * torch.sin(angles)
        y = length_catheter * torch.cos(angles) - 0.0203
        z = torch.ones_like(x) * 0.0013
        return torch.stack((x, y, z), dim=-1).squeeze(0)
    
    def publish_field(self, field: torch.Tensor):
        """Publish the magnetic field to the ROS topic."""
        field_msg = FieldStamped()
        field_msg.header.stamp = rospy.Time.now()
        field_msg.header.frame_id = 'mns'
        field_msg.field.vector.x = field[0].item()
        field_msg.field.vector.y = field[1].item()
        field_msg.field.vector.z = field[2].item()
        field_msg.field.position.x = 0.0
        field_msg.field.position.y = 0.0
        field_msg.field.position.z = 0.0
        self.field_publisher.publish(field_msg)

    def publish_goal(self, goal: torch.Tensor):
        """Publish the goal position to the ROS topic."""
        goal_msg = PointStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'mns'
        goal_msg.point.x = goal[0].item()
        goal_msg.point.y = goal[1].item()
        goal_msg.point.z = goal[2].item()
        self.goal_publisher.publish(goal_msg)
        

if __name__ == "__main__":
    rospy.init_node("skrl_controller")
    runner = Runner()
    rospy.loginfo("SKRL Controller initialized successfully.")
    
    # Keep the node running
    rospy.spin()