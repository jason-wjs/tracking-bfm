"""Runner for G1 BFM wbteleop tracking."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from rsl_rl.utils import check_nan

from tracking_bfm.tasks.distillation.rl.teacher import TeacherPolicyAdapter
from tracking_bfm.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class WbTeleopTrackingRunner(MotionTrackingOnPolicyRunner):
  """Tracking runner for wbteleop PPO plus teacher-action BC."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device, registry_name=registry_name)
    self.teacher_adapter: TeacherPolicyAdapter | None = None

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    if self.teacher_adapter is None:
      self.teacher_adapter = self._build_teacher_adapter()
    self.alg.set_teacher_adapter(self.teacher_adapter)
    self._maybe_initialize_from_pretrained()
    if self.alg.pure_bc_enabled:
      return self._learn_pure_bc(num_learning_iterations, init_at_random_ep_len)
    return super().learn(num_learning_iterations, init_at_random_ep_len)

  def _begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    if hasattr(self.alg, "set_learning_iteration"):
      self.alg.set_learning_iteration(iteration)
    super()._begin_adaptive_sampling_iteration(iteration)

  def _build_teacher_adapter(self) -> TeacherPolicyAdapter:
    algorithm_cfg = self.cfg.get("algorithm", {})
    checkpoint_path = algorithm_cfg.get("teacher_checkpoint_path", "")
    if not checkpoint_path:
      raise ValueError("teacher_checkpoint_path must be provided for wbteleop training")

    teacher_task_id = algorithm_cfg.get(
      "teacher_task_id",
      "Mjlab-TrackingBFM-Flat-Unitree-G1",
    )
    teacher_obs_group = algorithm_cfg.get("teacher_obs_group", "teacher_actor")
    teacher_runner_cls = load_runner_cls(teacher_task_id) or MjlabOnPolicyRunner
    teacher_cfg = asdict(load_rl_cfg(teacher_task_id))
    teacher_cfg["obs_groups"]["actor"] = (teacher_obs_group,)

    common_step_counter = getattr(self.env.unwrapped, "common_step_counter", None)
    with self._suppress_distributed_env_for_nested_runner():
      teacher_runner = teacher_runner_cls(
        self.env,
        teacher_cfg,
        log_dir=None,
        device=str(self.device),
      )
    teacher_runner.load(checkpoint_path, map_location=str(self.device))
    if common_step_counter is not None:
      self.env.unwrapped.common_step_counter = common_step_counter

    return TeacherPolicyAdapter(
      teacher_runner.get_inference_policy(device=self.device),
      obs_group=teacher_obs_group,
      policy_input_key=teacher_obs_group,
    )

  def _learn_pure_bc(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    """Run adaptive-sampling rollouts and update only the student actor by BC."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      self._begin_adaptive_sampling_iteration(it)
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          student_actions = self.alg.act(obs)
          actions = student_actions
          if getattr(self.alg, "pure_bc_rollout", "student") == "teacher":
            if self.teacher_adapter is None:
              raise ValueError(
                "teacher_adapter must be set for teacher pure_bc_rollout"
              )
            actions = self.teacher_adapter.act_mean(obs)

          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop

      loss_dict = self.alg.update_bc_only()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=(
          self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None
        ),
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()

  def _maybe_initialize_from_pretrained(self) -> None:
    """Initialize scratch RL+BC from pure-BC actor and teacher critic checkpoints."""
    if getattr(self, "_pretrained_initialized", False):
      return
    if self.cfg.get("resume", False):
      self._pretrained_initialized = True
      return
    algorithm_cfg = self.cfg.get("algorithm", {})
    strict = bool(algorithm_cfg.get("strict_init", True))

    if getattr(self.alg, "pure_bc_enabled", False):
      if algorithm_cfg.get("init_actor_std_from_teacher", False):
        teacher_checkpoint_path = algorithm_cfg.get("teacher_checkpoint_path", "")
        if not teacher_checkpoint_path:
          raise ValueError(
            "teacher_checkpoint_path must be provided when "
            "init_actor_std_from_teacher=True"
          )
        teacher_actor_state_dict = self._load_component_state_dict(
          teacher_checkpoint_path, "actor_state_dict"
        )
        self._copy_actor_distribution_state(
          self.alg.actor.state_dict(), teacher_actor_state_dict
        )
      self._pretrained_initialized = True
      return

    actor_checkpoint_path = algorithm_cfg.get("bc_actor_checkpoint_path", "")
    if actor_checkpoint_path:
      actor_state_dict = self._load_component_state_dict(
        actor_checkpoint_path, "actor_state_dict"
      )
      self.alg.actor.load_state_dict(actor_state_dict, strict=strict)

    if algorithm_cfg.get("init_critic_from_teacher", True):
      teacher_checkpoint_path = algorithm_cfg.get("teacher_checkpoint_path", "")
      if not teacher_checkpoint_path:
        raise ValueError(
          "teacher_checkpoint_path must be provided when init_critic_from_teacher=True"
        )
      critic_state_dict = self._load_component_state_dict(
        teacher_checkpoint_path, "critic_state_dict"
      )
      self.alg.critic.load_state_dict(critic_state_dict, strict=strict)

    self._pretrained_initialized = True

  def _load_component_state_dict(
    self,
    checkpoint_path: str,
    component_key: str,
  ) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
      checkpoint_path,
      map_location=str(self.device),
      weights_only=False,
    )
    state_dict = checkpoint.get(component_key)
    if state_dict is None:
      raise KeyError(f"{component_key} not found in checkpoint: {checkpoint_path}")
    return state_dict

  def _copy_actor_distribution_state(
    self,
    target_state_dict: dict[str, torch.Tensor],
    source_state_dict: dict[str, torch.Tensor],
  ) -> None:
    copied = False
    for key in ("distribution.std_param", "distribution.log_std_param"):
      if key not in source_state_dict:
        continue
      if key not in target_state_dict:
        raise KeyError(f"{key} not found in target actor state_dict")
      source = source_state_dict[key]
      target = target_state_dict[key]
      if source.shape != target.shape:
        raise ValueError(
          f"Teacher actor distribution shape mismatch for {key}: "
          f"teacher={tuple(source.shape)}, student={tuple(target.shape)}"
        )
      target.copy_(source.to(device=target.device, dtype=target.dtype))
      copied = True

    if not copied:
      raise KeyError("teacher actor checkpoint does not contain distribution std state")

  @contextmanager
  def _suppress_distributed_env_for_nested_runner(self):
    keys = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    old_values = {key: os.environ.get(key) for key in keys}
    for key in keys:
      os.environ.pop(key, None)
    try:
      yield
    finally:
      for key, value in old_values.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value
