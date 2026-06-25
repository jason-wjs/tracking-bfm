"""PPO variant for wbteleop tracking with teacher-action MSE."""

from __future__ import annotations

import math
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from tracking_bfm.tasks.distillation.rl.teacher import TeacherPolicyAdapter


def cosine_bc_weight(
  iteration: int,
  *,
  start: float,
  end: float,
  decay_steps: int,
) -> float:
  """Cosine decay from start to end, clamped at end after decay_steps."""
  if decay_steps <= 0:
    raise ValueError("bc_decay_steps must be positive")
  progress = min(max(int(iteration), 0), int(decay_steps)) / float(decay_steps)
  return float(end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class WbTeleopPPO(PPO):
  """PPO with an additional scheduled teacher-action MSE loss."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    *args,
    teacher_task_id: str = "Mjlab-TrackingBFM-Flat-Unitree-G1",
    teacher_checkpoint_path: str = "",
    teacher_obs_group: str = "teacher_actor",
    bc_weight_start: float = 0.5,
    bc_weight_end: float = 0.1,
    bc_decay_steps: int = 10_000,
    pure_bc_enabled: bool = False,
    pure_bc_weight: float = 1.0,
    pure_bc_rollout: str = "student",
    bc_actor_checkpoint_path: str = "",
    init_actor_std_from_teacher: bool = False,
    init_critic_from_teacher: bool = True,
    strict_init: bool = True,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.teacher_task_id = teacher_task_id
    self.teacher_checkpoint_path = teacher_checkpoint_path
    self.teacher_obs_group = teacher_obs_group
    self.bc_weight_start = float(bc_weight_start)
    self.bc_weight_end = float(bc_weight_end)
    self.bc_decay_steps = int(bc_decay_steps)
    self.pure_bc_enabled = bool(pure_bc_enabled)
    self.pure_bc_weight = float(pure_bc_weight)
    if pure_bc_rollout not in {"student", "teacher"}:
      raise ValueError("pure_bc_rollout must be one of {'student', 'teacher'}")
    self.pure_bc_rollout = pure_bc_rollout
    self.bc_actor_checkpoint_path = bc_actor_checkpoint_path
    self.init_actor_std_from_teacher = bool(init_actor_std_from_teacher)
    self.init_critic_from_teacher = bool(init_critic_from_teacher)
    self.strict_init = bool(strict_init)
    self.teacher_adapter: TeacherPolicyAdapter | None = None
    self.current_learning_iteration = 0

  def set_teacher_adapter(self, teacher_adapter: TeacherPolicyAdapter) -> None:
    self.teacher_adapter = teacher_adapter

  def set_learning_iteration(self, iteration: int) -> None:
    self.current_learning_iteration = int(iteration)

  def _current_bc_weight(self) -> float:
    return cosine_bc_weight(
      self.current_learning_iteration,
      start=self.bc_weight_start,
      end=self.bc_weight_end,
      decay_steps=self.bc_decay_steps,
    )

  def update(self) -> dict[str, float]:
    """Run PPO update with scheduled teacher-action MSE."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_bc_mse = 0.0
    mean_bc_loss = 0.0
    bc_weight = self._current_bc_weight()
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

    if self.teacher_adapter is None:
      raise ValueError("teacher_adapter must be set before WbTeleopPPO.update()")

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (  # type: ignore
            batch.advantages.std() + 1e-8
          )

      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"],
          obs=batch.observations,
          actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(  # type: ignore
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(
            obs=batch.observations, actions=None, env=self.symmetry["_env"]
          )

        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      rnd_loss = (
        self.rnd.compute_loss(batch.observations[:original_batch_size])
        if self.rnd
        else None
      )

      student_action_mean = self.actor(batch.observations[:original_batch_size])
      with torch.no_grad():
        teacher_action_mean = self.teacher_adapter.act_mean(
          batch.observations[:original_batch_size]
        )
      if student_action_mean.shape != teacher_action_mean.shape:
        raise ValueError(
          "Teacher and student action shapes must match: "
          f"student={tuple(student_action_mean.shape)}, "
          f"teacher={tuple(teacher_action_mean.shape)}"
        )
      bc_mse = F.mse_loss(student_action_mean, teacher_action_mean)
      bc_loss = bc_weight * bc_mse
      loss = loss + bc_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        assert rnd_loss is not None
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_bc_mse += bc_mse.item()
      mean_bc_loss += bc_loss.item()
      if mean_rnd_loss is not None:
        assert rnd_loss is not None
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_bc_mse /= num_updates
    mean_bc_loss /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    self.storage.clear()

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "bc_mse": mean_bc_mse,
      "bc_weight": float(bc_weight),
      "bc_loss": mean_bc_loss,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    return loss_dict

  def update_bc_only(self) -> dict[str, float]:
    """Update only the actor with teacher-action MSE from collected rollout obs."""
    if self.teacher_adapter is None:
      raise ValueError(
        "teacher_adapter must be set before WbTeleopPPO.update_bc_only()"
      )

    observations = self.storage.observations.flatten(0, 1)
    batch_size = observations.batch_size[0]
    if batch_size == 0:
      raise ValueError("rollout storage must contain at least one observation")

    num_mini_batches = max(1, min(self.num_mini_batches, batch_size))
    mini_batch_size = math.ceil(batch_size / num_mini_batches)
    mse_total = 0.0
    loss_total = 0.0
    grad_norm_total = 0.0
    updates = 0

    self.actor.train()
    for _ in range(self.num_learning_epochs):
      permutation = torch.randperm(batch_size, device=self.device)
      for start in range(0, batch_size, mini_batch_size):
        batch_idx = permutation[start : start + mini_batch_size]
        batch_obs = observations[batch_idx]

        if hasattr(self.actor, "update_normalization"):
          self.actor.update_normalization(batch_obs)

        student_action_mean = self.actor(batch_obs)
        with torch.no_grad():
          teacher_action_mean = self.teacher_adapter.act_mean(batch_obs)
        if student_action_mean.shape != teacher_action_mean.shape:
          raise ValueError(
            "Teacher and student action shapes must match: "
            f"student={tuple(student_action_mean.shape)}, "
            f"teacher={tuple(teacher_action_mean.shape)}"
          )

        mse_loss = F.mse_loss(student_action_mean, teacher_action_mean)
        loss = self.pure_bc_weight * mse_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        grad_norm = nn.utils.clip_grad_norm_(
          self.actor.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        mse_total += float(mse_loss.item())
        loss_total += float(loss.item())
        grad_norm_total += float(grad_norm.item())
        updates += 1

    self.storage.clear()
    return {
      "pure_bc_mse": mse_total / updates,
      "pure_bc_loss": loss_total / updates,
      "pure_bc_weight": float(self.pure_bc_weight),
      "pure_bc_grad_norm": grad_norm_total / updates,
    }

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "WbTeleopPPO":
    """Construct the PPO algorithm, preserving teacher-only observation groups."""
    alg_class: type[WbTeleopPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

    default_sets = ["actor", "critic"]
    if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )

    alg: WbTeleopPPO = alg_class(
      actor,
      critic,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    return alg

  def reduce_parameters(self) -> None:
    """Collect gradients from all GPUs and average them."""
    all_params = chain(self.actor.parameters(), self.critic.parameters())
    if self.rnd:
      all_params = chain(all_params, self.rnd.parameters())
    all_params = list(all_params)
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(
          all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel
