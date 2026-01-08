# import chex
import jax
import jax.numpy as jnp
from flax import nnx

from .ac import Actor, Critic


def mse_loss(predictions, targets):
    return jnp.mean(jnp.square(predictions - targets))


# REINFORCE
def reinforce_loss_fn(
    critic: Critic,
    statics: list[jax.Array],
    dynamics: list[jax.Array],
    rewards: list[jax.Array],
    dones: list[jax.Array],
    logprobs: list[jax.Array],
    action_probs: list[jax.Array],
    gamma: float = 0.99,
):
    Vs = [
        critic(statics[t], dynamics[t], action_probs[t])  # (B, 1)
        for t in range(len(rewards))
    ]
    R = jnp.hstack(rewards)  # (B, N)
    D = jnp.hstack(dones).astype(jnp.float32)  # (B, N)
    discounts = jnp.pow(gamma, jnp.arange(len(rewards)))
    G0 = jnp.sum(discounts * R * (1 - D), axis=1)
    V0 = Vs[0].squeeze()
    actor_loss = (G0 - V0) * logprobs[0]  # (B)
    for t in range(1, len(rewards)):
        Vt = Vs[t].squeeze()
        Gt = jnp.sum(discounts[:-t] * R[:, t:] * (1 - D[:, t:]), axis=1)  # (B)
        actor_loss += (Gt - Vt) * logprobs[t]  # (B)
    actor_loss = -jnp.mean(actor_loss)
    V = jnp.hstack(Vs)  # (B, N)
    critic_loss = mse_loss(V, R)  # (1)

    return actor_loss, critic_loss


def reinforce_update(
    actor: Actor,
    critic: Critic,
    batch: dict[str, list[jax.Array]],
    actor_optim: nnx.Optimizer,
    critic_optim: nnx.Optimizer,
    gamma: float = 0.99,
):
    statics = batch["statics"]
    dynamics = batch["dynamics"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    logprobs = batch["logprobs"]
    action_probs = batch["action_probs"]
    grad_fn = nnx.value_and_grad(
        reinforce_loss_fn,
        argnums=(5, 0),  # w.r.t. logprobs and critic respectively
        has_aux=True,
    )
    (actor_loss, critic_loss), (actor_grad, critic_grad) = grad_fn(
        critic,
        statics,
        dynamics,
        rewards,
        dones,
        logprobs,
        action_probs,
        gamma,
    )
    actor_optim.update(actor.ptrnet, actor_grad)
    critic_optim.update(critic, critic_grad)

    return actor_loss, critic_loss


def eval_step(
    actor: Actor,
    critic: Critic,
    batch: dict[str, list[jax.Array]],
    metrics: nnx.Metric,
    gamma: float = 0.99,
):
    statics = batch["statics"]
    dynamics = batch["dynamics"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    logprobs = batch["logprobs"]
    action_probs = batch["action_probs"]
    (actor_loss, critic_loss) = reinforce_loss_fn(
        critic,
        statics,
        dynamics,
        rewards,
        dones,
        logprobs,
        action_probs,
        gamma,
    )
    metrics.update(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
    )


def rollout():
    pass
