import copy

import torch
import torch.nn.functional as F
from network import StateCritic, VRPActor
from utils import decode_states


class REINFORCE:
    def __init__(
        self,
        actor: VRPActor,
        critic: StateCritic,
        *,
        gamma: float = 0.95,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        max_grad_norm: float = 2.0,
        **kwargs,
    ) -> None:
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        statics: list[torch.Tensor],
        dynamics: list[torch.Tensor],
        rewards: list[torch.Tensor],
        dones: list[torch.Tensor],
        logprobs: list[torch.Tensor],
        action_probs: list[torch.Tensor],
        *,
        device=None,
    ):
        # statics : list[torch.Tensor (B, _, N)]
        # dynamics : list[torch.Tensor (B, _, N)]
        # rewards : list[torch.Tensor (B, 1)]
        # dones : list[torch.Tensor (B, 1)]
        # logprobs : list[torch.Tensor (B)]
        # action_probs : list[torch.Tensor (B, N)]

        # V0 = self.critic(statics[0], dynamics[0], action_probs[0].detach())  # (B, 1)
        # R = torch.hstack(rewards)  # (B, N)
        # Ret = R.sum(1, keepdim=True) # (B, 1)
        # Adv = (Ret - V0).detach() # (B, 1)
        # Lp = torch.vstack(logprobs).sum(0) # (B)
        # actor_loss = -torch.mean(Adv.squeeze() * Lp) # (1)
        # critic_loss = F.mse_loss(V0, Ret)  # (1)

        Vs = [
            self.critic(
                statics[t].detach(),
                dynamics[t].detach(),
                action_probs[t].detach(),
            )  # (B, 1)
            for t in range(len(rewards))
        ]
        R = torch.hstack(rewards)  # (B, N)
        D = torch.hstack(dones).to(torch.float32)  # (B, N) : done = 1.0, not done = 0.0
        discounts = torch.pow(  # (N)
            self.gamma,
            torch.arange(len(rewards), device=device),
        )
        G0 = torch.sum(discounts * R * (1 - D), dim=1)  # (B)
        V0 = Vs[0].squeeze().detach()  # (B)
        actor_loss = (G0 - V0) * logprobs[0]  # (B)
        for t in range(1, len(rewards)):
            Vt = Vs[t].squeeze().detach()  # (B)
            Gt = torch.sum(discounts[:-t] * R[:, t:] * (1 - D[:, t:]), dim=1)  # (B)
            actor_loss += (Gt - Vt) * logprobs[t]  # (B)
        actor_loss = -torch.mean(actor_loss)  # (1)
        V = torch.hstack(Vs)  # (B, N)
        critic_loss = F.mse_loss(V, R.detach())  # (1)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }


class PPO:
    def __init__(
        self,
        actor: VRPActor,
        critic: StateCritic,
        *,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        lmbda: float = 0.01,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        max_grad_norm: float = 2.0,
        **kwargs,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.actor = actor
        self.critic = critic
        self.old_actor = copy.deepcopy(actor)
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        rewards: list[torch.Tensor],
        xstates: list[torch.Tensor],
        dones: list[torch.Tensor],
        logprobs: list[torch.Tensor],
        action_probs: list[torch.Tensor],
        *,
        epochs: int = 5,
        minibatchsize: int = 128,
        device=None,
        generator: torch.Generator | None = None,
    ):
        # states : list[torch.Tensor (B, _, N)]
        # action : list[torch.Tensor (B, _, N)]
        # rewards : list[torch.Tensor (B, 1)]
        # dones : list[torch.Tensor (B, 1)]
        # logprobs : list[torch.Tensor (B)]
        # action_probs : list[torch.Tensor (B, N)]

        statics, dynamics, x0s, masks, hxs = decode_states(states)
        bstatics = torch.cat(statics).detach()
        bdynamics = torch.cat(dynamics).detach()
        bx0s = torch.cat(x0s).detach()
        bmasks = torch.cat(masks).detach()
        bactions = torch.cat(actions).detach()  # (B, 1)
        bold_logprobs = torch.cat(logprobs).detach().squeeze()  # (B)
        bold_actprobs = torch.cat(action_probs).detach()  # (B, N)
        brewards = torch.cat(rewards).detach().squeeze()  # (B)
        bdones = torch.cat(dones).float().detach().squeeze()  # (B)

        hx_sample = next((h for h in hxs if h is not None), None)
        if hx_sample is not None:
            hx_zero = torch.zeros_like(hx_sample, device=device)
            sanitized_hxs = [h if h is not None else hx_zero for h in hxs]
            bhxs = torch.cat(sanitized_hxs).detach()
        else:
            bhxs = None

        with torch.no_grad():
            values = self.critic(bstatics, bdynamics, bold_actprobs).squeeze()
            advantages = torch.zeros_like(brewards, device=device)
            returns = torch.zeros_like(brewards, device=device)
            gae = 0.0
            xvalue = 0.0
            for t in reversed(range(len(rewards))):
                if bdones[t]:
                    gae = 0.0
                    xvalue = 0.0
                delta = brewards[t] + self.gamma * xvalue - values[t]
                gae = delta + self.gamma * self.lmbda * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
                xvalue = values[t]

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        datasize = bstatics.size(0)
        indices = torch.arange(datasize, device=device)

        results = {"actor_loss": [], "critic_loss": []}
        for _ in range(epochs):
            idxs = indices[torch.randperm(datasize, device=device, generator=generator)]
            for start in range(0, datasize, minibatchsize):
                mbidx = idxs[start : start + minibatchsize]
                mbstatics = bstatics[mbidx]
                mbdynamics = bdynamics[mbidx]
                mbx0s = bx0s[mbidx]
                mbmasks = bmasks[mbidx]
                mbhxs = bhxs[mbidx] if bhxs is not None else None
                mbactions = bactions[mbidx]
                mboldlogprobs = bold_logprobs[mbidx]
                mboldactprobs = bold_actprobs[mbidx]
                mbadvantages = advantages[mbidx]
                mbreturns = returns[mbidx]

                _, _, mbnew_actprobs, *_ = self.actor.action(
                    static=mbstatics,
                    dynamic=mbdynamics,
                    x0=mbx0s,
                    hx=mbhxs,
                    mask=mbmasks,
                    device=device,
                    generator=generator,
                )

                new_probs = torch.gather(
                    input=mbnew_actprobs,
                    dim=1,
                    index=mbactions,
                ).squeeze()
                new_logprobs = torch.log(new_probs + 1e-9)
                ratios = torch.exp(new_logprobs - mboldlogprobs)
                surr1 = ratios * mbadvantages
                surr2 = torch.mul(
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon),
                    mbadvantages,
                )
                clipped_objective = torch.mean(torch.min(surr1, surr2))
                entropy = torch.mean(
                    torch.distributions.Categorical(probs=mbnew_actprobs).entropy()
                )
                actor_loss = -(clipped_objective + self.lmbda * entropy)
                # critic loss
                V = self.critic(mbstatics, mbdynamics, mboldactprobs).squeeze()
                critic_loss = F.mse_loss(V, mbreturns)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optim.step()

                results["actor_loss"].append(actor_loss.detach())
                results["critic_loss"].append(critic_loss.detach())

        actor_losses = torch.hstack(results["actor_loss"])
        critic_losses = torch.hstack(results["critic_loss"])
        return {
            "actor_loss": torch.mean(actor_losses),
            "critic_loss": torch.mean(critic_losses),
        }

    def synchronize_old_actor(self):
        self.old_actor = copy.deepcopy(self.actor)
