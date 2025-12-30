import chex
import jax
import jax.numpy as jnp
from flax import nnx

from ptrnet.network import DEFAULT_INITIALIZER, PointerNetwork


class Actor:
    """Actor that uses a PointerNetwork to select actions, supporting both greedy and sampling-based policies."""

    def __init__(
        self,
        ptrnet: PointerNetwork,
        *,
        greedy: bool = False,  # set to true for inference
    ) -> None:
        self.ptrnet = ptrnet
        self.greedy = greedy

    def action(
        self,
        static: jax.Array,  # (B, 2, N)
        dynamic: jax.Array,  # (B, 2, N)
        x0: jax.Array,  # (B, 2, 1)
        *,
        mask: jax.Array | None = None,  # (B, N)
        carry: tuple[jax.Array, jax.Array] | None = None,
        rngs: nnx.Rngs,
    ):
        if mask is None:
            batchsize, _, seqsize = static.shape
            mask = jnp.ones((batchsize, seqsize))
        logits, carry = self.ptrnet(
            static,
            dynamic,
            x0,
            carry=carry,
            rngs=rngs,
            training=not self.greedy,
        )
        logits = logits + jnp.log(mask)
        action_probs = nnx.softmax(logits)
        if not self.greedy:
            ptr = jax.random.categorical(
                logits=logits, replace=True, key=rngs.default()
            )
        else:
            ptr = jnp.argmax(logits, axis=-1, keepdims=True)
        probs = jnp.take_along_axis(action_probs, ptr.reshape(-1, 1))
        logprob = jnp.log(probs)
        return ptr, logprob, action_probs, carry


class Critic(nnx.Module):
    """Critic network that evaluates the value of a given state, incorporating static and dynamic embeddings and attention over action probabilities."""

    def __init__(
        self,
        static_dim: int,  # M_sta
        dynamic_dim: int,  # M_dyn
        hidden_dim: int,  # H_sta, H_dyn
        *,
        param_init: nnx.Initializer = DEFAULT_INITIALIZER,
        rngs: nnx.Rngs,
    ) -> None:
        self.staticembed = nnx.Conv(
            in_features=static_dim,  # M_sta
            out_features=hidden_dim,  # H_sta
            kernel_size=(1,),
            kernel_init=param_init,
            rngs=rngs,
        )
        self.dynamicembed = nnx.Conv(
            in_features=dynamic_dim,  # M_dyn
            out_features=hidden_dim,  # H_dyn
            kernel_size=(1,),
            kernel_init=param_init,
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=2 * hidden_dim,  # H_sta + H_dyn
            out_features=hidden_dim,  # H_den1
            kernel_init=param_init,
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            in_features=hidden_dim,  # H_den1
            out_features=1,
            kernel_init=param_init,
            rngs=rngs,
        )
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim

    def __call__(
        self,
        static: jax.Array,  # (B, M_sta, N)
        dynamic: jax.Array,  # (B, M_dyn, N)
        action_probs: jax.Array,  # (B, N)
    ):
        chex.assert_shape(static, (None, self.static_dim, ...))
        chex.assert_shape(dynamic, (None, self.dynamic_dim, ...))
        chex.assert_rank(action_probs, 2)
        chex.assert_equal_shape_suffix((static, dynamic, action_probs), suffix_len=1)

        static = jnp.transpose(static, (0, 2, 1))  # (B, N, 2)
        dynamic = jnp.transpose(dynamic, (0, 2, 1))  # (B, N, 2)
        static_hidden = self.staticembed(static)  # (B, N, H_sta)
        dynamic_hidden = self.dynamicembed(dynamic)  # (B, N, H_dyn)
        hidden = jnp.concat(  # (B, N, H_sta + H_dyn)
            (static_hidden, dynamic_hidden), axis=-1
        )

        probs = action_probs[:, None]  # (B, 1, N)
        weighted_state = probs @ hidden  # (B, 1, H_sta + H_dyn)
        weighted_state = weighted_state.squeeze(1)  # (B, H_sta + H_dyn)

        output = nnx.relu(self.dense1(weighted_state))
        output = self.dense2(output)
        return output
