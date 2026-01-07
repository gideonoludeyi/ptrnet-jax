import chex
import jax
import jax.numpy as jnp
from flax import nnx

DEFAULT_INITIALIZER = nnx.initializers.xavier_uniform()


class Embed(nnx.Module):
    """Embeds input data using a convolutional layer."""

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 128,
        *,
        param_init: nnx.Initializer = DEFAULT_INITIALIZER,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_features=input_dim,
            out_features=output_dim,
            kernel_size=(1,),
            kernel_init=param_init,
            rngs=rngs,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x: jax.Array):
        chex.assert_shape(x, (None, None, self.input_dim))

        return self.conv(x)


class RNNDecoder(nnx.Module):
    """RNN-based decoder using an LSTM cell with dropout."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        dropout_rate: float = 0,
        *,
        param_init: nnx.Initializer = DEFAULT_INITIALIZER,
        rngs: nnx.Rngs,
    ) -> None:
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.rnn = nnx.RNN(
            cell=nnx.LSTMCell(
                in_features=input_dim,
                hidden_features=hidden_dim,
                kernel_init=param_init,
                recurrent_kernel_init=param_init,
                rngs=rngs,
            ),
            return_carry=True,
            rngs=rngs,
        )
        self.input_dim = input_dim

    def __call__(
        self,
        input: jax.Array,
        *,
        carry: tuple[jax.Array, jax.Array] | None = None,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
    ):
        chex.assert_shape(input, (None, None, self.input_dim))

        input = self.dropout(input, deterministic=not training, rngs=rngs)
        carry, output = self.rnn(
            input,
            initial_carry=carry,
            return_carry=True,
            rngs=rngs,
        )
        return output, carry

    def init(
        self,
        input_shape: tuple[int, ...],
        time_major: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        ndim = len(input_shape)
        # https://github.com/google/flax/blob/v0.12.2/flax/nnx/nn/recurrent.py#L831
        time_axis = 0 if time_major else ndim - (self.rnn.cell.num_feature_axes + 1)
        # https://github.com/google/flax/blob/v0.12.2/flax/nnx/nn/recurrent.py#L860-L862
        return self.rnn.cell.initialize_carry(
            input_shape[:time_axis] + input_shape[time_axis + 1 :], rngs=rngs
        )

    def h(self, carry: tuple[jax.Array, jax.Array]):
        return carry[1]


class Attention(nnx.Module):
    """Attention mechanism for combining static, dynamic, and decoder embeddings."""

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        static_embed_dim: int = 128,
        dynamic_embed_dim: int = 128,
        decoder_dim: int = 128,
        param_init: nnx.Initializer = DEFAULT_INITIALIZER,
        rngs: nnx.Rngs,
    ) -> None:
        self.static_embed_dim = static_embed_dim
        self.dynamic_embed_dim = dynamic_embed_dim
        self.decoder_dim = decoder_dim
        self.va = nnx.Param(
            param_init(
                key=rngs.params(),
                shape=(1, hidden_dim),
                dtype=jnp.float32,
            ),
        )
        self.Wa = nnx.Param(
            param_init(
                key=rngs.params(),
                shape=(hidden_dim, static_embed_dim + dynamic_embed_dim + decoder_dim),
                dtype=jnp.float32,
            ),
        )
        self.vc = nnx.Param(
            param_init(
                key=rngs.params(),
                shape=(1, hidden_dim),
                dtype=jnp.float32,
            ),
        )
        self.Wc = nnx.Param(
            param_init(
                key=rngs.params(),
                shape=(hidden_dim, 2 * (static_embed_dim + dynamic_embed_dim)),
                dtype=jnp.float32,
            ),
        )

    def __call__(
        self,
        static_embed: jax.Array,
        dynamic_embed: jax.Array,
        memory: jax.Array,
    ):
        chex.assert_shape(  # (B, N, M_sta)
            static_embed, (None, None, self.static_embed_dim)
        )
        chex.assert_shape(  # (B, N, M_dyn)
            dynamic_embed, (None, None, self.dynamic_embed_dim)
        )
        chex.assert_rank(memory, 2)  # (B, M_mem)

        batchsize, seqsize, _ = static_embed.shape
        input_embed = jnp.concat(  # (B, N, M_sta + M_dyn)
            (static_embed, dynamic_embed), axis=-1
        )
        memory_broadcast = jnp.repeat(  # (B, N, M_mem)
            memory[:, None],  # (B, 1, M_mem)
            repeats=seqsize,
            axis=1,
        )

        hidden = jnp.concat(  # (B, N, M_sta + M_dyn + M_mem)
            (input_embed, memory_broadcast), axis=-1
        )
        hidden = hidden.transpose(0, 2, 1)  # (B, M_sta + M_dyn + M_mem, N)
        va = jnp.repeat(self.va[None], repeats=batchsize, axis=0)  # (B, 1, H)
        Wa = jnp.repeat(  # (B, H, M_sta + M_dyn + M_mem)
            self.Wa[None], repeats=batchsize, axis=0
        )
        alignment = nnx.softmax(  # (B, 1, N)
            va @ nnx.tanh(Wa @ hidden),
        )

        context = alignment @ input_embed  # (B, 1, M_sta + M_dyn)
        context = jnp.broadcast_to(context, input_embed.shape)  # (B, N, M_sta + M_dyn)

        input_embed_with_ctx = jnp.concat(  # (B, N, 2 * (M_sta + M_dyn))
            (input_embed, context), axis=2
        )
        input_embed_with_ctx = jnp.transpose(  # (B, 2 * (M_sta + M_dyn), N)
            input_embed_with_ctx, (0, 2, 1)
        )
        vc = jnp.repeat(self.vc[None], repeats=batchsize, axis=0)  # (B, 1, H)
        Wc = jnp.repeat(  # (B, H, 2 * (M_sta + M_dyn))
            self.Wc[None], repeats=batchsize, axis=0
        )
        logits = vc @ nnx.tanh(Wc @ input_embed_with_ctx)  # (B, 1, N)
        logits = logits.squeeze(1)  # (B, N)
        return logits


class PointerNetwork(nnx.Module):
    """A complete pointer network combining embeddings, an RNN decoder, and an attention mechanism for sequence-to-sequence tasks."""

    def __init__(
        self,
        static_input_dim: int = 2,
        dynamic_input_dim: int = 2,
        static_embed_dim: int = 128,
        dynamic_embed_dim: int = 128,
        decoder_dim: int = 128,
        attention_dim: int = 128,
        rnn_dropout: float = 0.1,
        *,
        param_init: nnx.Initializer = DEFAULT_INITIALIZER,
        rngs: nnx.Rngs,
    ) -> None:
        self.staticembedding = Embed(
            input_dim=static_input_dim,
            output_dim=static_embed_dim,
            param_init=param_init,
            rngs=rngs,
        )
        self.dynamicembedding = Embed(
            input_dim=dynamic_input_dim,
            output_dim=dynamic_embed_dim,
            param_init=param_init,
            rngs=rngs,
        )
        self.decoder = RNNDecoder(
            input_dim=static_embed_dim,
            hidden_dim=decoder_dim,
            dropout_rate=rnn_dropout,
            param_init=param_init,
            rngs=rngs,
        )
        self.attention = Attention(
            static_embed_dim=static_embed_dim,
            dynamic_embed_dim=dynamic_embed_dim,
            decoder_dim=decoder_dim,
            hidden_dim=attention_dim,
            param_init=param_init,
            rngs=rngs,
        )
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim

    def __call__(
        self,
        static: jax.Array,
        dynamic: jax.Array,
        x0: jax.Array,
        *,
        carry: tuple[jax.Array, jax.Array] | None = None,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
    ):
        chex.assert_shape(static, (None, self.static_input_dim, ...))
        chex.assert_shape(dynamic, (None, self.dynamic_input_dim, ...))
        chex.assert_shape(x0, (None, self.static_input_dim, 1))

        static = jnp.transpose(static, (0, 2, 1))
        dynamic = jnp.transpose(dynamic, (0, 2, 1))
        x0 = jnp.transpose(x0, (0, 2, 1))
        staticembed = self.staticembedding(static)
        dynamicembed = self.dynamicembedding(dynamic)
        x0embed = self.staticembedding(x0)
        _, carry = self.decoder(x0embed, carry=carry, rngs=rngs, training=training)
        h: jax.Array = self.decoder.h(carry)  # type: ignore - infallible: carry is not None
        logits = self.attention(staticembed, dynamicembed, h)
        return logits, carry

    def init(
        self,
        batch_dim: int,
        *,
        time_major: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        x0embed_shape = (batch_dim, self.staticembedding.output_dim, 1)
        return self.decoder.init(x0embed_shape, time_major=time_major, rngs=rngs)
