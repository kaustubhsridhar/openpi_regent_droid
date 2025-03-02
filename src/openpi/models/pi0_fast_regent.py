import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTRegentConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250
    num_retrieved_observations: int = 5
    use_action_interpolation: bool = False
    lamda: float = 10.0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTRegent":
        return Pi0FASTRegent(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing
    
    def get_freeze_filter_with_frozen_img_encoder(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            # Freeze both llm (except lora parts) and img components
            return nnx.Any(
                nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
                nnx_utils.PathRegex(".*img.*")
            )
        return nnx.Nothing


class Pi0FASTRegent(_model.BaseModel):
    def __init__(self, config: Pi0FASTRegentConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.num_retrieved_observations = config.num_retrieved_observations
        self.use_action_interpolation = config.use_action_interpolation
        self.max_token_len = config.max_token_len # max token len for the "prompt, state, action" prompt
    
    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            # image_token_embeddings = obs.images[name] # Alt: no need to embed as we are feeding the embeddings in directly but this embedding is averaged over patches!

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add tokenized inputs
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # return embeddings, input mask, and ar mask
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )
    
    def combine_attn_masks(self, list_of_attn_masks, batch_size, seq_len, num_observations):
        # for the attn masks, you need to combine them as block diagonal, 
        # with the remaining lower triangular elements being true, and the remaining upper triangular elements being false
        assert seq_len % num_observations == 0
        block_len = seq_len // num_observations
        attn_mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=bool)
        for i in range(num_observations):
            attn_mask = attn_mask.at[:, i*block_len:(i+1)*block_len, i*block_len:(i+1)*block_len].set(list_of_attn_masks[i])
        # finally, do a logical or with a lower triangular mask
        attn_mask = jnp.logical_or(attn_mask, jnp.tril(jnp.ones((batch_size, seq_len, seq_len), dtype=bool)))
        return attn_mask
    
    def interpolate_actions(self, logits, first_targets, exp_lamda_distances):
        # assert the input shapes
        batch_size, decode_len, vocab_size = logits.shape
        if decode_len == 1:
            # call from sample_actions
            pass
        else:
            # call from compute_loss
            num_observations = self.num_retrieved_observations + 1
            # assert decode_len == num_observations * (self.max_token_len - 1)
            # assert first_targets.shape == (batch_size, self.max_token_len - 1, vocab_size)
            # assert exp_lamda_distances.shape == (batch_size, num_observations, 1)

            # repeat the exp_lamda_distances for each token in the decode_len
            exp_lamda_distances_repeated = einops.repeat(exp_lamda_distances, "b o 1 -> b (o t) 1", t=self.max_token_len - 1)
            # assert exp_lamda_distances_repeated.shape == (batch_size, decode_len, 1)
            
            # acquire and repeat the Retrieve-and-Play (first) targets
            first_targets_repeated = einops.repeat(first_targets, "b t v -> b (o t) v", o=num_observations)
            # assert first_targets_repeated.shape == (batch_size, decode_len, vocab_size)

        # discrete interpolation
        new_logits = exp_lamda_distances_repeated * first_targets_repeated + (1 - exp_lamda_distances_repeated) * jax.nn.softmax(logits, axis=-1)

        return new_logits

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, regent_observation: _model.RegentObservation, actions: _model.Actions, *, train: bool = False, decode_indices: at.Int[at.Array, ""] = None
    ) -> at.Float[at.Array, "*b ah"]:
        num_observations = self.num_retrieved_observations + 1
        assert num_observations == self.num_retrieved_observations + 1
        list_of_input_token_embeddings = []
        list_of_attn_masks = []
        list_of_loss_masks = []
        list_of_targets = []
        for i in range(num_observations):
            prefix = f"retrieved_{i}_" if i < self.num_retrieved_observations else "query_"
            this_observation = _model.extract_observation_from_regent_observation(regent_observation, prefix)
            this_observation = _model.preprocess_observation(
                rng, this_observation, train=train, image_keys=list(this_observation.images.keys())
            )

            # Compute inputs: one big forward pass of prefix + suffix at once
            this_input_token_embeddings, this_input_mask, this_ar_mask = self.embed_inputs(this_observation)
            this_attn_mask = make_attn_mask(this_input_mask, this_ar_mask)

            list_of_input_token_embeddings.append(this_input_token_embeddings)
            list_of_attn_masks.append(this_attn_mask)
            list_of_loss_masks.append(this_observation.token_loss_mask[:, 1:])

            # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
            # FYI, query_observation is the last observation in the batch
            this_targets = jax.nn.one_hot(
                this_observation.tokenized_prompt[:, 1:],
                self.PaliGemma.llm.module.vocab_size,
            )
            list_of_targets.append(this_targets)

        # combine most lists along the num tokens axis
        input_token_embeddings = jnp.concatenate(list_of_input_token_embeddings, axis=1)
        batch_size, seq_len = input_token_embeddings.shape[0:2]
        attn_mask = self.combine_attn_masks(list_of_attn_masks, batch_size, seq_len, num_observations)
        loss_mask = jnp.concatenate(list_of_loss_masks, axis=1)
        targets = jnp.concatenate(list_of_targets, axis=1)

        print(f'input_token_embeddings shape: {input_token_embeddings.shape}')
        print(f'attn_mask shape: {attn_mask.shape}')
        print(f'loss_mask shape: {loss_mask.shape}')
        print(f'targets shape: {targets.shape}')

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        print(f'pre_logits shape: {pre_logits.shape}')

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, decode_indices],
        )
        print(f'logits shape: {logits.shape}')

        if self.use_action_interpolation:
            new_logits = self.interpolate_actions(logits=logits, first_targets=targets[:, :self.max_token_len - 1, :], 
                                                  exp_lamda_distances=regent_observation.exp_lamda_distances)
            print(f'new_logits shape: {new_logits.shape}')
            # clamp the logits to avoid nan from log(0) and instabilities from log(1)
            epsilon = 1e-9
            new_logits = jnp.clip(new_logits, epsilon, 1 - epsilon)
            # take log
            logp = jnp.log(new_logits)
        else:
            logp = jax.nn.log_softmax(logits, axis=-1)
        print(f'logp shape: {logp.shape}')

        # Compute CE loss on token targets
        token_pplx = jnp.sum(targets * logp, axis=-1)
        print(f'token_pplx shape: {token_pplx.shape}')
        loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
        print(f'loss shape: {loss.shape}')
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        regent_observation: _model.RegentObservation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        # TODO: this is a hack to get the image keys.
        num_observations = self.num_retrieved_observations + 1
        assert num_observations == self.num_retrieved_observations + 1
        list_of_prefix_token_embeddings = []
        list_of_prefix_masks = []
        list_of_prefix_attn_masks = []
        for i in range(num_observations):
            prefix = f"retrieved_{i}_" if i < self.num_retrieved_observations else "query_"
            this_observation = _model.extract_observation_from_regent_observation(regent_observation, prefix)
            this_observation = _model.preprocess_observation(
                None, this_observation, train=False, image_keys=list(this_observation.images.keys())
            )

            # embed inputs
            this_prefix_token_embeddings, this_prefix_mask, this_prefix_ar_mask = self.embed_inputs(this_observation)
            this_prefix_attn_mask = make_attn_mask(this_prefix_mask, this_prefix_ar_mask)

            list_of_prefix_token_embeddings.append(this_prefix_token_embeddings)
            list_of_prefix_masks.append(this_prefix_mask)
            list_of_prefix_attn_masks.append(this_prefix_attn_mask)

            # get the first targets (of the first retrieved observation)
            if i == 0:
                first_targets = jax.nn.one_hot(
                    this_observation.tokenized_prompt[:, 1:],
                    self.PaliGemma.llm.module.vocab_size,
                )

        # combine all input token embeddings and attn masks along the num tokens axis
        prefix_token_embeddings = jnp.concatenate(list_of_prefix_token_embeddings, axis=1)
        prefix_mask = jnp.concatenate(list_of_prefix_masks, axis=1)
        batch_size, seq_len = prefix_token_embeddings.shape[0:2]
        prefix_attn_mask = self.combine_attn_masks(list_of_prefix_attn_masks, batch_size, seq_len, num_observations)
        print(f'prefix_token_embeddings shape: {prefix_token_embeddings.shape}')
        print(f'prefix_mask shape: {prefix_mask.shape}')
        print(f'prefix_attn_mask shape: {prefix_attn_mask.shape}')

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # prepare decoding -- final logit decodes the first token
        last_logit_old = prefix_logits[:, -1:]

        # possible action interpolation
        if self.use_action_interpolation:
            print(f'step: 0 / {max_decoding_steps}')
            print(f'full first_targets shape: {first_targets.shape}')
            print(f'exp_lamda_distances for only last query observation shape: {regent_observation.exp_lamda_distances[:, -1:, :].shape}')
            print(f'last_logit_old shape: {last_logit_old.shape}')
            last_logit = self.interpolate_actions(logits=last_logit_old, first_targets=first_targets[:, 0:1, :], 
                                                  exp_lamda_distances=regent_observation.exp_lamda_distances[:, -1:, :])
            print(f'last_logit shape: {last_logit.shape}')

        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            last_logit_old, output_tokens, cache, _, step = carry

            # possible action interpolation
            if self.use_action_interpolation:
                print(f'step: {step} / {max_decoding_steps}')
                print(f'last_logit_old shape: {last_logit_old.shape}')
                last_logit = self.interpolate_actions(logits=last_logit_old, first_targets=first_targets[:, step+1:step+2, :], 
                                                      exp_lamda_distances=regent_observation.exp_lamda_distances[:, -1:, :])
                print(f'last_logit shape: {last_logit.shape}')

            # Sample token from last logit
            if temperature > 0.0:
                last_logit = last_logit / temperature
                token = jax.random.categorical(rng, last_logit, axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, output_tokens, _, _, _ = jax.lax.while_loop(cond, step, (last_logit, output_tokens, kv_cache, False, 0))
        return output_tokens
