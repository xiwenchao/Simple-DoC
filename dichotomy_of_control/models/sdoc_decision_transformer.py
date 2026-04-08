# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stochastic Decision transformer architecture in Tensorflow."""
import tensorflow as tf
from dichotomy_of_control import utils


class TransformerConfig:
  """Minimal transformer config for the local TensorFlow implementation."""

  def __init__(self,
               n_embd,
               n_layer=12,
               n_head=12,
               n_inner=None,
               activation_function='gelu',
               resid_pdrop=0.1,
               attn_pdrop=0.1,
               embd_pdrop=0.1,
               layer_norm_epsilon=1e-5,
               **kwargs):
    if n_embd % n_head != 0:
      raise ValueError('n_embd must be divisible by n_head.')
    self.n_embd = n_embd
    self.n_layer = n_layer
    self.n_head = n_head
    self.n_inner = n_inner
    self.activation_function = activation_function
    self.resid_pdrop = resid_pdrop
    self.attn_pdrop = attn_pdrop
    self.embd_pdrop = embd_pdrop
    self.layer_norm_epsilon = layer_norm_epsilon
    for key, value in kwargs.items():
      setattr(self, key, value)


def _get_activation(activation_name):
  """Returns a Keras-compatible activation used by the feed-forward block."""
  if activation_name == 'gelu_new':
    return lambda x: tf.keras.activations.gelu(x, approximate=True)
  if activation_name == 'gelu':
    return tf.keras.activations.gelu
  return tf.keras.activations.get(activation_name)


class PositionlessTransformerBlock(tf.keras.layers.Layer):
  """Causal self-attention block used by the decision transformer."""

  def __init__(self, config, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
    self._ln_1 = tf.keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon, name='ln_1')
    self._attn = tf.keras.layers.MultiHeadAttention(
        num_heads=config.n_head,
        key_dim=config.n_embd // config.n_head,
        dropout=config.attn_pdrop,
        name='attn')
    self._attn_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
    self._ln_2 = tf.keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon, name='ln_2')
    self._fc = tf.keras.layers.Dense(
        inner_dim, activation=_get_activation(config.activation_function),
        name='fc')
    self._proj = tf.keras.layers.Dense(config.n_embd, name='proj')
    self._proj_dropout = tf.keras.layers.Dropout(config.resid_pdrop)

  def call(self, hidden_states, attention_mask, training=False):
    sequence_length = tf.shape(hidden_states)[1]
    batch_size = tf.shape(hidden_states)[0]
    causal_mask = tf.linalg.band_part(
        tf.ones((sequence_length, sequence_length), dtype=tf.bool), -1, 0)

    if attention_mask is not None:
      padding_mask = tf.cast(attention_mask[:, tf.newaxis, :], tf.bool)
      padding_mask = tf.broadcast_to(
          padding_mask, (batch_size, sequence_length, sequence_length))
      attn_mask = tf.logical_and(padding_mask, causal_mask[tf.newaxis, :, :])
      use_causal_mask = False
    else:
      attn_mask = causal_mask
      use_causal_mask = False

    attn_input = self._ln_1(hidden_states)
    attn_output = self._attn(
        query=attn_input,
        value=attn_input,
        key=attn_input,
        attention_mask=attn_mask,
        use_causal_mask=use_causal_mask,
        training=training)
    hidden_states = hidden_states + self._attn_dropout(
        attn_output, training=training)

    mlp_input = self._ln_2(hidden_states)
    mlp_output = self._fc(mlp_input)
    mlp_output = self._proj(mlp_output)
    mlp_output = self._proj_dropout(mlp_output, training=training)
    return hidden_states + mlp_output


class PositionlessTFGPT2MainLayer(tf.keras.layers.Layer):
  """GPT2 transformer layer without position embeddings.

  Takes input embeddings and attention mask directly, as opposed to usual
  transformer layer, which takes in inputs and maps to embeddings internally.
  """

  def __init__(self, config, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

    self._config = config
    self._drop = tf.keras.layers.Dropout(config.embd_pdrop)
    self._h = [
        PositionlessTransformerBlock(config, name='h_._{}'.format(i))
        for i in range(config.n_layer)
    ]
    self._ln_f = tf.keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon, name="ln_f")

  def call(self, inputs_embeds, attention_mask, training=False):
    """Forward pass for the PositionlessTFGPT2MainLayer.

    Args:
      inputs_embeds: A tf.Tensor of input embeddings.
      attention_mask: A tf.Tensor of booleans indicating which inputs can be
        attended to by the layer.
      training: A bool for whether training or evaluating.

    Returns:
      hidden_states: A tf.Tensor representing the final hidden states.
    """
    hidden_states = inputs_embeds
    hidden_states = self._drop(hidden_states, training=training)

    for block in self._h:
      hidden_states = block(
          hidden_states,
          attention_mask=attention_mask,
          training=training)

    hidden_states = self._ln_f(hidden_states)
    return hidden_states


class StochasticDecisionTransformer(tf.keras.Model):
  """Uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)."""

  def __init__(self,
               state_dim,
               act_dim,
               hidden_size,
               context_len,
               max_ep_len,
               model_type='transformer',
               future_len=None,
               sample_per_step=True,
               normalize_action=True,
               normalize_return=True,
               latent_size=32,
               **kwargs):
    """Initializes a DecisionTransformer.

    Args:
      state_dim: An int dimension of observations.
      act_dim: An int dimension of actions.
      hidden_size: An int size of hidden layers in transformer.
      context_len: An int maximum context length.
      max_ep_len: An int maximum episode length.
      **kwargs: Additional kwargs passed onto GPT2Config.
    """
    super().__init__()
    self._state_dim = state_dim
    self._act_dim = act_dim
    self._hidden_size = hidden_size
    self._latent_size = latent_size or hidden_size
    self._context_len = context_len
    self._future_len = future_len or context_len
    self._model_type = model_type
    self._sample_per_step = sample_per_step
    config = TransformerConfig(
        vocab_size=1,  # doesn't matter -- we don't use the vocab
        n_embd=hidden_size,
        **kwargs)
    # note: the only difference between this GPT2Model and the default
    # Huggingface version is that the positional embeddings are removed
    # (since we'll add those ourselves)
    self._transformer = PositionlessTFGPT2MainLayer(config)

    self._embed_timestep = tf.keras.layers.Embedding(max_ep_len, hidden_size)
    self._embed_return = tf.keras.layers.Dense(hidden_size)
    self._embed_state = tf.keras.layers.Dense(hidden_size)
    self._embed_action = tf.keras.layers.Dense(hidden_size)

    self._embed_ln = tf.keras.layers.LayerNormalization()

    self._predict_action = utils.create_mlp(
        [self._context_len, self._hidden_size + self._latent_size],
        self._act_dim,
        last_layer_activation='tanh' if normalize_action else None)
    # The latent Gaussian is conditioned on the current state-token
    # representation after the main transformer. This hidden state already
    # summarizes the available past context through causal self-attention.
    self._predict_latent = utils.create_mlp(
        self._hidden_size, self._latent_size * 2)
    self._predict_value = utils.create_mlp(
        [self._context_len, self._hidden_size + self._latent_size],
        1,
        last_layer_activation='tanh' if normalize_return else None)

    # Keep track of sampling frequency of z during inference.
    self._z = None
    self._z_counter = 0

  @tf.function
  def call(self,
           states,
           actions,
           returns_to_go,
           timesteps,
           attention_mask,
           future_states,
           future_actions,
           future_returns_to_go,
           future_timesteps,
           future_attention_mask,
           future_samples=100,
           training=False,
           z=None):
    """Forward pass for DecisionTransformer.

    Args:
      states: A tf.Tensor representing a batch of state trajectories of shape
        `[B, T, ...]`.
      actions: A tf.Tensor representing a batch of action trajectories of shape
        `[B, T, ...]`.
      returns_to_go: A tf.Tensor representing a batch of returns-to-go of shape
        `[B, T]`.
      timesteps: A tf.Tensor representing a batch of timesteps of shape `[B,
        T]`.
      attention_mask: A tf.Tensor representing which parts of the trajectories
        can be attended to by the model. For example, it cannot use the future
        to predict a next action. Shape `[B, T]`.
      training: A bool representing whether we are training or evaluating.

    Returns:
      action_preds: A tf.Tensor representing predicted actions
        of shape `[B, T]`.
    """

    batch_size, seq_length = states.shape[0], states.shape[1]
    # embed each modality with a different head
    state_embeddings = tf.reshape(
        self._embed_state(
            tf.reshape(states,
                       [batch_size * seq_length] + states.shape[2:].as_list())),
        [batch_size, seq_length, -1])
    action_embeddings = self._embed_action(actions)
    returns_embeddings = self._embed_return(returns_to_go)
    time_embeddings = self._embed_timestep(timesteps)

    # time embeddings are treated similar to positional embeddings
    state_embeddings = state_embeddings + time_embeddings
    action_embeddings = action_embeddings + time_embeddings
    returns_embeddings = returns_embeddings + time_embeddings

    # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
    # which works nice in an autoregressive sense since states predict actions
    stacked_inputs = tf.stack(
        (returns_embeddings, state_embeddings, action_embeddings), axis=1)
    stacked_inputs = tf.transpose(stacked_inputs, perm=[0, 2, 1, 3])
    stacked_inputs = tf.reshape(stacked_inputs,
                                (batch_size, 3 * seq_length, self._hidden_size))
    stacked_inputs = self._embed_ln(stacked_inputs)

    # to make the attention mask fit the stacked inputs, have to stack it too
    stacked_attention_mask = tf.stack(
        (attention_mask, attention_mask, attention_mask), axis=1)
    stacked_attention_mask = tf.transpose(stacked_attention_mask, [0, 2, 1])
    stacked_attention_mask = tf.reshape(stacked_attention_mask,
                                        (batch_size, 3 * seq_length))

    # we feed in the input embeddings (not word indices as in NLP) to the model
    x = self._transformer(
        inputs_embeds=stacked_inputs,
        attention_mask=stacked_attention_mask,
        training=training,
    )

    # reshape x so that the second dimension corresponds to the original
    # returns (0), states (1), or actions (2); i.e. x[:,1,t] is for s_t
    x = tf.reshape(x, (batch_size, seq_length, 3, self._hidden_size))
    x = tf.transpose(x, (0, 2, 1, 3))
    # Use the final state-token hidden state from the past transformer, not the
    # raw observation embedding. This is a past-contextualized representation of
    # the current state.
    current_past_representation = x[:, 1, -1, :]

    latent = self._predict_latent(current_past_representation)
    latent_mean, latent_logvar = tf.split(latent, 2, axis=-1)
    latent_std = tf.exp(0.5 * latent_logvar)

    if training:
      f_pred = latent_mean + tf.random.normal(tf.shape(latent_mean)) * latent_std
      value_preds = self._predict_value(
          tf.concat(
              [x[:, 1],
               tf.repeat((f_pred)[:, None, :], seq_length, axis=1)],
              axis=-1))
    else:
      if z is not None:
        f_pred = z
        value_preds = self._predict_value(
            tf.concat(
                [x[:, 1],
                 tf.repeat((f_pred)[:, None, :], seq_length, axis=1)],
                axis=-1))[Ellipsis, -1]
      else:
        repeated_mean = tf.repeat(latent_mean, future_samples, axis=0)
        repeated_std = tf.repeat(latent_std, future_samples, axis=0)
        candidate_z = repeated_mean + tf.random.normal(
            tf.shape(repeated_mean)) * repeated_std
        repeated_state_tokens = tf.repeat(x[:, 1], future_samples, axis=0)
        value_preds = self._predict_value(
            tf.concat([
                repeated_state_tokens,
                tf.repeat(candidate_z[:, None, :], seq_length, axis=1)
            ],
                      axis=-1))
        value_preds = tf.reshape(value_preds,
                                 [batch_size, future_samples, seq_length])[Ellipsis,
                                                                           -1]
        best_idx = tf.argmax(value_preds, axis=1)
        value_preds = tf.squeeze(
            tf.gather(value_preds, best_idx, axis=1), axis=1)
        f_pred = tf.squeeze(
            tf.gather(
                tf.reshape(candidate_z, [batch_size, future_samples, -1]),
                best_idx,
                axis=1),
            axis=1)

    # get predictions
    action_preds = self._predict_action(
        tf.concat(
            [x[:, 1],
             tf.repeat((f_pred)[:, None, :], seq_length, axis=1)],
            axis=-1))

    # Keep the old return signature for compatibility with the existing
    # trainer/evaluation code. Prior-related outputs are tied to the same latent
    # statistics, and energies are set so the old energy loss is neutral.
    energies = tf.eye(batch_size, dtype=action_preds.dtype) * 1000.0
    return (action_preds, value_preds, f_pred, latent_mean, latent_logvar,
            latent_mean, latent_logvar, energies)

  def get_action(self, states, actions, returns_to_go, timesteps):
    """Predict a next action given a trajectory.

    Args:
      states: A tf.Tensor representing a sequence of states of shape `[T, ...]`.
      actions: A tf.Tensor representing a sequence of actions of shape `[T,
        ...]`.
      returns_to_go: A tf.Tensor representing a sequence of returns-to-go of
        shape `[T]`.
      timesteps: A tf.Tensor representing a sequence of timesteps of shape
        `[T]`.

    Returns:
      action: A tf.Tensor representing a single next action.
    """
    states = tf.reshape(states, [1, -1, self._state_dim])
    actions = tf.reshape(actions, (1, -1, self._act_dim))
    returns_to_go = tf.reshape(returns_to_go, (1, -1, 1))
    timesteps = tf.reshape(timesteps, (1, -1))

    states = states[:, -self._context_len:]
    actions = actions[:, -self._context_len:]
    returns_to_go = returns_to_go[:, -self._context_len:]
    timesteps = timesteps[:, -self._context_len:]

    # pad all tokens to sequence length
    attention_mask = tf.concat([
        tf.zeros(self._context_len - states.shape[1]),
        tf.ones(states.shape[1])
    ],
                               axis=0)
    attention_mask = tf.reshape(attention_mask, (1, -1))
    states = tf.concat([
        tf.zeros([
            states.shape[0], self._context_len - states.shape[1],
            self._state_dim
        ]), states
    ],
                       axis=1)
    actions = tf.concat([
        tf.zeros((actions.shape[0], self._context_len - actions.shape[1],
                  self._act_dim)), actions
    ],
                        axis=1)
    returns_to_go = tf.concat([
        tf.zeros((returns_to_go.shape[0],
                  self._context_len - returns_to_go.shape[1], 1)), returns_to_go
    ],
                              axis=1)
    timesteps = tf.concat([
        tf.zeros((timesteps.shape[0], self._context_len - timesteps.shape[1]),
                 dtype=tf.int64), timesteps
    ],
                          axis=1)
    future_states = tf.zeros(
        [states.shape[0], self._future_len] + list(states.shape[2:]),
        dtype=states.dtype)
    future_actions = tf.zeros(
        [actions.shape[0], self._future_len, actions.shape[-1]],
        dtype=actions.dtype)
    future_returns_to_go = tf.zeros(
        [returns_to_go.shape[0], self._future_len, returns_to_go.shape[-1]],
        dtype=returns_to_go.dtype)
    future_timesteps = tf.zeros([timesteps.shape[0], self._future_len],
                                dtype=timesteps.dtype)
    future_attention_mask = tf.zeros(
        [attention_mask.shape[0], self._future_len], dtype=attention_mask.dtype)

    action_preds, value_preds, self._z, _, _, _, _, _ = self(
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask,
        future_states,
        future_actions,
        future_returns_to_go,
        future_timesteps,
        future_attention_mask,
        training=False,
        z=self._z if not self._sample_per_step else None,
    )

    self._z_counter += 1
    if self._z_counter == self._future_len:
      self._z = None
      self._z_counter = 0

    return action_preds[0, -1]
