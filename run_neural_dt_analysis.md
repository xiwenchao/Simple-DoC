# Analysis of `python -m dichotomy_of_control.scripts.run_neural_dt --load_dir='./tests/testdata'`

## Scope

This note explains what happens when running:

```bash
python -m dichotomy_of_control.scripts.run_neural_dt --load_dir='./tests/testdata'
```

The analysis is based on the source code path rooted at:

- `dichotomy_of_control/scripts/run_neural_dt.py`
- `dichotomy_of_control/scripts/stochastic_decision_transformer_training.py`
- `dichotomy_of_control/scripts/stochastic_decision_transformer_evaluation.py`
- `dichotomy_of_control/models/stochastic_decision_transformer.py`
- `utils.py`
- `dice_rl/data/*`

## 1. High-level runtime flow

The entrypoint is `main()` in `dichotomy_of_control/scripts/run_neural_dt.py`.

With the provided command, the script builds this dataset directory name:

```text
./tests/testdata/FrozenLake-v1_tabularTrue_alpha-1.0_seed0_numtraj100_maxtraj100
```

This comes from the default flags:

- `env_name = FrozenLake-v1`
- `tabular_obs = True`
- `alpha = -1.0`
- `env_seed = 0`
- `num_trajectory = 100`
- `max_trajectory_length = 100`

The end-to-end flow is:

1. Load the saved dataset from disk with `Dataset.load(...)`.
2. Infer state/action dimensions from the dataset spec.
3. Create a `FrozenLakeWrapper` environment for evaluation.
4. Convert the saved replay data into NumPy trajectories.
5. Compute dataset-level state mean/std for normalization.
6. Build a stochastic decision transformer model.
7. Build the optimizer and sequence data loader.
8. Train for `max_iters`, each with `num_steps_per_iter` minibatches.
9. After each training iteration, evaluate the model in FrozenLake.

## 2. What exactly is loaded from disk

`Dataset.load(directory)` reconstructs a saved dataset object from:

- `dataset-ctr.pkl`: constructor metadata
- `dataset-ckpt-*`: TensorFlow checkpoint state

The object is a `TFOffpolicyDataset`, which stores experience as `EnvStep`s.

An `EnvStep` contains:

- `step_type`
- `step_num`
- `observation`
- `action`
- `reward`
- `discount`
- `policy_info`
- `env_info`
- `other_info`

So the persisted object is not yet a DT-specific trajectory dataset. It is a replay-style step dataset that later gets converted into trajectories.

### Important detail: trajectory membership is stored by the dataset, not by `EnvStep`

There is no explicit `trajectory_id` field inside each `EnvStep`.

Instead, `TFOffpolicyDataset` tracks episode boundaries in a separate
`episode_info_table`. For each episode, it stores:

- `episode_start_id`
- `episode_end_id`
- `episode_start_type`
- `episode_end_type`

So trajectory grouping is recovered by row ranges in the replay table, not by a
per-step trajectory label.

That is why the later DT conversion step does not infer trajectories by scanning
raw `EnvStep`s manually. It calls `dataset.get_all_episodes()`, which uses the
episode metadata to reconstruct contiguous episodes.

## 3. How steps and transitions are defined

The original per-step semantics come from `dice_rl/data/tf_agents_onpolicy_dataset.py`.

For a non-terminal step at time `t`, the stored step is:

- `observation = s_t`
- `action = a_t`
- `reward = r_t`

where `r_t` is the immediate reward returned after applying `a_t` to `s_t`.

Concretely:

1. The policy sees the current environment time step.
2. It chooses `a_t`.
3. The environment steps forward to the next observation.
4. The reward returned by that environment step is stored as the reward for the current `(s_t, a_t)` pair.

So the stored tuple is aligned in the standard RL way:

```text
(s_t, a_t, r_t, s_{t+1})
```

The next state is not stored as a separate field in `EnvStep`; it is implicit in the next `EnvStep`'s observation.

### Terminal-step handling

When the environment reaches a terminal observation, the code still emits a final `EnvStep` with `step_type = LAST`.

That terminal step carries:

- the terminal observation
- the previous action
- the terminal reward

This is convenient for replay storage, but it means the saved episode contains an extra final observation-like step.

## 4. How replay data becomes DT trajectories

The script calls:

```python
trajectories = utils.convert_to_np_dataset(
    dataset,
    tabular_obs=FLAGS.tabular_obs,
    tabular_act=True)
```

This converts the replay dataset into a list of dictionaries, one per episode:

```python
{
    'observations': ...,
    'actions': ...,
    'rewards': ...,
    'dones': ...
}
```

The trajectory structure comes from:

```python
episodes, valid_steps = dataset.get_all_episodes()
```

So the grouping into trajectories is driven by the `TFOffpolicyDataset`
episode metadata, not by a trajectory field inside `EnvStep`.

### Observations

For FrozenLake, observations are discrete state indices. Since `tabular_obs=True`, each state is converted to a one-hot vector.

If FrozenLake is the standard 4x4 grid, then:

- `state_dim = 16`
- each state is represented as a 16-dimensional one-hot vector

### Actions

Actions are also treated as discrete and one-hot encoded because `tabular_act=True`.

For FrozenLake:

- `act_dim = 4`
- each action becomes a 4-dimensional one-hot vector

### Rewards

The stored reward for each trajectory position is the immediate reward from the corresponding environment transition.

For FrozenLake this is typically:

- `0` for most steps
- `1` when the goal is reached

### Dones

The trajectory marks:

- `done = True` if the step is terminal
- `done = True` if the step slot is invalid padding beyond the episode length

### Important detail

`convert_to_np_dataset()` computes `episode_next_states`, but does not keep them in the final trajectory dictionary. The training code therefore works with `(state, action, reward, done)` sequences, not explicit next-state sequences.

## 5. FrozenLake-specific representation at inference time

For evaluation, `run_neural_dt.py` creates a `FrozenLakeWrapper`.

That wrapper does two things:

1. On `reset()`, it converts the discrete state index from Gym into a one-hot vector.
2. On `step(action)`, it interprets the model output by `argmax(action)` and passes the resulting discrete action into the real FrozenLake environment.

So although the model outputs a continuous 4D vector, the environment ultimately sees a discrete action:

```text
argmax(model_action_vector)
```

This is why the action head can be trained with MSE against one-hot targets and still work for a discrete environment.

## 6. How training batches are constructed

Training uses `StochasticSequenceDataLoader`.

Each minibatch is built as follows:

1. Sample trajectories with probability proportional to trajectory length.
2. For each sampled trajectory, pick a random starting index `si`.
3. Extract a past context window:

```text
[si, si + context_len)
```

4. Extract a future window:

```text
[si + context_len, si + context_len + future_len)
```

5. Compute discounted cumulative sums over rewards for those windows.
6. Left-pad shorter sequences to the fixed context/future lengths.
7. Normalize states with dataset mean/std.
8. Build attention masks to indicate which tokens are real and which are padding.

The data loader returns:

- past states
- past actions
- past rewards
- past dones
- past return-to-go (`rtg`)
- past timesteps
- past attention mask
- future states
- future actions
- future rewards
- future return-to-go
- future timesteps
- future attention mask

### Padding conventions

Padding uses:

- zeros for states
- `-10` for actions
- zeros for rewards
- `2` for dones
- zeros for timesteps and returns

Only positions with mask value `1` are used for loss computation.

## 7. Model architecture

The main model is `StochasticDecisionTransformer`.

It has three important sub-networks:

1. `self._transformer`
   - processes the past sequence
2. `self._future_seq_net`
   - encodes actual future information during training
3. `self._prior_seq_net`
   - predicts a latent distribution using only past information

### Tokenization scheme

Each time step is turned into three tokens:

- reward token
- state token
- action token

These are stacked in the order:

```text
(r_1, s_1, a_1, r_2, s_2, a_2, ...)
```

Each modality has its own embedding layer:

- reward embedding
- state embedding
- action embedding

Timestep embeddings are then added to all three.

The resulting sequence is fed into a causal transformer.

Although the model argument names say `returns_to_go`, the actual training call
passes immediate rewards into those slots. So the code path should be read as
"reward token" conditioning during training, not textbook DT return-to-go
conditioning.

### Latent variable

The stochastic part of the model is a latent variable `z`.

During training:

- the future encoder predicts a Gaussian posterior `q(z | future)`
- the prior encoder predicts a Gaussian prior `p(z | past)`

The model samples a latent from the future posterior and uses it together with the past state representation to predict:

- action
- value

This is the implementation of the "dichotomy of control" idea in the code: use future information during training to learn a latent summarizing uncontrollable or hidden future aspects, while also learning a prior that can later be sampled from past context alone.

## 8. What one training step actually does

Each training step:

1. Gets a batch from the data loader.
2. Runs the model with `training=True`.
3. Uses the past window and future window differently:
   - past window to build the main context representation
   - future window to build a latent posterior target
4. Produces:
   - action predictions
   - value predictions
   - future posterior parameters
   - past prior parameters
   - an energy matrix
5. Computes losses for the action head, value head, prior matching, and energy term.
6. Backpropagates through the total loss.

### 8.1 Inputs used by the model during training

The data loader returns both immediate rewards and reward-to-go arrays:

- `rewards`
- `rtg`
- `future_rewards`
- `future_rtg`

But the model call in `train_step()` is:

```python
self._model(
    states,
    actions,
    rewards,
    timesteps,
    attention_mask,
    future_states,
    future_actions,
    future_rewards,
    future_timesteps,
    future_attention_mask,
    training=True,
)
```

So the transformer itself is trained on:

- past reward sequence
- past state sequence
- past action sequence

plus a future reward/state/action window for latent inference.

The `rtg` tensor is not fed into the transformer during training. It is used
only as the value-head target later in the loss computation.

### 8.2 Latent training signal

The stochastic latent part is trained as follows:

1. The past context goes through the main transformer.
2. The future window goes through `future_seq_net` to produce a Gaussian
   posterior:

```text
q(z | future)
```

3. The past context goes through `prior_seq_net` to produce a Gaussian prior:

```text
p(z | past)
```

4. A latent sample from the future posterior is concatenated with the current
   state representation.
5. That combined representation is used by both the action head and value head.

This is the training-time mechanism that injects future information into the
representation while still teaching a past-only prior for inference.

### 8.3 Action-head training

The action head is:

- input: state-token representation from the past transformer concatenated with
  the sampled latent `z`
- target: one-hot action from the replay trajectory
- loss: mean squared error

So this is behavior cloning conditioned on:

- past reward/state/action context
- sampled latent inferred from the future during training

An auxiliary action accuracy is also logged by comparing the predicted action
argmax to the target action argmax.

### 8.4 Value-head training

The value head is trained separately from the action head, even though they
share the same latent-conditioned state representation.

Its setup is:

- input: the same state-token representation plus sampled latent `z`
- target: `value_target = rtg[:, :-1]`
- loss: mean squared error

So the value head is explicitly trained to predict return-to-go, even though the
main transformer tokens are immediate rewards rather than return-to-go tokens.

For FrozenLake, this is plausible because the episode return is bounded and the
value head output is passed through a final `tanh`.

### 8.5 Prior-matching loss

The prior loss is:

```text
KL(q(z | future) || p(z | past))
```

This teaches the past-only prior to approximate the future-informed posterior.

### 8.6 Energy loss

The energy loss uses:

- the current context embedding from the past transformer
- the sampled latent
- matched future embeddings
- shuffled future embeddings

to encourage compatibility with the correct future and incompatibility with
mismatched futures.

### 8.7 Total loss

The final loss is:

```text
loss =
    action_loss
  + value_loss
  + prior_weight * prior_loss
  + energy_weight * energy_loss
```

with:

- `action_loss`: MSE on one-hot actions
- `value_loss`: MSE on return-to-go targets
- `prior_loss`: KL between posterior and prior
- `energy_loss`: contrastive energy objective

## 9. How inference is implemented

Evaluation is done by `evaluate_stochastic_decision_transformer_episode(...)`.

For each episode:

1. Reset the environment and get the one-hot initial state.
2. Keep growing histories of:
   - states
   - actions
   - "target_return"
   - timesteps
3. At each step, append one padding action slot and one padding return slot.
4. Call `model.get_action(...)`.
5. Convert the returned action vector to a discrete action with `argmax`.
6. Step the environment.
7. Append the new observation and realized reward to history.
8. Stop when `done=True` or `max_ep_len` is reached.

### `get_action(...)`

`get_action()`:

1. Crops histories to the last `context_len` entries.
2. Left-pads them to fixed size.
3. Creates zero-filled "future" tensors.
4. Calls the model in inference mode.
5. Returns the last predicted action.

### Intended latent selection behavior

The intended design is:

1. Sample candidate latents from the past-only prior.
2. Score them with the value head.
3. Pick the latent with highest predicted value.
4. Decode the action using that latent.

That is the inference-side approximation of choosing among plausible futures using only information available from the past.

## 10. Important implementation caveats in the current code

These are not theoretical comments. They are direct consequences of the current implementation.

### Caveat 1: the transformer is trained on immediate rewards, not return-to-go

The data loader computes both:

- `rewards`
- `rtg`

But in `train_step()`, the model is called with `rewards` and `future_rewards` in the positions whose parameter names are `returns_to_go` and `future_returns_to_go`.

So:

- the transformer token stream is conditioned on immediate rewards
- the value head target still uses return-to-go

This means the model is not using the standard DT-style return-to-go conditioning during training.

### Caveat 2: the value head is trained on return-to-go, but the transformer context is still reward-conditioned

So the code mixes two signals:

- reward tokens as model input
- return-to-go as value supervision

This is a real part of the current implementation and should be described that
way, rather than as standard DT return-token training.

### Caveat 3: evaluation ignores the requested `target_return`

`evaluate_stochastic_decision_transformer_episode(...)` accepts a `target_return` argument, but then immediately overwrites it with an empty tensor:

```python
target_return = tf.zeros((0, 1), dtype=tf.float32)
```

So the requested target return passed from `run_neural_dt.py` is not actually used.

Instead, the code appends realized rewards as the episode progresses. So the model is effectively conditioned on reward history, not on the user-specified desired return.

### Caveat 4: prior-sample selection is not implemented cleanly

During inference, the code samples `prior_pred` from the prior distribution and appears to want to evaluate many candidate latents.

However, the value-scoring block uses repeated `f_pred` rather than the sampled `prior_pred` candidates. In practice, that means the "sample many prior latents and choose the best one by value" logic is not actually doing what the code structure suggests.

#### What latent is actually used for inference

There are two inference branches in the model:

1. If an external or cached `z` is provided, the model uses that `z` directly.
2. If no `z` is provided, the model samples from the prior and is supposed to
   choose the best prior sample using the value head.

The second branch is the one used by default in this script, because
`get_action()` passes `z=None` when `sample_per_step=True`.

The actual sequence inside `call(..., training=False)` is:

1. Build `f_pred` from the future encoder:

```text
f_pred ~ q(z | future_inputs)
```

2. Build `prior_pred` from the prior:

```text
prior_pred ~ p(z | past)
```

3. Repeat `prior_mean` and `prior_logvar` many times and sample many prior
   candidates.
4. Compute value scores, but mistakenly feed repeated `f_pred` into the value
   head instead of the sampled `prior_pred` candidates.
5. Take `best_idx = argmax(value_preds)`.
6. Use `best_idx` to select one sample out of `prior_pred`.
7. Feed that selected prior sample into the action head.

So the action head does ultimately use a prior sample, but the choice is not
actually based on comparing prior candidates by value.

Because the value head sees the same repeated `f_pred` for every candidate:

- all candidate scores are effectively identical
- `argmax` does not perform a meaningful comparison across prior samples
- the chosen prior sample is effectively arbitrary rather than "best by value"

In other words, the `z` used for the action head is:

- a cached latent `z`, if caching is enabled and the cached branch is used
- otherwise, one sampled latent from `p(z | past)`, but not a genuinely
  value-selected one

There is one more subtlety. During inference, `get_action()` fills the future
inputs with zeros. So the `f_pred` used in the mistaken value-scoring path is
not a true future-informed latent anyway; it is the future encoder output on
dummy zero future inputs.

#### What happens when `sample_per_step=False`

The model stores the chosen latent in `self._z`.

If `sample_per_step=False`, then:

1. The first action call samples and chooses a latent using the flawed logic
   above.
2. That latent is cached in `self._z`.
3. The next calls reuse the cached `z` directly.
4. After `future_len` steps, the cache is cleared and a new latent is sampled.

So in that mode, inference uses one reused prior latent for several steps, but
the first latent in the block still comes from the same imperfect selection
procedure.

### Caveat 5: next states are computed but not used in training

`convert_to_np_dataset()` derives next-state information from consecutive `EnvStep`s, but it discards `next_state` in the final trajectory representation. The neural DT training path therefore does not directly train on explicit `(s_t, a_t, r_t, s_{t+1})` tuples.

## 11. Practical summary

When you run the command on the test data:

- the script loads a saved FrozenLake replay dataset
- reconstructs trajectories from episode metadata stored by `TFOffpolicyDataset`
- converts discrete states and actions into one-hot vectors
- samples random trajectory windows plus future windows
- trains a stochastic transformer with:
  - past reward/state/action context modeling
  - future-informed latent inference
  - past-only latent prior matching
  - action-head training against one-hot actions
  - value-head training against return-to-go
  - energy regularization
- evaluates by rolling out in FrozenLake using `argmax` over the predicted action vector

The core idea is:

- during training, use true future information to learn a latent representation of future uncertainty
- also learn a prior over that latent from past context only
- at inference, act using only past context and sampled latent futures

But the current implementation deviates from the textbook decision-transformer formulation in a few places, especially:

- reward-token conditioning instead of RTG-token conditioning during training
- separate value-head supervision on RTG
- ignored target return during evaluation
- imperfect prior-sample selection logic at inference

## 12. Source files worth reading first

If you want to trace this yourself, the highest-value files are:

- `dichotomy_of_control/scripts/run_neural_dt.py`
- `utils.py`
- `dichotomy_of_control/scripts/stochastic_decision_transformer_training.py`
- `dichotomy_of_control/models/stochastic_decision_transformer.py`
- `dichotomy_of_control/scripts/stochastic_decision_transformer_evaluation.py`
- `dice_rl/data/tf_agents_onpolicy_dataset.py`
- `dice_rl/data/tf_offpolicy_dataset.py`
