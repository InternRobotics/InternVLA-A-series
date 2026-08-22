# Reinforcement fine-tuning InternVLA-A1.5

This repository supports two reward-based updates for the InternVLA-A1.5
continuous flow policy:

- **RWFM**: reward/advantage-weighted flow matching for ordinary student rollouts.
- **GRPO**: group-relative flow matching for multiple actions sampled from the
  same observation and task.

Both methods avoid claiming an exact action likelihood for the implicit flow
policy. The GRPO implementation follows the robotics group-relative variant in
[Reinforcement Learning for Flow-Matching Policies](https://arxiv.org/abs/2507.15073):
it standardizes rewards within a rollout group and applies non-negative
exponential advantage weights to the conditional flow-matching loss.

## Common rollout data

Collect a LeRobot dataset by running the current student checkpoint in the
target environment. Every sample must contain:

- the normal InternVLA observation, task, and action fields;
- `next.reward` for every environment step;
- episode boundaries so future reward padding can be masked correctly.

The loader computes `sum(gamma**t * reward[t])` over the configured reward
horizon. `--dataset.rl_rollout_dataset=true` is a provenance assertion; set it
only for rollouts produced by the current student. Repeat rollout collection
after each policy update round for strict on-policy training.

## RWFM

RWFM can consume independently sampled rollouts. It normalizes returns over the
local batch, or accepts a precomputed scalar advantage through
`--dataset.rl_advantage_key=<field>`.

```bash
BATCH_SIZE=8 \
RL_REWARD_HORIZON=100 \
bash launch/internvla_a15_rl_rwfm.sh owner/student-rollouts abs
```

## GRPO grouped data

For every observation/task condition, sample exactly `G` different student
action chunks and assign them one dataset-unique integer group identifier. The
default field is `grpo_group_id` and the default group size is four.

The sampler builds groups from those identifiers, shuffles complete groups
instead of individual records, and validates the identifiers again in the
loss. The current grouped loader requires:

- exactly one non-streaming rollout dataset;
- every group identifier appearing exactly `rl.group_size` times;
- per-device `batch_size` divisible by `rl.group_size`;
- exactly `rl.group_size` samples for every group identifier in a batch.

```bash
BATCH_SIZE=8 \
GRPO_GROUP_SIZE=4 \
GRPO_GROUP_ID_KEY=grpo_group_id \
RL_REWARD_HORIZON=100 \
bash launch/internvla_a15_rl_grpo.sh owner/grouped-student-rollouts abs
```

For group returns `R_i`, GRPO computes
`A_i = (R_i - mean_group(R)) / (std_group(R) + eps)` and weights each
conditional flow-matching loss by
`clip(exp(A_i / temperature), min_weight, max_weight)`.

## Joint Kairos OPD and RL

The Kairos launcher supports either RL algorithm.

RWFM plus OPD:

```bash
RL_ENABLE=true \
RL_ALGORITHM=rwfm \
KAIROS_ENDPOINT=http://127.0.0.1:8006 \
bash launch/internvla_a15_opd_kairos.sh \
  owner/student-rollouts \
  /path/to/kairos/robotwin_dataset_stats.json \
  abs
```

GRPO plus OPD:

```bash
RL_ENABLE=true \
RL_ALGORITHM=grpo \
RL_GROUP_SIZE=4 \
RL_GROUP_ID_KEY=grpo_group_id \
KAIROS_ENDPOINT=http://127.0.0.1:8006 \
bash launch/internvla_a15_opd_kairos.sh \
  owner/grouped-student-rollouts \
  /path/to/kairos/robotwin_dataset_stats.json \
  abs
```

The combined objective contains the selected RL loss, the continuous Kairos
reverse-KL loss, and a small supervised flow-matching prior.

## Important options

- `rl.algorithm`: `rwfm` or `grpo`.
- `rl.gamma`: reward discount.
- `rl.reward_horizon`: number of future rewards used for each rollout state.
- `rl.temperature`: smaller values emphasize larger advantages.
- `rl.min_weight/max_weight`: bound exponential advantage weights.
- `rl.group_size`: number of alternatives per condition for GRPO.
- `rl.loss_weight`: RL objective weight.
- `rl.sft_loss_weight`: behavior-cloning prior retained during RL.
