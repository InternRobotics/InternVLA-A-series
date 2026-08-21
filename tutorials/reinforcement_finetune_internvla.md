# Reinforcement fine-tuning InternVLA-A1.5

This repository supports Reward-Weighted Flow Matching (RWFM) for
InternVLA-A1.5. RWFM is suitable for an implicit flow policy because it does
not require exact action log probabilities. It exponentially reweights each
sample's flow-matching loss using the relative discounted return of a student
rollout batch.

References: [Reinforcement Learning for Flow-Matching Policies](https://arxiv.org/abs/2507.15073)
and [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177).

## Rollout data

Collect a LeRobot dataset by running the current student checkpoint in the
target environment. It must contain:

- the normal InternVLA observation, task, and action fields;
- `next.reward` for every environment step;
- episode boundaries so future reward padding can be masked correctly.

The training loader requests a future reward window and computes
`sum(gamma**t * reward[t])`. Set `rl.reward_horizon` larger than the action
chunk if sparse rewards occur late. Alternatively, store a precomputed scalar
advantage in the dataset and pass its field name with
`--dataset.rl_advantage_key=<field>`.

`--dataset.rl_rollout_dataset=true` is a provenance assertion. Set it only for
data actually produced by the current student. For strict on-policy training,
repeat rollout collection after every training round.

## RL-only training

```bash
BATCH_SIZE=8 \
RL_REWARD_HORIZON=100 \
bash launch/internvla_a15_rl_rwfm.sh owner/student-rollouts abs
```

The batch size should be greater than one when batch-normalized advantages are
used. Important options are:

- `rl.gamma`: reward discount.
- `rl.temperature`: smaller values concentrate updates on higher-return actions.
- `rl.min_weight/max_weight`: bound the exponential advantage weights.
- `rl.loss_weight`: RWFM objective weight.
- `rl.sft_loss_weight`: behavior-cloning prior retained during RL.
- `rl.normalize_advantage`: normalize returns within each distributed-local batch.

## Joint Kairos OPD + environment reward training

The same student rollout can carry both environment rewards and Kairos teacher
labels. Start the Kairos service, then enable RL in the OPD launcher:

```bash
RL_ENABLE=true \
KAIROS_ENDPOINT=http://127.0.0.1:8006 \
bash launch/internvla_a15_opd_kairos.sh \
  owner/student-rollouts \
  /path/to/kairos/robotwin_dataset_stats.json \
  abs
```

The combined objective contains the RWFM loss, the continuous Kairos reverse-KL
loss, and a small supervised flow-matching prior. This implementation does not
claim PPO compatibility: exact likelihood ratios are unavailable from the
current InternVLA-A1.5 flow policy and Kairos action-only API.
