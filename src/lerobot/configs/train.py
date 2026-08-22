# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, VQADatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.opd import OPDConfig
from lerobot.rl import FlowRLConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    vqa_dataset: VQADatasetConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)
    rl: FlowRLConfig = field(default_factory=FlowRLConfig)
    checkpoint_path: Path | None = field(init=False, default=None)
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            self.checkpoint_path = policy_dir.parent

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with `--policy.path`."
            )

        self.opd.validate()
        if hasattr(self.policy, "opd_enabled"):
            # The top-level OPD config is authoritative, including when an OPD
            # checkpoint is reused for ordinary supervised fine-tuning.
            self.policy.opd_enabled = False
        if self.opd.enable:
            if self.policy.type != "internvla_a1_5":
                raise ValueError("OPD with a Kairos teacher currently supports only internvla_a1_5")
            if not getattr(self.dataset, "include_opd_inputs", False):
                raise ValueError("OPD requires dataset.include_opd_inputs=true")
            if self.opd.require_on_policy and not getattr(
                self.dataset, "opd_rollout_dataset", False
            ):
                raise ValueError(
                    "OPD requires a student rollout dataset; set dataset.opd_rollout_dataset=true "
                    "only for data actually collected by the current student"
                )
            if self.vqa_dataset is not None and self.vqa_dataset.repo_id:
                raise ValueError("OPD does not support mixed VQA batches; disable vqa_dataset")
            if getattr(self.policy, "video_loss_only", False):
                raise ValueError("OPD requires policy.video_loss_only=false")
            if getattr(self.policy, "inference_backend", "standard") != "standard":
                raise ValueError("OPD training currently requires inference_backend='standard'")
            self.policy.opd_enabled = True
            self.policy.opd_loss_weight = self.opd.loss_weight
            self.policy.opd_sft_loss_weight = self.opd.sft_loss_weight
            self.policy.opd_student_std = self.opd.student_std
            self.policy.opd_max_kl_per_dim = self.opd.max_kl_per_dim

        self.rl.validate()
        if hasattr(self.policy, "rl_enabled"):
            self.policy.rl_enabled = False
        if self.rl.enable:
            if self.policy.type != "internvla_a1_5":
                raise ValueError("Flow RL currently supports only internvla_a1_5")
            if self.dataset is None or not getattr(self.dataset, "include_rl_signals", False):
                raise ValueError("RL requires dataset.include_rl_signals=true")
            if self.rl.require_on_policy and not getattr(
                self.dataset, "rl_rollout_dataset", False
            ):
                raise ValueError(
                    "RL requires data collected by the current student; set "
                    "dataset.rl_rollout_dataset=true only for genuine student rollouts"
                )
            if self.vqa_dataset is not None and self.vqa_dataset.repo_id:
                raise ValueError("RL does not support mixed VQA batches; disable vqa_dataset")
            if getattr(self.policy, "video_loss_only", False):
                raise ValueError("RL requires policy.video_loss_only=false")
            if getattr(self.policy, "inference_backend", "standard") != "standard":
                raise ValueError("RL training currently requires inference_backend='standard'")
            if self.rl.algorithm == "grpo":
                if not getattr(self.dataset, "rl_group_id_key", None):
                    raise ValueError(
                        "GRPO requires dataset.rl_group_id_key to identify rollouts "
                        "sampled for the same observation and task"
                    )
                if getattr(self.dataset, "rl_advantage_key", None) is not None:
                    raise ValueError(
                        "GRPO computes group-relative advantages from rewards; "
                        "do not set dataset.rl_advantage_key"
                    )
                if getattr(self.dataset, "streaming", False):
                    raise ValueError("GRPO currently requires a non-streaming grouped dataset")
                if self.batch_size % self.rl.group_size != 0:
                    raise ValueError(
                        "GRPO batch_size must be divisible by rl.group_size so groups are not split"
                    )
            self.policy.rl_enabled = True
            self.policy.rl_algorithm = self.rl.algorithm
            self.policy.rl_gamma = self.rl.gamma
            self.policy.rl_reward_horizon = self.rl.reward_horizon
            self.policy.rl_normalize_advantage = self.rl.normalize_advantage
            self.policy.rl_temperature = self.rl.temperature
            self.policy.rl_min_weight = self.rl.min_weight
            self.policy.rl_max_weight = self.rl.max_weight
            self.policy.rl_loss_weight = self.rl.loss_weight
            self.policy.rl_sft_loss_weight = self.rl.sft_loss_weight
            self.policy.rl_require_on_policy = self.rl.require_on_policy
            self.policy.rl_group_size = self.rl.group_size

        if not self.job_name:
            self.job_name = f"{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional
