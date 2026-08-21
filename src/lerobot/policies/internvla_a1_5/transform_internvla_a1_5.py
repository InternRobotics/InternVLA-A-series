from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5Tokenizer
from transformers.models.qwen3_vl import Qwen3VLProcessor
from transformers.utils import cached_file

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.opd import (
    OPD_IMAGE_MASK,
    OPD_IMAGE_PREFIX,
    OPD_IS_ON_POLICY,
    OPD_RAW_STATE,
    OPD_ROBOT_TYPE,
    OPD_TASK,
)
from lerobot.policies.internvla_a1_5.action_tokens import ensure_qwen35_action_tokens
from lerobot.transforms.constants import DEFAULT_IMAGE_TOKEN, SYSTEM_MESSAGE
from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR

LABEL_MODE_NONE = 0
LABEL_MODE_TEXT = 1
LABEL_MODE_FAST = 2
LABEL_MODE_BOTH = 3


@DataTransformFn.register_subclass("capture_opd_inputs")
@dataclass
class CaptureOPDInputsTransformFn(DataTransformFn):
    """Keep raw state, camera views, and task text for a remote OPD teacher.

    This transform must run before delta conversion, normalization, image
    remapping, or the final unifying transform.
    """

    is_on_policy: bool = False
    max_views: int = 3
    state_keys: list[str] | None = None
    image_keys: list[str] | None = None
    robot_type: str = ""
    task_key: str = "task"

    def hydrate(
        self, dataset: LeRobotDataset | StreamingLeRobotDataset
    ) -> CaptureOPDInputsTransformFn:
        schema = get_schema(dataset.meta.robot_type).resolve()
        return replace(
            self,
            state_keys=schema.get_state_keys(),
            image_keys=list(schema.image_mapping),
            robot_type=dataset.meta.robot_type,
        )

    @staticmethod
    def _align_for_cat(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        max_ndim = max(tensor.ndim for tensor in tensors)
        return [tensor if tensor.ndim == max_ndim else tensor.unsqueeze(-1) for tensor in tensors]

    def __call__(self, data: DataDict) -> DataDict:
        if not self.state_keys or not self.image_keys or not self.robot_type:
            raise RuntimeError("CaptureOPDInputsTransformFn must be hydrated with a dataset")
        missing_state = [key for key in self.state_keys if key not in data]
        missing_images = [key for key in self.image_keys if key not in data]
        if missing_state or missing_images:
            raise KeyError(
                f"Cannot capture OPD inputs; missing state={missing_state}, images={missing_images}"
            )

        state_parts = self._align_for_cat([data[key] for key in self.state_keys])
        data[OPD_RAW_STATE] = torch.cat(state_parts, dim=-1).detach().clone().float()

        valid_views = min(len(self.image_keys), self.max_views)
        first_image = data[self.image_keys[0]]
        for view_index in range(self.max_views):
            image = data[self.image_keys[view_index]] if view_index < valid_views else first_image
            if image.ndim == 4:  # image_delta_indices yields [T,C,H,W]; teacher uses current frame
                image = image[0]
            data[f"{OPD_IMAGE_PREFIX}{view_index}"] = image.detach().clone()
        data[OPD_IMAGE_MASK] = torch.tensor(
            [view_index < valid_views for view_index in range(self.max_views)], dtype=torch.bool
        )
        data[OPD_TASK] = str(data.get(self.task_key, ""))
        data[OPD_ROBOT_TYPE] = self.robot_type
        data[OPD_IS_ON_POLICY] = torch.tensor(self.is_on_policy, dtype=torch.bool)
        return data


def _fast_processor_kwargs(model_name_or_path: str) -> dict[str, str]:
    tokenizer_file = Path(model_name_or_path) / "tokenizer.json"
    if tokenizer_file.is_file():
        return {"tokenizer_file": str(tokenizer_file)}
    resolved = cached_file(model_name_or_path, "tokenizer.json")
    return {"tokenizer_file": resolved}


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    if image.dim() == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    array = (image.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


@DataTransformFn.register_subclass("internvla_a1_5_chat_processor")
@dataclass
class InternVLAA15ChatProcessorTransformFn(DataTransformFn):
    """Combined subtask + fast action chat processor for A1.5 policy.

    Supports four label modes:
      - BOTH: assistant = "sub_task: <text> ; Action: <fast tokens>", both supervised
      - FAST: assistant = fast action tokens only
      - TEXT: assistant = subtask text only
      - NONE: no supervision
    """

    pretrained_model_name_or_path: str = "Qwen/Qwen3.5-2B"
    max_length: int = 650
    task_key: str = "task"
    language_memory_key: str = "language_memory"
    num_views: int = 3
    truncation: bool = True
    padding: str = "max_length"

    tokenize_state: bool = True
    max_state_dim: int = 32

    use_fast_action_tokens: bool = True
    action_text_key: str = "action.action_text"
    action_token_min: int = 248077
    action_token_max: int = 250124
    mode: str = "train"

    action_mode: str = "joint"

    def hydrate(self, dataset: LeRobotDataset | StreamingLeRobotDataset) -> InternVLAA15ChatProcessorTransformFn:
        schema = get_schema(dataset.meta.robot_type)
        return replace(self, action_mode=getattr(schema, "action_mode", "joint"))

    def __post_init__(self):
        self.processor = Qwen3VLProcessor.from_pretrained(self.pretrained_model_name_or_path)
        ensure_qwen35_action_tokens(self.processor.tokenizer)
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

    def _encode_state(self, data: DataDict) -> str:
        if not self.tokenize_state or OBS_STATE not in data:
            return ""
        state = deepcopy(data[OBS_STATE])
        state = pad_vector(state, self.max_state_dim)
        state_np = state.cpu().numpy() / 3
        discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        return "State: " + " ".join(map(str, discretized))

    def __call__(self, data: DataDict) -> DataDict:
        state_str = self._encode_state(data)

        images, valid_image_indices = [], []
        for i in range(self.num_views):
            k = f"{OBS_IMAGES}.image{i}"
            if bool(data[f"{k}_mask"]):
                images.append(data[k])
                valid_image_indices.append(i)

        user_text = "Task: " + str(data.get(self.task_key, ""))
        lang_mem = str(data.get(self.language_memory_key, "")).strip()
        if lang_mem:
            user_text = user_text + "; " + lang_mem
        user_text = user_text + "; " + f"Control Mode: <{self.action_mode}>"
        if self.tokenize_state and state_str:
            user_text = user_text + "; " + state_str

        num_valid_images = len(valid_image_indices)
        user_content = [{"type": "image"} for _ in range(num_valid_images)]

        has_fast = (
            self.use_fast_action_tokens
            and self.action_text_key in data
            and str(data.get(self.action_text_key, "")).strip() != ""
        )
        has_sub_task = "sub_task" in data and str(data.get("sub_task", "")).strip() != ""

        if has_fast and has_sub_task:
            label_mode = LABEL_MODE_BOTH
        elif has_fast:
            label_mode = LABEL_MODE_FAST
        elif has_sub_task:
            label_mode = LABEL_MODE_TEXT
        else:
            label_mode = LABEL_MODE_NONE
        if self.mode == "eval":
            label_mode = LABEL_MODE_NONE
            user_text = user_text + "; Output: <Subtask, Action>"

        # Build assistant text
        if label_mode == LABEL_MODE_BOTH:
            assistant_text = f"{data['sub_task']}; {data[self.action_text_key]}"
            user_text = user_text + f"; Output: <SubTask, Action>"
        elif label_mode == LABEL_MODE_FAST:
            assistant_text = f"{data[self.action_text_key]}"
            user_text = user_text + f"; Output: <Action>"
        elif label_mode == LABEL_MODE_TEXT:
            assistant_text = f"{data['sub_task']}"
            user_text = user_text + f"; Output: <SubTask>"
        else:
            assistant_text = ""

        user_content.append({"type": "text", "text": user_text})

        # Tokenize
        if label_mode != LABEL_MODE_NONE:
            messages_full = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ]
            text_full = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )
            inputs_full = self.processor(
                text=[text_full], images=images, do_rescale=False,
                return_tensors="pt", padding="max_length",
                max_length=self.max_length, truncation=True,
            )
            input_ids = inputs_full.input_ids[0]
            attention_mask = inputs_full.attention_mask[0]
            labels = input_ids.clone()

            if label_mode == LABEL_MODE_FAST:
                # Only supervise action token positions
                act_mask = (input_ids >= self.action_token_min) & (input_ids <= self.action_token_max)
                act_pos = act_mask.nonzero(as_tuple=False)
                if act_pos.numel() > 0:
                    labels[:act_pos[0].item()] = -100
                else:
                    labels[:] = -100
            else:
                # BOTH or TEXT: supervise everything after the prompt
                messages_prompt = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ]
                text_prompt = self.processor.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True
                )
                inputs_prompt = self.processor(
                    text=[text_prompt], images=images, do_rescale=False,
                    return_tensors="pt", padding=False,
                )
                prompt_len = inputs_prompt.input_ids.shape[1]
                labels[:prompt_len] = -100

            labels[attention_mask == 0] = -100
        else:
            # NONE mode
            if self.mode == "train":
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ]
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs_full = self.processor(
                    text=[text_prompt], images=images, do_rescale=False,
                    return_tensors="pt", padding=self.padding,
                    max_length=self.max_length, truncation=self.truncation,
                )
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ]
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs_full = self.processor(
                    text=[text_prompt], images=images, do_rescale=False,
                    return_tensors="pt", padding=False,
                )
            input_ids = inputs_full.input_ids[0]
            attention_mask = inputs_full.attention_mask[0]
            labels = torch.full_like(input_ids, -100)

        # Disable attention for masked image slots
        vs_pos = (input_ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
        ve_pos = (input_ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
        for i in range(min(self.num_views, len(vs_pos), len(ve_pos))):
            if i not in valid_image_indices:
                start, end = vs_pos[i].item(), ve_pos[i].item()
                attention_mask[start:end + 1] = 0
                labels[start:end + 1] = -100

        # fast_token_mask: marks action token positions for suffix blocking
        fast_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if label_mode in (LABEL_MODE_FAST, LABEL_MODE_BOTH):
            act_pos = (
                (input_ids >= self.action_token_min) & (input_ids <= self.action_token_max)
            ).nonzero(as_tuple=False)
            if act_pos.numel() > 0:
                first_act = act_pos[0].item()
                fast_token_mask[first_act:] = attention_mask[first_act:].bool()

        # Output
        data[f"{OBS_STR}.input_ids"] = input_ids
        data[f"{OBS_STR}.attention_mask"] = attention_mask
        data[f"{OBS_STR}.pixel_values"] = inputs_full.pixel_values
        data[f"{OBS_STR}.image_grid_thw"] = inputs_full.image_grid_thw
        data[f"{OBS_STR}.fast_token_mask"] = fast_token_mask
        data["VQA.labels"] = labels
        data["vqa_type"] = torch.tensor(2 if label_mode != LABEL_MODE_NONE else 0, dtype=torch.long)
        data["label_mode"] = torch.tensor(label_mode, dtype=torch.long)
        return data


@DataTransformFn.register_subclass("internvla_a1_5_vqa_processor")
@dataclass
class InternVLAA15VQAProcessorTransformFn(DataTransformFn):
    """VQA processor used by InternVLA-A1.5 mixed robot/VQA training."""

    pretrained_model_name_or_path: str = "Qwen/Qwen3.5-2B"
    max_length: int = 650
    num_views: int = 3
    truncation: bool = True
    padding: str = "max_length"

    def __post_init__(self) -> None:
        self.processor = Qwen3VLProcessor.from_pretrained(self.pretrained_model_name_or_path)
        ensure_qwen35_action_tokens(self.processor.tokenizer)
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

    def _image_is_valid(self, data: DataDict, index: int) -> bool:
        robot_mask_key = f"{OBS_IMAGES}.image{index}_mask"
        if robot_mask_key in data:
            return bool(data[robot_mask_key])
        vqa_mask_key = f"mask{index}"
        if vqa_mask_key in data:
            return bool(data[vqa_mask_key])
        return f"{OBS_IMAGES}.image{index}" in data

    def _collect_images(self, data: DataDict) -> list[Image.Image]:
        images = []
        for i in range(self.num_views):
            key = f"{OBS_IMAGES}.image{i}"
            if self._image_is_valid(data, i) and key in data:
                images.append(tensor_to_pil_image(data[key]))
        if not images:
            raise ValueError("InternVLAA15VQAProcessorTransformFn requires at least one valid image.")
        return images

    def _build_user_text(self, conversation: list[dict]) -> tuple[str, str]:
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("InternVLAA15VQAProcessorTransformFn requires non-empty `conversation`.")

        user_text = str(conversation[0].get("content", ""))
        assistant_text = str(conversation[1].get("content", "")) if len(conversation) > 1 else ""
        user_text = re.sub(
            r"<\|vision_start\|><\|image_pad\|><\|vision_end\|>", "", user_text
        ).strip()
        user_text = user_text.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        return f"Task: {user_text}; Control Mode: <vqa>; Output: <Answer>", assistant_text

    def __call__(self, data: DataDict) -> DataDict:
        pil_images = self._collect_images(data)
        user_text, assistant_text = self._build_user_text(data.get("conversation", []))

        user_content = [{"type": "image"} for _ in pil_images]
        user_content.append({"type": "text", "text": user_text})
        messages_prompt = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
        messages_full = [
            *messages_prompt,
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_full = self.processor(
            text=[text_full],
            images=pil_images,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=pil_images,
            return_tensors="pt",
            padding=False,
        )

        input_ids = inputs_full.input_ids[0]
        attention_mask = inputs_full.attention_mask[0]
        labels = input_ids.clone()
        labels[: inputs_prompt.input_ids.shape[1]] = -100
        labels[attention_mask == 0] = -100

        data[f"{OBS_STR}.input_ids"] = input_ids
        data[f"{OBS_STR}.attention_mask"] = attention_mask
        data[f"{OBS_STR}.pixel_values"] = inputs_full.pixel_values
        data[f"{OBS_STR}.image_grid_thw"] = inputs_full.image_grid_thw
        data[f"{OBS_STR}.fast_token_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)
        data["VQA.labels"] = labels
        data["label_mode"] = torch.tensor(LABEL_MODE_TEXT, dtype=torch.long)
        return data


@DataTransformFn.register_subclass("fast_internvla_a1_5_action_tokenizer")
@dataclass
class FASTInternVLAA15ActionTokenizerTransformFn(DataTransformFn):
    
    action_tokenizer_name: str = "physical-intelligence/fast"
    qwen35_model_name: str = "Qwen/Qwen3.5-2B"
    max_action_tokens: int = 256
    chunk_size: int = 50
    max_action_dim: int = 32
    
    # Qwen3.5 special action token range
    action_token_min: int = 248077
    action_token_max: int = 250124
    
    assistant_end_tokens: list[int] = None
    
    stop_token_1: int = 248046  # First stop token
    stop_token_2: int = 198     # Second stop token (newline)
    
    def __post_init__(self):
        logging.info("Loading FAST tokenizer from %s", self.action_tokenizer_name)
        self.action_tokenizer = AutoProcessor.from_pretrained(
            self.action_tokenizer_name,
            trust_remote_code=True,
            **_fast_processor_kwargs(self.action_tokenizer_name),
        ) 
        self.action_tokenizer.time_horizon = self.chunk_size
        self.action_tokenizer.action_dim = self.max_action_dim
        # Load Qwen3.5 tokenizer
        self.qwen35_tokenizer = Qwen3_5Tokenizer.from_pretrained(
            self.qwen35_model_name
        )
        ensure_qwen35_action_tokens(self.qwen35_tokenizer)
        # Initialize assistant_end_tokens if not set
        if self.assistant_end_tokens is None:
            self.assistant_end_tokens = [248045, 74455, 198, 248068, 271, 248069, 271]

    def _act_tokens_to_qwen35_tokens(self, tokens: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        """
        Convert FAST action tokens to Qwen3.5 special action token range.
        
        Formula: action_token_min + fast_token
        This maps FAST tokens [0, 2047] to Qwen3.5 special range [action_token_min, action_token_max]
        
        Args:
            tokens: FAST token IDs
        
        Returns:
            Qwen3.5 special action token IDs
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)
        elif isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens)
        
        return self.action_token_min + tokens
    
    def _fast_tokens_to_text(self, fast_tokens: list[int]) -> str:
        """
        Convert FAST token IDs to text representation for chat template.

        Example: [0, 1, 2] -> "<robot_action_0><robot_action_1><robot_action_2>"

        Args:
            fast_tokens: List of FAST token IDs

        Returns:
            Text string with special token markers
        """
        return ''.join([f"<robot_action_{token}>" for token in fast_tokens])

    def extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> list[list[int]]:
        """
        Extract action tokens from the generated token sequence.
        
        Extracts tokens between:
        - Start: after assistant_end_tokens [248045, 74455, 198, 248068, 271, 248069, 271]
        - End: before first stop token [248046, 198]

        Adapted from StarVLA QwenFast._extract_action_token_ids()

        Args:
            generated_ids: Generated token sequence of shape (B, L)

        Returns:
            List of action token IDs for each batch: ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
            Rule: Extract tokens between assistant end markers and first stop token.
        """
        act_min = self.action_token_min
        act_max = self.action_token_max
        
        # Use config values
        assistant_end_marker = self.assistant_end_tokens[-1]
        stop_token_1 = self.stop_token_1  # 248046
        stop_token_2 = self.stop_token_2  # 198
        
        results = []
        for b in range(generated_ids.size(0)):
            tokens_list = generated_ids[b].tolist()
            
            # Find start position (after ASSISTANT_END)
            start_idx = -1
            for i, token in enumerate(tokens_list):
                if token == assistant_end_marker:
                    start_idx = i + 1  # Start from next token
                    break
            
            if start_idx == -1:
                # No assistant end marker found
                results.append([])
                continue
            
            # Find end position (before first STOP_TOKEN_1, STOP_TOKEN_2 sequence)
            end_idx = len(tokens_list)
            for i in range(start_idx, len(tokens_list) - 1):
                if tokens_list[i] == stop_token_1 and tokens_list[i + 1] == stop_token_2:
                    end_idx = i
                    break
            
            # Extract action tokens from the range [start_idx, end_idx)
            action_tokens = []
            for i in range(start_idx, end_idx):
                token = tokens_list[i]
                if act_min <= token <= act_max:
                    action_tokens.append(token)
            
            results.append(action_tokens)
        
        return results

    def decode_action_tokens(
        self,
        batch_vlm_tokens: list[list[int]],
    ) -> list[list[int] | None]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.

        Adapted from StarVLA QwenFast._decode_action_tokens()

        Args:
            batch_vlm_tokens: List of VLM action token sequences (with offset)

        Returns:
            List of FAST token ID sequences (without offset).
            None is returned for sequences with no action tokens.
            fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        act_min = self.action_token_min
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            # Subtract offset to get FAST token IDs
            fast_ids = [t - act_min for t in seq]
            batch_fast_token_ids.append(fast_ids)

        return batch_fast_token_ids

    def decode_action_tokens_to_actions(
        self,
        batch_fast_token_ids: list[list[int] | None],
    ) -> list[np.ndarray | None]:
        """
        Decode FAST token IDs back to continuous actions using FAST tokenizer.

        This is the inverse of the tokenization process in __call__.

        Args:
            batch_fast_token_ids: List of FAST token ID sequences (without offset)

        Returns:
            List of action arrays with shape (num_actions, action_dim).
            None is returned for sequences with no tokens or invalid tokens.
        """
        batch_actions = []
        for fast_ids in batch_fast_token_ids:
            if fast_ids is None or len(fast_ids) == 0:
                batch_actions.append(None)
                continue

            try:
                # Decode FAST tokens back to continuous actions
                # The FAST tokenizer should have a decode method
                actions = self.action_tokenizer.decode([fast_ids])
                batch_actions.append(actions[0])

            except Exception as e:
                logging.warning(f"Failed to decode FAST tokens to actions: {e}")
                batch_actions.append(None)

        return batch_actions

    def __call__(self, data: DataDict) -> DataDict:
        action = data[ACTION]
        device = action.device if isinstance(action, torch.Tensor) else torch.device("cpu")
        
        # Handle different action shapes (following LeRobot pattern)
        if action.dim() == 1:
            # Single timestep action, expand to chunk
            action = action.unsqueeze(0)
        
        chunk_size, action_dim = action.shape
        
        # Pad action dimension if needed
        if action_dim < self.max_action_dim:
            action = F.pad(action, (0, self.max_action_dim - action_dim))
        
        # Pad or truncate chunk_size if needed
        if chunk_size < self.chunk_size:
            # Pad by repeating the last action
            pad_len = self.chunk_size - chunk_size
            action = torch.cat([action, action[-1:].repeat(pad_len, 1)], dim=0)
        elif chunk_size > self.chunk_size:
            # Truncate to chunk_size
            action = action[:self.chunk_size]
        
        # Convert to numpy for FAST tokenizer (expects [batch, time, dim])
        action_np = action.cpu().numpy().astype(np.float32)
        action_np = action_np[np.newaxis, :, :]  # Add batch dimension: (1, chunk_size, action_dim)
        
        # Tokenize using FAST tokenizer (following StarVLA pattern)
        try:
            batch_fast_tokens = self.action_tokenizer(action_np)
            fast_tokens = batch_fast_tokens[0] if isinstance(batch_fast_tokens, list) else batch_fast_tokens
        except Exception as e:
            logging.warning(f"FAST tokenization failed: {e}. Using empty tokens.")
            fast_tokens = []
        
        if len(fast_tokens) == 0:
            # Create empty tokens if tokenization failed
            data["action.fast_tokens"] = torch.zeros(self.max_action_tokens, dtype=torch.long, device=device)
            data["action.qwen35_tokens"] = torch.zeros(self.max_action_tokens, dtype=torch.long, device=device)
            data["action.action_text"] = ""
            data["action.token_mask"] = torch.zeros(self.max_action_tokens, dtype=torch.bool, device=device)
        else:
            # Flatten if needed
            if isinstance(fast_tokens, torch.Tensor) and fast_tokens.dim() > 1:
                fast_tokens = fast_tokens.flatten()
            elif isinstance(fast_tokens, np.ndarray):
                fast_tokens = fast_tokens.flatten().tolist()
            
            # Convert to Qwen3.5 special action tokens
            qwen35_tokens = self._act_tokens_to_qwen35_tokens(fast_tokens)
            
            # Convert to text representation for chat template
            action_text = self._fast_tokens_to_text(fast_tokens if isinstance(fast_tokens, list) else fast_tokens.tolist())
            
            # Store results
            data["action.fast_tokens"] = torch.tensor(fast_tokens, dtype=torch.long, device=device)
            data["action.qwen35_tokens"] = qwen35_tokens if isinstance(qwen35_tokens, torch.Tensor) else torch.tensor(qwen35_tokens, dtype=torch.long, device=device)
            data["action.action_text"] = action_text
            data["action.token_mask"] = torch.ones(len(fast_tokens), dtype=torch.bool, device=device)
        
        return data


@DataTransformFn.register_subclass("extract_video_frames")
@dataclass
class ExtractVideoFramesTransformFn(DataTransformFn):
    """Extract multi-frame video data for WAN and reduce camera keys to single frame for VLM.

    When image_delta_indices produces [T, C, H, W] tensors per camera key,
    this transform saves the full sequence from the source view as
    'observation.video_frames' (for WAN supervision) and replaces all
    multi-frame camera keys with frame[0] only (for the Qwen VLM processor).
    """

    source_view: str = f"{OBS_IMAGES}.image0"
    video_key: str = "observation.video_frames"
    normalize_to_minus1_1: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        src = data[self.source_view]
        if src.ndim == 4:  # [T, C, H, W]
            video = src
            if self.normalize_to_minus1_1:
                video = video * 2.0 - 1.0
            data[self.video_key] = video

            for i in range(3):
                k = f"{OBS_IMAGES}.image{i}"
                if k in data and data[k].ndim == 4:
                    data[k] = data[k][0]
        return data
