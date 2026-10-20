"""API 请求体模型（Pydantic v2）。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelIn(BaseModel):
    display_name: str = ""
    model_id: str = ""
    source: str = "auto"
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    max_seq_length: int = Field(default=2048, ge=0)
    chat_template: Optional[str] = None


class ModelUpdateIn(ModelIn):
    original_name: str = ""


class RenameIn(BaseModel):
    old: str
    new: str


class InspectIn(BaseModel):
    kind: Literal["upload", "hf", "existing"] = "upload"
    upload_id: str = ""
    hf_id: str = ""
    split: str = "train"
    existing_name: str = ""


class MappingIn(BaseModel):
    token: str
    instruction_cols: list[str] = Field(default_factory=list)
    input_cols: list[str] = Field(default_factory=list)
    think_cols: list[str] = Field(default_factory=list)
    output_cols: list[str] = Field(default_factory=list)
    fixed_instruction: str = ""


class GenerateIn(MappingIn):
    output_name: str = ""


class TrainIn(BaseModel):
    experiment_name: str = ""
    resume_training: bool = False
    truncate_dataset: bool = False
    max_samples: int = Field(default=200, ge=1)
    training_mode: str = "steps"
    num_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=50, ge=1)
    selected_model_name: str = ""
    selected_dataset_names: list[str] = Field(default_factory=list)
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=8, ge=1)
    lr: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=2048, ge=256)


class LoadIn(BaseModel):
    base_model_name: str
    lora_name: str
