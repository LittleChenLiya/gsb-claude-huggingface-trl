# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

"""Tests for SFT dataset filtering statistics (`SFTConfig.dataset_filter_stats`).

All fixtures are built locally (tiny word-level tokenizer and tiny Llama model), so no network access is required.
"""

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from trl import SFTConfig, SFTTrainer


# Renders the message contents as-is, and fails (Jinja `UndefinedError`) for the sentinel content, which simulates a
# chat template failure for a single example.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['content'] == 'TRIGGER_TEMPLATE_ERROR' %}{{ message['nonexistent_key'].attribute }}{% endif %}"
    "{{ message['content'] }}"
    "{% endfor %}"
)

WORDS = ["alpha", "beta", "gamma", "delta", "zeta", "lorem", "TRIGGER_TEMPLATE_ERROR"]

MAX_LENGTH = 16


def make_tokenizer():
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({word: i + 3 for i, word in enumerate(WORDS)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def make_model(vocab_size):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def make_trainer(tmp_path, samples, dataset_filter_stats):
    tokenizer = make_tokenizer()
    model = make_model(len(tokenizer.get_vocab()))
    dataset = Dataset.from_list([{"messages": [{"role": "user", "content": content}]} for content in samples])
    args = SFTConfig(
        output_dir=str(tmp_path),
        max_length=MAX_LENGTH,
        packing=False,
        shuffle_dataset=False,
        dataset_filter_stats=dataset_filter_stats,
        bf16=False,
        gradient_checkpointing=False,
        report_to=[],
        save_strategy="no",
    )
    return SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)


def kept_contents(trainer):
    return [messages[0]["content"] for messages in trainer.train_dataset["messages"]]


def test_filter_stats_enabled(tmp_path):
    samples = [
        "alpha beta",  # kept
        "   ",  # dropped: empty text
        "lorem " * 20,  # dropped: overlong (20 tokens > max_length=16)
        "TRIGGER_TEMPLATE_ERROR",  # dropped: chat template failure
        "gamma delta",  # kept
    ]
    trainer = make_trainer(tmp_path, samples, dataset_filter_stats=True)

    # Kept examples preserve their original relative order
    assert kept_contents(trainer) == ["alpha beta", "gamma delta"]
    # Only counts are recorded, per category
    assert trainer.dataset_filter_stats == {
        "train": {"total": 5, "kept": 2, "empty_text": 1, "overlong": 1, "template_error": 1}
    }


def test_filter_stats_disabled_behavior_unchanged(tmp_path):
    samples = [
        "alpha beta",  # kept as-is
        "zeta",  # kept as-is
        "lorem " * 20,  # kept, truncated to max_length (not dropped)
        "gamma delta",  # kept as-is
    ]
    trainer = make_trainer(tmp_path, samples, dataset_filter_stats=False)

    # All samples are kept, in the original order; the overlong one is truncated, not dropped
    assert kept_contents(trainer) == ["alpha beta", "zeta", "lorem " * 20, "gamma delta"]
    assert len(trainer.train_dataset[2]["input_ids"]) == MAX_LENGTH
    # No statistics are recorded when the flag is off
    assert trainer.dataset_filter_stats == {}


def test_template_failure_raises_when_stats_disabled(tmp_path):
    samples = ["alpha beta", "TRIGGER_TEMPLATE_ERROR"]
    with pytest.raises(Exception, match="nonexistent_key"):
        make_trainer(tmp_path, samples, dataset_filter_stats=False)
