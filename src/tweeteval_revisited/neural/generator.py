import typing
import logging

import cltrier_lib
import torch
import transformers

from tweeteval_revisited.neural import util

transformers.logging.set_verbosity_error()


class Generator(torch.nn.Module):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        model: transformers.AutoModelForCausalLM,
        freeze: bool = True,
    ):
        super().__init__()

        self.tokenizer: transformers.AutoTokenizer = tokenizer
        self.model: transformers.AutoModelForCausalLM = model

        # fix: ValueError: Asking to pad but the tokenizer does not have a padding token.
        # src: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/76
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        for msg in (
            f"frozen: {freeze}",
            f"params: {util.get_model_params(self):,d}",
            f"memory usage: {util.calculate_model_memory_usage(self)}",
            f"model type: {self.model.config.model_type}",
        ):
            logging.info(f"GENERATOR | {msg}")

    def format_chat(
        self, batch: typing.List[cltrier_lib.inference.schemas.Chat]
    ) -> typing.List[str]:
        return [
            self.tokenizer.apply_chat_template(
                chat.model_dump()["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for chat in batch
        ]

    def tokenize(
        self, batch: typing.List[str]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(next(self.model.parameters()).device)

    def prepare(
        self, batch: typing.List[cltrier_lib.inference.schemas.Chat]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenize(self.format_chat(batch))

    def decode(self, batch: typing.List[torch.Tensor]) -> typing.List[str]:
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def embed(self, batch: typing.List[str]) -> torch.Tensor:
        inputs: transformers.tokenization_utils_base.BatchEncoding = self.prepare(batch)
        outputs: transformers.modeling_outputs.CausalLMOutputWithPast = self.model(
            **inputs, output_hidden_states=True
        )

        return outputs.logits, outputs.hidden_states[-1], inputs.attention_mask

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size
