import typing

import torch
import transformers


transformers.logging.set_verbosity_error()


class Encoder(torch.nn.Module):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        model: transformers.AutoModelForCausalLM,
    ):
        super().__init__()

        self.tokenizer: transformers.AutoTokenizer = tokenizer
        self.model: transformers.AutoModelForCausalLM = model

        # fix: ValueError: Asking to pad but the tokenizer does not have a padding token.
        # src: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/76
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(
        self, batch: typing.List[str]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(next(self.model.parameters()).device)

    def prepare(
        self, batch: typing.List[str]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenize(batch)

    def decode(self, batch: typing.List[torch.Tensor]) -> typing.List[str]:
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def embed(self, batch: typing.List[str]) -> torch.Tensor:
        model_inputs = self.prepare(batch)
        outputs = self.model(**model_inputs, output_hidden_states=True)

        return outputs.logits, outputs.hidden_states[-1]

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size
