import logging

import pydantic
import torch

from tweeteval_revisited.neural import util


class ClassifierArgs(pydantic.BaseModel):
    num_layers: int = 8
    num_heads: int = 4

    dtype: torch.dtype = torch.bfloat16

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class Classifier(torch.nn.Module):
    def __init__(
        self, input_size: int, out_size: int, args: ClassifierArgs = ClassifierArgs()
    ):
        super().__init__()
        self.args: ClassifierArgs = args

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=input_size,
                batch_first=True,
                nhead=args.num_heads,
                dtype=args.dtype,
            ),
            num_layers=self.args.num_layers,
        )
        self.linear = torch.nn.Linear(input_size, out_size, dtype=args.dtype)

        for msg in (
            f"trainable params: {util.get_model_trainable_params(self):,d}",
            f"memory usage: {util.calculate_model_memory_usage(self)}",
        ):
            logging.info(f"CLASSIFIER | {msg}")

    def forward(self, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logging.debug(f"CLASSIFIER | forward input: {batch.size()}")

        hidden: torch.Tensor = self.encoder(
            batch, src_key_padding_mask=mask.to(torch.bfloat16)
        )
        hidden = hidden[:, 0, :]
        logging.debug(f"CLASSIFIER | encoder hidden: {hidden.size()}")

        hidden = self.linear(hidden)
        logging.debug(f"CLASSIFIER | linear output: {hidden.size()}")

        return hidden
