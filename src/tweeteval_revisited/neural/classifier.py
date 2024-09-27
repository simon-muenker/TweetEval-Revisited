import typing

import pydantic
import torch


class ClassifierArgs(pydantic.BaseModel):
    encoder_num_layers: int = 8
    encoder_layers_config: typing.Dict = dict(
        nhead=4, batch_first=True, dtype=torch.bfloat16
    )

    lstm_config: typing.Dict = dict(
        num_layers=4, batch_first=True, dtype=torch.bfloat16
    )

    linear_config: typing.Dict = dict(dtype=torch.bfloat16)


class Classifier(torch.nn.Module):
    def __init__(
        self, input_size: int, out_size: int, args: ClassifierArgs = ClassifierArgs()
    ):
        super().__init__()
        self.args: ClassifierArgs = args

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=input_size, **self.args.encoder_layers_config
            ),
            num_layers=self.args.encoder_num_layers,
        )
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=input_size, **self.args.lstm_config
        )
        self.linear = torch.nn.Linear(input_size, out_size, **self.args.linear_config)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.encoder(batch)
        _, (hidden, _) = self.lstm(hidden)
        hidden = hidden[-1]
        hidden = self.linear(hidden)

        return hidden
