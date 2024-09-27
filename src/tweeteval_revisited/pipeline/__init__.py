import pathlib
import typing

import datasets

import pydantic
import rich
import rich.progress
import torch

from tweeteval_revisited import neural
from tweeteval_revisited.pipeline.tracker import Epoch, Tracker


class PipelineArgs(pydantic.BaseModel):
    epochs: int = 50
    batch_size: int = 64

    optimizer_config: typing.Dict = dict(
        lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    )

    report_path: pathlib.Path = pathlib.Path(".")


class Pipeline(pydantic.BaseModel):
    data_train: datasets.Dataset
    data_test: datasets.Dataset

    encoder: neural.Encoder
    classifier: neural.Classifier
    objective: torch.nn.Module

    args: PipelineArgs = PipelineArgs()
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __call__(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), **self.args.optimizer_config
        )

        tracker: Tracker = Tracker(report_path=self.args.report_path)

        for n in range(1, self.args.epochs + 1):
            epoch = Epoch(n=n)

            for batch in rich.progress.track(
                self._get_data_loader(self.data_train), "Training ...", transient=True
            ):
                epoch.add_loss_train(self._step(*batch, optimizer))

            with torch.no_grad():
                for batch in rich.progress.track(
                    self._get_data_loader(self.data_test), "Testing ...", transient=True
                ):
                    epoch.add_loss_test(self._step(*batch))

            tracker.add(epoch)

    def _step(
        self,
        src: typing.List[str],
        tgt: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        with torch.no_grad():
            _, embeds = self.encoder.embed(src)

        preds = self.classifier.forward(embeds)

        loss: torch.Tensor = self.objective(preds, tgt)

        if optimizer:
            self._optimize(loss, optimizer)

        return loss.item()

    def _optimize(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _get_data_loader(self, dataset: torch.utils.data.Dataset):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            collate_fn=self._collate,
        )

    def _collate(
        self, batch: typing.List[typing.Dict]
    ) -> typing.Tuple[typing.List[str], torch.Tensor]:
        src, tgt = [], []

        for sample in batch:
            src.append(sample["text"])
            tgt.append(sample["label"])

        return src, torch.tensor(tgt, device=next(self.classifier.parameters()).device)


__all__ = ["Pipeline", "PipelineArgs"]
