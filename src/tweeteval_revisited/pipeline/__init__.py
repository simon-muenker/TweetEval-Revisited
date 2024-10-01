import pathlib
import typing
import logging

import datasets

import pydantic
import sklearn
import sklearn.metrics
import torch

from tweeteval_revisited import neural
from tweeteval_revisited.pipeline import tracking, util


class PipelineArgs(pydantic.BaseModel):
    epochs: int = 50
    batch_size: int = 64

    report_path: pathlib.Path = pathlib.Path(".")


class Pipeline(pydantic.BaseModel):
    data_train: datasets.Dataset
    data_test: datasets.Dataset

    generator: neural.Generator
    classifier: neural.Classifier
    objective: torch.nn.Module
    optimizer: torch.optim.Optimizer
    collate_fn: typing.Callable

    args: PipelineArgs = PipelineArgs()
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __call__(self):
        tracker: tracking.Tracker = tracking.Tracker(report_path=self.args.report_path)

        try:
            for n in range(1, self.args.epochs + 1):
                epoch: tracking.Epoch = tracking.Epoch(n=n)

                for batch in self._get_data_loader(
                    self.data_train, desc="training ..."
                ):
                    epoch.add_observation("train", *self._step(*batch, optimize=True))

                with torch.no_grad():
                    for batch in self._get_data_loader(
                        self.data_test, desc="testing ..."
                    ):
                        epoch.add_observation("test", *self._step(*batch))

                tracker.add(epoch)

        except KeyboardInterrupt:
            logging.warning("training interrupted by user. shutting down.")

    def _step(
        self,
        src: typing.List[str],
        tgt: torch.Tensor,
        optimize: bool = False,
    ) -> typing.Tuple[float, float]:
        with torch.no_grad():
            _, embeds, attention_mask = self.generator.embed(src)
            targets = torch.tensor(
                tgt, device=next(self.classifier.parameters()).device
            )

        with torch.enable_grad():
            preds = self.classifier.forward(embeds, mask=attention_mask)
            loss: torch.Tensor = self.objective(preds, targets)

            if optimize:
                self._optimize(loss)

        metric: float = sklearn.metrics.accuracy_score(
            targets.cpu(), torch.argmax(preds, dim=1).cpu()
        )

        return loss.item(), metric

    def _optimize(self, loss: torch.Tensor):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _get_data_loader(self, dataset: datasets.Dataset, desc: str) -> typing.Dict:
        return util.get_data_loader(
            dataset,
            desc=desc,
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
        )


__all__ = ["Pipeline", "PipelineArgs", "tracking", "util"]
