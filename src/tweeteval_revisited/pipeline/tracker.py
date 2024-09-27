import datetime
import pathlib
import typing

import pandas
import pydantic
import rich


class ValueContainer(pydantic.BaseModel):
    label: str
    values: typing.List[float] = []

    def add(self, val: float):
        self.values.append(val)

    @pydantic.computed_field
    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @pydantic.model_serializer()
    def serialize_model(self) -> float:
        return self.mean

    def to_series(self, iteration: int = 1) -> pandas.DataFrame:
        return pandas.Series(
            data=self.values,
            index=list(
                range(1 + len(self.values) * (iteration - 1), len(self.values) + 1)
            ),
            name=self.label,
        )


class Epoch(pydantic.BaseModel):
    n: int

    train: ValueContainer = ValueContainer(label="train")
    test: ValueContainer = ValueContainer(label="test")

    _start_time: datetime.datetime = pydantic.PrivateAttr(
        default_factory=datetime.datetime.now
    )
    _end_time: datetime.datetime | None = pydantic.PrivateAttr(default=None)

    @pydantic.computed_field
    @property
    def duration(self) -> datetime.timedelta:
        if self._end_time:
            return self._end_time - self._start_time

        else:
            return datetime.datetime.now() - self._start_time

    def add_loss_train(self, loss: float):
        self.train.add(loss)

    def add_loss_test(self, loss: float):
        self.test.add(loss)

    def end(self):
        self._end_time = datetime.datetime.now()
        self.log()

    def log(self):
        rich.print(
            f"[{self.n:03d}]\t",
            f"loss(train): {self.train.mean:2.4f}\t",
            f"loss(test): {self.test.mean:2.4f}\t",
            f"duration: {self.duration}",
        )

    def to_df(self) -> pandas.DataFrame:
        return self.train.to_series(self.n).join(
            self.test.to_series(self.n), how="outer"
        )


class Tracker(pydantic.BaseModel):
    epochs: typing.List[Epoch] = []

    report_path: pathlib.Path = pathlib.Path(".")
    reporth_name: str = "train.tracking"

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __del__(self):
        if self.epochs:
            self.to_df().to_json(
                self.report_path / f"{self.reporth_name}.json",
                orient="records",
                indent=4,
            )

    def add(self, epoch: Epoch) -> None:
        epoch.end()
        self.epochs.append(epoch)

    def to_df(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.model_dump()["epochs"])
