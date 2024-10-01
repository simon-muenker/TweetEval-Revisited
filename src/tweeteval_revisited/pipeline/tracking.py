import datetime
import pathlib
import typing

import pandas
import pydantic
import rich


class ValueContainer(pydantic.BaseModel):
    label: str | typing.Tuple[str, str]
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

    observations: typing.Dict[typing.Tuple[str, str], ValueContainer] = {
        label: ValueContainer(label=label)
        for label in [
            ("loss", "train"),
            ("loss", "test"),
            ("acc", "train"),
            ("acc", "test"),
        ]
    }

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

    def add_observation(
        self, split: typing.Literal["train", "test"], loss: float, metric: float
    ):
        self.observations[("loss", split)].add(loss)
        self.observations[("acc", split)].add(metric)

    def end(self):
        self._end_time = datetime.datetime.now()
        self.log()

    def log(self):
        rich.print(
            f"[{self.n:03d}]\t",
            *[
                f"{label[0]}({label[1]}): {data.mean:2.4f}\t"
                for label, data in self.observations.items()
            ],
            f"duration: {self.duration}",
        )

    def to_df(self) -> pandas.DataFrame:
        return pandas.concat(
            [
                data.to_series(self.n).rename(label)
                for label, data in self.observations.items()
            ],
            axis=1,
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


__all__ = ["Tracker", "Epoch", "ValueContainer"]
