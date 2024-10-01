import torch
import typing

import rich
import rich.progress


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    desc: str,
    batch_size: int = 32,
    collate_fn: typing.Callable = lambda x: x,
):
    return rich.progress.track(
        torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
        description=desc,
        transient=True,
    )
