import typing

import torch
import transformers
import datasets

import sklearn
import sklearn.metrics

import tweeteval_revisited as twer
import _config as cfg


def to_numeric(i: str) -> int:
    try:
        return int(i)

    except ValueError:
        return 0


if __name__ == "__main__":
    dataset = datasets.load_dataset(cfg.DATASET_SLUG, name=cfg.INSTANCE)

    pipe = transformers.pipeline(
        "text-generation",
        model=cfg.MODEL_SLUG,
        torch_dtype=torch.bfloat16,
        device_map=cfg.DEVICE,
    )

    metric: twer.pipeline.tracking.ValueContainer = (
        twer.pipeline.tracking.ValueContainer(label="acc")
    )

    for batch in twer.pipeline.util.get_data_loader(
        dataset["validation"], desc="Inferencing", collate_fn=cfg.collate
    ):
        src, tgt = batch

        preds: typing.List[int] = [
            to_numeric(response[0]["generated_text"][-1]["content"])
            for response in pipe(
                [chat.model_dump()["messages"] for chat in src], max_new_tokens=1
            )
        ]

        metric.add(sklearn.metrics.accuracy_score(tgt, preds))

    print(metric.mean)
