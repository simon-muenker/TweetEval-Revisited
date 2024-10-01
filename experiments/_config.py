import typing
import logging

import torch

import cltrier_lib


logging.getLogger().setLevel(logging.INFO)


DATASET_SLUG: str = "cardiffnlp/tweet_eval"
INSTANCE: str = "offensive"

# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
MODEL_SLUG: str = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE: str = "cuda:0"


def collate(
    batch: typing.List[typing.Dict],
) -> typing.Tuple[typing.List[cltrier_lib.inference.schemas.Chat], torch.Tensor]:
    src, tgt = [], []

    for sample in batch:
        src.append(
            cltrier_lib.inference.schemas.Chat(
                messages=[
                    cltrier_lib.inference.schemas.Message(
                        role="system",
                        content="You will be given a social media post. Your task is to provide an 'offensiveness rating' scoring if the social media post contains offensive content. Give your answer as either 0 (not-offensive) or 1 (offensive).\n\nHere is the scale you should use to build your answer:\n0: The social media post does not contain offensive content. 1: The social media post contains offensive content.\n\nProvide in your feedback only the numeric value.",
                    ),
                    cltrier_lib.inference.schemas.Message(
                        role="user",
                        content=f"Social Media Post: {sample["text"]}",
                    ),
                ]
            )
        )
        tgt.append(sample["label"])

    return src, tgt
