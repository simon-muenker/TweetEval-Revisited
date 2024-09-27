import pathlib

import torch
import datasets
import transformers

import tweeteval_revisited


DATASET_SLUG: str = "cardiffnlp/tweet_eval"
INSTANCE: str = "sentiment"

MODEL_SLUG: str = "meta-llama/Llama-3.2-1B"
DEVICE: str = "cuda:3"

if __name__ == "__main__":
    dataset = datasets.load_dataset(DATASET_SLUG, name=INSTANCE)
    encoder = tweeteval_revisited.neural.Encoder(
        tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL_SLUG),
        model=transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_SLUG,
            torch_dtype="auto",
            device_map=DEVICE,
        ),
    )

    tweeteval_revisited.pipeline.Pipeline(
        data_train=dataset["test"],
        data_test=dataset["validation"],
        encoder=encoder,
        classifier=tweeteval_revisited.neural.Classifier(
            input_size=encoder.hidden_size,
            out_size=dataset["train"].features["label"].num_classes,
        ).to(torch.device(DEVICE)),
        objective=torch.nn.CrossEntropyLoss(),
        args=tweeteval_revisited.pipeline.PipelineArgs(
            epochs=5, batch_size=128, report_path=pathlib.Path(".")
        ),
    )()
