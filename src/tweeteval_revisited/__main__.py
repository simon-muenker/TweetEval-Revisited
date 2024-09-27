import pathlib

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
        
        )
    )

    tweeteval_revisited.pipeline.Pipeline(
        data_train=dataset["train"],
        data_test=dataset["test"],
        encoder=encoder,
        classifier=tweeteval_revisited.neural.Classifier(
            input_size=encoder.hidden_size,
            out_size=2
        ),
        args=tweeteval_revisited.pipeline.PipelineArgs(
            epochs=50, batch_size=32, report_path=pathlib.Path(".")
        )
    )