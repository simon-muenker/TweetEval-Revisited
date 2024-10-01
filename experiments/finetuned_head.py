import pathlib

import torch
import datasets
import transformers

import tweeteval_revisited as twer
import _config as cfg


if __name__ == "__main__":
    dataset = datasets.load_dataset(cfg.DATASET_SLUG, name=cfg.INSTANCE)

    generator = twer.neural.Generator(
        tokenizer=transformers.AutoTokenizer.from_pretrained(cfg.MODEL_SLUG),
        model=transformers.AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_SLUG,
            torch_dtype="auto",
            device_map=cfg.DEVICE,
        ),
    )
    classifier = classifier = twer.neural.Classifier(
        input_size=generator.hidden_size,
        out_size=dataset["train"].features["label"].num_classes,
    ).to(torch.device(cfg.DEVICE))

    twer.pipeline.Pipeline(
        data_train=dataset["validation"],
        data_test=dataset["validation"],
        generator=generator,
        classifier=classifier,
        objective=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(classifier.parameters()),
        collate_fn=cfg.collate,
        args=twer.pipeline.PipelineArgs(
            epochs=50, batch_size=16, report_path=pathlib.Path(".")
        ),
    )()
