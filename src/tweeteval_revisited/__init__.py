import datasets

DATASET_SLUG: str = "cardiffnlp/tweet_eval"
INSTANCE: str = "sentiment"


dataset = datasets.load_dataset(DATASET_SLUG, name=INSTANCE)
print(dataset)
