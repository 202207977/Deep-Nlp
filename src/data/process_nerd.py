import pandas as pd
from transformers import pipeline

NERD_CSV_PATH = "conll2003_sentences.csv"


def transform_ner_dataset(sentiment_analyzer, path: str) -> None:
    """
    Transforms a NER dataset by applying sentiment analysis.

    Params:
        path : str
            Path to the CSV file containing the dataset.

    """

    df = pd.read_csv(path)

    df["tokens"] = (
        df["tokens"].apply(eval) if isinstance(df["tokens"][0], str) else df["tokens"]
    )

    df["sentiment"] = df["tokens"].apply(
        lambda x: sentiment_analyzer(" ".join(x))[0]["label"]
    )

    df.to_csv("dataset.csv", index=False)


if __name__ == "__main__":
    sentiment_analyzer = pipeline("sentiment-analysis")
    transform_ner_dataset(sentiment_analyzer, NERD_CSV_PATH)
