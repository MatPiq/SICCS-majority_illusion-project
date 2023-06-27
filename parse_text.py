import json
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from datetime import datetime
import pathlib as pl
import pandas as pd
import spacy
from nltk.corpus import stopwords


def extract_replies(data: dict) -> List[Tuple[str, str]]:
    """
    Recursively extracts all replies from a comment.

    param
    ------
    data: dict
        Dictionary containing the comment data.

    return
    ------
    results: List[Tuple[str, str]]
        List of tuples containing the reply text and the user.
    """

    results = []
    if "replies" in data and data["replies"]:
        for reply in data["replies"]:
            reply_body = reply["body"]
            reply_author = reply["author"]
            results.append((reply_body, reply_author))
            results.extend(extract_replies(reply))
    return results


def get_user_texts(
    thread_dir: str = "../reddit/svenskpolitik/",
    from_date="2022-01-31",
    to_date="2022-12-31",
) -> Dict[str, str]:
    """
    Extracts all texts from a given thread directory and returns a dictionary
    with the user as key and all texts as value.

    param
    ------
    thread_dir: str
        Path to the directory containing the threads.
    from_date: str
        Date in format YYYY-MM-DD
    to_date: str
        Date in format YYYY-MM-DD

    return
    ------
    user_texts: Dict[str, str]
        Dictionary with user as key and all texts as value.
    """

    user_texts = defaultdict(str)

    from_date = datetime.strptime(from_date, "%Y-%m-%d")
    to_date = datetime.strptime(to_date, "%Y-%m-%d")

    for thread_path in pl.Path(thread_dir).iterdir():
        with open(thread_path, "r") as f:
            thread = json.load(f)
            created_at = datetime.fromtimestamp(thread["created_utc"])
            thread_author = thread["author"]
            user_texts[thread_author] += thread["selftext"]

            if created_at >= from_date and created_at <= to_date:
                # users = []
                for comment in thread["comments"]:
                    comment_author = comment["author"]
                    user_texts[comment_author] += comment["body"]

                    replies = extract_replies(comment)

                    for reply, user in replies:
                        user_texts[user] += reply

    # Remove empty texts and deleted users
    user_texts = {k: v for k, v in user_texts.items() if v and k != "DELETED"}

    return user_texts


def create_docs(user_texts: Dict[str, str]) -> List[List[str]]:
    """
    Creates a list of documents from a dictionary

    param
    ------
    user_texts: Dict[str, str]
        Dictionary with user as key and all texts as value.

    return
    ------
    docs: List[List[str]]
        List of documents. Each document is a list of words.
    """

    nlp = spacy.load("sv_core_news_sm")
    swedish_stopwords = stopwords.words("swedish")

    docs = []
    for text in user_texts.values():
        doc = nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.lower_ not in swedish_stopwords
        ]
        docs.append(tokens)

    return docs


def create_dfm(docs: List[List[str]], doc_names: List[str]) -> pd.DataFrame:
    """
    Creates a document frequency matrix from a list of documents.

    param
    ------
    docs: List[List[str]]
        List of documents. Each document is a list of words.
    doc_names: List[str]
        List of document names.

    return
    ------
    dfm: pd.DataFrame
        Document frequency matrix. Each row is a document and each column is a
        word. Column names are the vocabulary and row names are the document
        names (user names).
    """
    vocab = []
    for doc in docs:
        vocab.extend(list(doc))

    vocab = set(vocab)
    # print(len(vocab))
    voc2idx = {w: i for i, w in enumerate(vocab)}

    dfm = np.zeros(shape=(len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        for token in doc:
            dfm[i, voc2idx[token]] += 1

    return pd.DataFrame(dfm, columns=list(vocab), index=doc_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thread_dir",
        type=str,
        default="data/svenskpolitik/",
        help="Path to the directory containing the threads.",
    )
    args = parser.parse_args()

    print("Parsing texts...")
    # get the texts for each user
    user_texts = get_user_texts(args.thread_dir)

    # create and save the cleaned texts
    docs = create_docs(user_texts)

    dfm = create_dfm(docs, list(user_texts.keys()))
    print(
        f"Saving dfm with vocabulary size {dfm.shape[1]} and {dfm.shape[0]} documents to data/dfm.csv"
    )
    dfm.to_csv("data/dfm.csv", index=False)

    # Save the cleaned texts as json
    print("Saving processed texts in data/processed_user_texts.json")
    with open("data/processed_user_texts.json", "w") as f:
        json.dump(user_texts, f)


if __name__ == "__main__":
    main()
