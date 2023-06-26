import json
import pathlib as pl
from datetime import datetime
from typing import List, Tuple
import argparse


def get_replies(data: dict | List[dict]) -> List[Tuple[str, str]]:
    """
    Get all author replies from a comment tree
    Creates a list of directed edges (author_reply, author)

    param
    ------
    data: dict or List[dict]
        comment tree

    return
    ------
    author_replies: List[tuple]
        list of directed edges (author_reply, author)
    """
    author_replies = []

    if isinstance(data, list):
        for item in data:
            author_replies.extend(get_replies(item))
    elif isinstance(data, dict):
        if "author" in data:
            if data["replies"]:  # if list of replies is not empty
                author = data["author"]
                for r in range(len(data["replies"])):
                    author_reply = data["replies"][r]["author"]
                    if author != "DELETED" and author_reply != "DELETED":
                        author_replies.append((author_reply, author))

                author_replies.extend(get_replies(data["replies"]))
        else:
            for value in data.values():
                author_replies.extend(get_replies(value))
    return author_replies


def parse_edge_list(
    thread_dir: str,
    from_date: str = "2022-01-01",
    to_date: str = "2022-12-31",
) -> List[Tuple[str, str]]:
    """
    Parse edge list from thread directory

    param
    ------
    thread_dir: str
        path to thread directory
    from_date: str
        start date in YYYY-MM-DD format
    to_date: str
        end date in YYYY-MM-DD format

    return
    ------
    edge_list: List[tuple]
    """

    edge_list = []

    assert pl.Path(thread_dir).exists(), "Thread directory does not exist"

    from_date = datetime.strptime(from_date, "%Y-%m-%d")
    to_date = datetime.strptime(to_date, "%Y-%m-%d")

    for thread_path in pl.Path(thread_dir).iterdir():
        with open(thread_path, "r") as f:
            thread = json.load(f)
            created_at = datetime.fromtimestamp(thread["created_utc"])

            # Check if thread is within date range
            if created_at >= from_date and created_at <= to_date:
                # Get author of thread
                thread_author = thread["author"]

                for comment in thread["comments"]:
                    comment_author = comment["author"]
                    if thread_author != "DELETED" and comment_author != "DELETED":
                        edge_list.append((thread_author, comment_author))
                    edge_list.extend(get_replies(comment))

    return edge_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thread_dir",
        type=str,
        default="data/svenskpolitik",
        help="Path to thread dir",
    )
    parser.add_argument("--from_date", type=str, default="2022-01-01")
    parser.add_argument("--to_date", type=str, default="2022-12-31")
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    edge_list = parse_edge_list(args.thread_dir, args.from_date, args.to_date)

    with open(args.output, "w") as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Edge list written to {args.output}")
    print(f"Number of edges: {len(edge_list)}")
    print(
        f"Number of unique nodes: {len(set([edge[0] for edge in edge_list] + [edge[1] for edge in edge_list]))}"
    )


if __name__ == "__main__":
    main()
