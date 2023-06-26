import numpy as np
import argparse
from typing import Tuple
import pathlib as pl
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


# Check if figs directory exists
if not pl.Path("figs").exists():
    pl.Path("figs").mkdir()


def create_graph(
    edge_list_path: str = "data/edgelist.txt",
) -> Tuple[nx.DiGraph, nx.Graph]:
    """Create a graph from an edge list.

    param
    -----
    edge_list_path: str
        Path to the edge list.

    returns
    -------
    G_dir:
        nx.DiGraph: Directed graph.
    G_undir:
        nx.Graph: Undirected graph.
    """
    G_dir = nx.DiGraph()
    with open(edge_list_path, "r") as f:
        for line in f:
            node1, node2 = line.strip().split()
            G_dir.add_edge(node1, node2)
    return G_dir, G_dir.to_undirected()


def plot_degree_dist(G: nx.Graph) -> None:
    # Get degree distribution
    counts = Counter(G.degree[node] for node in G.nodes)

    fig, axs = plt.subplots(nrows=2, dpi=300, figsize=(8, 8))
    axs[0].plot(counts.keys(), counts.values(), "ko")
    axs[0].set_xlabel("Degree")
    axs[0].set_ylabel("Freq.")

    axs[1].loglog(counts.keys(), counts.values(), "ko")
    axs[1].set_xlabel("Degree (log)")
    axs[1].set_ylabel("Freq. (log)")

    fig.suptitle("Degree Distribution")
    fig.tight_layout()
    plt.savefig("figs/degree_distribution.png")


def plot_top_k_degree(G: nx.DiGraph, k: int = 50) -> None:
    indegrees = sorted(
        {node: G.in_degree[node] for node in G.nodes}.items(), key=lambda item: item[1]
    )
    outdegrees = sorted(
        {node: G.out_degree[node] for node in G.nodes}.items(), key=lambda item: item[1]
    )

    fig, axs = plt.subplots(ncols=2, dpi=500, figsize=(10, 10))

    for user, degree in indegrees[-k:]:
        axs[0].plot(degree, user, "ko")
    axs[0].set_xlabel("In degree")

    for user, degree in outdegrees[-k:]:
        axs[1].plot(degree, user, "ko")
    axs[1].set_xlabel("Out degree")

    fig.suptitle("Top degree users")
    fig.tight_layout()

    plt.savefig("figs/top_degrees.png")


def calculate_majority_illusion(
    G: nx.Graph, topic_dist_path: str = "data/user-topic_distribution.csv"
) -> np.ndarray:
    # Get largest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    # Get topic Distribution
    df = pd.read_csv(topic_dist_path, index_col=0)

    # Get users
    users = df.index.tolist()

    percieved_topic_dist = []
    for node in G.nodes:
        G_ego = nx.ego_graph(G, node)

        # Get topic distribution of neighbors
        # check that neighbors are in users and not the node itself
        tmp_dists = np.array(
            [
                df.loc[neighbor].to_numpy().squeeze()
                for neighbor in G_ego.nodes
                if neighbor in users and neighbor != node
            ]
        )

        if len(tmp_dists) > 1:
            percieved_topic_dist.append(np.mean(tmp_dists, axis=0))

        percieved_topic_dist.append(np.mean(tmp_dists, axis=0))

    print(percieved_topic_dist)

    # Calculate majority illusion

    global_topic_dist = df.to_numpy().mean(axis=0)

    return percieved_topic_dist - global_topic_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_list_path",
        type=str,
        default="data/edgelist.txt",
        help="Path to the edge list.",
    )
    parser.add_argument(
        "--topic_dist_path",
        type=str,
        default="data/user-topic_distribution.csv",
        help="Path to the user-topic distribution.",
    )
    args = parser.parse_args()

    G_dir, G_undir = create_graph(edge_list_path=args.edge_list_path)

    # Plot degree distribution
    plot_degree_dist(G_undir)

    # Plot top k degree users
    plot_top_k_degree(G_dir)

    # Calculate majority illusion
    majority_illusion = calculate_majority_illusion(
        G_undir, topic_dist_path=args.topic_dist_path
    )


if __name__ == "__main__":
    main()
