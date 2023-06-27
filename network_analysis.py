import numpy as np
import argparse
import joypy
from typing import Tuple, List
import pathlib as pl
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

TOPIC_LABELS = [
    "election issues / problems",
    "deleted comments",
    "energy and climate",
    "coalition formation",
    "right-wing democratic critique",
    "voting strategies / voting system",
    "discussing extremism",
    "anti-liberalism (anti-lgbtqi+, anti-immigration)",
    "anti-establishment",
    "bashing the moderates",
    "undefined",
    "nato",
    "malmo politics",
    "right-wing critique",
    "ukraine invasion",
    "sex work",
    "military threat from russia",
    "liberal party politics - values",
    "drug policy",
    "liberal party politics - taxes and money",
]


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

    fname = "figs/degree_distribution.png"
    print(f"Saving figure to {fname}")
    plt.savefig(fname)


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

    fname = "figs/top_degree_users.png"
    print(f"Saving figure to {fname}")

    plt.savefig(fname)


def calculate_majority_illusion(
    G: nx.Graph, topic_dist_path: str = "data/user-topic_distribution.csv"
) -> Tuple[np.ndarray, List[str]]:
    # Get largest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    # Get topic Distribution
    df = pd.read_csv(topic_dist_path, index_col=0)

    # Get users
    users = df.index.tolist()

    # Store node names
    node_names = []

    percieved_topic_dist = []
    for node in G.nodes:
        G_ego = nx.ego_graph(G, node)

        # Get topic distribution of neighbors
        # check that neighbors are in users and not the node itself
        tmp_dists = np.array(
            [
                df.loc[neighbor, :].to_numpy().flatten()
                for neighbor in G_ego.nodes
                if neighbor in users and neighbor != node
            ]
        )

        # Make sure that there are neighbors
        if len(tmp_dists) > 0:
            percieved_topic_dist.append(np.mean(tmp_dists, axis=0))
            node_names.append(node)

    # Calculate majority illusion
    percieved_topic_dist = np.vstack(percieved_topic_dist)

    global_topic_dist = df.to_numpy().mean(axis=0)

    return (percieved_topic_dist - global_topic_dist), node_names


def plot_majority_ridge_plot(majority_illusion: np.ndarray) -> None:
    plot_df = pd.DataFrame(majority_illusion, columns=TOPIC_LABELS).melt()
    fig, _ = joypy.joyplot(
        plot_df,
        by="variable",
        column="value",
        range_style="own",
        grid="y",
        linewidth=0.2,
        legend=False,
        figsize=(6, 3),
        fade=True,
        # grid=None, linewidth=0.1, legend=False, fade=True, figsize=(5,10),
        title="Perception deviation distribution",
        ylabelsize=4,
        kind="normalized_counts",
        bins=60,
    )

    fname = "figs/majority_illusion_ridge_plot.png"
    print(f"Saving figure to {fname}")

    plt.savefig(fname, dpi=700)


def plot_topic_degree_corr(
    G: nx.Graph, topic_dist_path: str = "data/user-topic_distribution.csv"
) -> None:
    topic_df = pd.read_csv(topic_dist_path, index_col=0)
    topic_deg_df = topic_df.merge(
        pd.DataFrame(tuple(G.degree), columns=["user", "degree"]), on="user"
    )

    fig, axs = plt.subplots(ncols=5, nrows=4, dpi=500)
    axs = axs.flatten()

    for i, (topic, topic_label) in enumerate(
        zip([f"topic_{i+1}" for i in range(20)], TOPIC_LABELS)
    ):
        axs[i].plot(
            topic_deg_df["degree"], topic_deg_df[topic], "bo", alpha=0.4, markersize=2
        )
        axs[i].set_title(topic_label, size=5)

    fig.text(0.5, -0.01, "Degree", ha="center")
    fig.text(-0.01, 0.5, "Topic proportion", va="center", rotation="vertical")
    fig.suptitle("Degree - topic correlation")
    fig.tight_layout()
    plt.savefig("figs/degree_topic_correlation.png", dpi=700, bbox_inches="tight")


def plot_illusion_network(
    G: nx.Graph, node_names: List[str], majority_illusion: np.ndarray, topic: int = 15
) -> None:
    fig, ax = plt.subplots(dpi=500, figsize=(10, 10))

    # Subset G to only include nodes with majority illusion
    G = G.subgraph(node_names)

    G = nx.Graph(G)

    # Remopve self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    degrees = [G.degree[node] for node in G.nodes]

    colors = majority_illusion[:, topic]

    pos = nx.spring_layout(G)

    fig = nx.draw_networkx_nodes(
        G,
        pos=pos,
        alpha=0.6,  # node_size=2,
        edgecolors="black",
        linewidths=0.2,
        node_size=degrees,
        ax=ax,
        node_color=colors,
        cmap=plt.cm.coolwarm,
    )

    nx.draw_networkx_edges(
        G,
        pos=pos,
        width=0.1,
        alpha=0.6,
        arrowsize=1,
        ax=ax,
    )

    plt.colorbar(fig, shrink=0.4)
    ax.axis("off")

    fname = f"figs/majority_illusion_network_topic_{topic}.png"
    print(f"Saving {fname}")

    plt.savefig(
        fname,
        dpi=700,
        bbox_inches="tight",
    )


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
    majority_illusion, valid_nodes = calculate_majority_illusion(
        G_undir, topic_dist_path=args.topic_dist_path
    )

    # Plot majority illusion
    plot_majority_ridge_plot(majority_illusion)

    # Plot topic degree correlation
    plot_topic_degree_corr(G_undir, topic_dist_path=args.topic_dist_path)

    # Plot illusion network
    plot_illusion_network(G_undir, valid_nodes, majority_illusion, topic=15)


if __name__ == "__main__":
    main()
