import json
import os
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms, evaluation
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

model = SentenceTransformer("all-MiniLM-L12-v2")

MASTODON_FOLDER = "../../data/mastodon/"
BSKY_FOLDER = "../../data/bsky/"
TRUTHSOCIAL_FOLDER = "../../data/truthsocial/"
REDDIT_FOLDER = "../../data/reddit/"
AUTHOR_NODE_FILE = "author_nodes.txt"
AUTHOR_TO_POST_FILE = "author_to_post.json"
REPLY_EDGES_FILE = "reply_edges.txt"
USERNAME_TO_ID_FILE = "username_to_id.txt"
POST_TO_LABEL_FILE = "post_to_label.json"


def load_username_to_id(folder) -> dict:
    username_to_id = {}

    # Open and read the file line by line
    with open(folder + USERNAME_TO_ID_FILE, mode="r") as file:
        next(file)  # Skip the header line
        for line in file:
            # Split each line by spaces (or tabs if needed) and unpack into username and account_id
            try:
                username, account_id = (
                    line.strip().split()
                )  # Use .split('\t') if tab-separated
                username_to_id[username] = account_id  # Store in dictionary
            except:
                continue
    return username_to_id


# construct the author nodes and reply edges
def construct_network(folder) -> nx.DiGraph:
    G = nx.read_edgelist(folder + REPLY_EDGES_FILE, create_using=nx.DiGraph)
    author_nodes = []
    with open(folder + AUTHOR_NODE_FILE, "r") as f:
        author_nodes = f.read().splitlines()
    G.add_nodes_from(author_nodes)
    return G


def compute_network_statistics(G: nx.DiGraph) -> dict:
    """
    Compute basic network statistics including:
    - Number of nodes
    - Number of edges
    - Average degree
    - Average clustering coefficient
    - Proportion of isolated nodes
    - Diameter of the network (if connected)

    Parameters:
    - network: A NetworkX graph object

    Returns:
    A dictionary containing the computed statistics.
    """

    # Number of nodes
    num_nodes = G.number_of_nodes()

    # Number of edges
    num_edges = G.number_of_edges()

    # Average degree (for directed graph: in-degree and out-degree separately)
    in_degrees = dict(G.in_degree())  # Dictionary of in-degrees per node
    out_degrees = dict(G.out_degree())  # Dictionary of out-degrees per node

    # Average degree: mean of in-degree and out-degree combined
    avg_in_degree = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out_degree = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_degree = (avg_in_degree + avg_out_degree) / 2

    # Average clustering coefficient (only for undirected graphs, so convert)
    # if len(G) > 1:  # We can't compute clustering coefficient if there's only one node
    #     undirected_G = G.to_undirected()
    #     avg_clustering = nx.average_clustering(undirected_G)
    # else:
    #     avg_clustering = 0

    # Proportion of isolated nodes (nodes with degree 0)
    isolated_nodes = list(nx.isolates(G))  # Nodes with no connections
    num_isolated_nodes = len(isolated_nodes)
    proportion_isolated = num_isolated_nodes / num_nodes if num_nodes > 0 else 0

    print("Network Statistics:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average in-degree: {avg_in_degree:.2f}")
    print(f"Average out-degree: {avg_out_degree:.2f}")
    print(f"Average degree: {avg_degree:.2f}")
    # print(f"Average clustering coefficient: {stats['avg_clustering']:.4f}")
    print(f"Proportion of isolated nodes: {proportion_isolated:.4f}")

    # Return the computed statistics as a dictionary
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_in_degree": avg_in_degree,
        "avg_out_degree": avg_out_degree,
        "avg_degree": avg_degree,
        #    "avg_clustering": avg_clustering,
        "proportion_isolated_nodes": proportion_isolated,
    }

    return stats


def fastLexRank(df: pd.DataFrame) -> pd.DataFrame:
    posts = df["post"]
    embeddings = model.encode(posts, show_progress_bar=True)
    # sum in column
    z = embeddings.sum(axis=0)
    # normalize the sum
    z = z / np.sqrt((z**2).sum(axis=0))
    ap = np.dot(embeddings, z)
    # normalize the scores by its sum
    df["ap"] = ap
    df.sort_values(by="ap")
    return df


def pagerank_20top_users(
    G: nx.DiGraph, network: str, id_to_username: dict
) -> list[str]:
    pagerank = nx.pagerank(G, max_iter=10000)
    # sort the dictionary by value
    sorted_pr = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))

    if network == "bsky":
        top_users = [k for k in list(sorted_pr.keys())[:20]]
        top_users_score = dict(list(sorted_pr.items())[:20])
        top_users_score = dict(
            [(s, i + 1) for i, s in enumerate(list(top_users_score.keys()))]
        )
        return top_users, top_users_score
    # elif network == "reddit":
    #     username_to_id = reddit_username_to_id
    top_users = [id_to_username[k] for k in list(sorted_pr.keys())[:20]]
    top_users_score = dict(list(sorted_pr.items())[:20])
    top_users_score = dict(
        [(id_to_username[s], i + 1) for i, s in enumerate(list(top_users_score.keys()))]
    )
    return top_users, top_users_score


def assign_partisan_labels(post_to_author, post_labels):
    """
    Assigns a partisan label to each user based on the average of their posts' partisanship labels.

    Parameters:
    - post_to_author (dict): Dictionary mapping posts to their authors.
    - post_labels (dict): Dictionary mapping posts to their partisan labels (left, lean left, center, lean right, right).

    Returns:
    - user_partisanship (dict): Dictionary mapping users to their average partisan score.
    """
    label_to_score = {
        "left": -1,
        "lean left": -0.5,
        "center": 0,
        "lean right": 0.5,
        "right": 1,
        "error": 0,
    }

    # Initialize a dictionary to store each user's scores
    user_scores = defaultdict(list)

    # Map each post's label to its author and aggregate scores
    for post, author in post_to_author.items():
        if post in post_labels:
            score = label_to_score[post_labels[post]]
            user_scores[author].append(score)

    # Calculate the average score per user
    user_partisanship = {user: np.mean(scores) for user, scores in user_scores.items()}
    return user_partisanship


def detect_leiden_communities(G):
    """
    Apply Leiden community detection on a graph.

    Parameters:
    - G (nx.Graph): The input graph.

    Returns:
    - communities (dict): Dictionary mapping each node to its community label.
    """
    # Detect communities using Leiden algorithm
    leiden_communities = algorithms.leiden(G)
    communities = {}
    for idx, community in enumerate(leiden_communities.communities):
        for node in community:
            communities[node] = idx
    # Evaluate modularity
    mod_result = evaluation.newman_girvan_modularity(G, leiden_communities)
    modularity = mod_result.score

    return communities, modularity


def calculate_partisanship_distribution(user_partisanship, communities, top_n=10):
    """
    Calculates the distribution of partisanship scores for the largest communities.

    Parameters:
    - user_partisanship (dict): Dictionary mapping users to their partisanship scores.
    - communities (dict): Dictionary mapping users to their community labels.
    - top_n (int): Number of largest communities to consider.

    Returns:
    - community_df (pd.DataFrame): DataFrame with columns 'community_id', 'size', and 'partisanship_scores'.
    """
    # Aggregate users by communities
    community_users = defaultdict(list)
    for user, community in communities.items():
        if user in user_partisanship:
            community_users[community].append(user_partisanship[user])

    # Calculate community sizes and retain only the largest top_n communities
    community_sizes = sorted(
        [(community, len(users)) for community, users in community_users.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    largest_communities = {community: size for community, size in community_sizes}

    # Prepare data for each of the top_n largest communities
    community_data = [
        {
            "community_id": community,
            "size": largest_communities[community],
            "partisanship_scores": community_users[community],
        }
        for community in largest_communities
    ]

    # Convert to DataFrame
    community_df = pd.DataFrame(community_data)

    return community_df


def main_network_analysis(folder):
    # Step 1: Load data with progress indicator
    print("Loading data...")
    with tqdm(total=100, desc="Loading Data") as pbar:
        G = construct_network(folder)
        pbar.update(50)

        with open(folder + AUTHOR_TO_POST_FILE, "r") as f:
            post_to_author = json.load(f)
        pbar.update(25)

        with open(folder + POST_TO_LABEL_FILE, "r") as f:
            post_to_label = json.load(f)
        pbar.update(25)

    # Step 2: Assign partisan labels to users
    print("Assigning partisan labels to users...")
    user_partisanship = assign_partisan_labels(post_to_author, post_to_label)
    tqdm.write("Completed assigning partisan labels.")

    # Step 3: Apply Leiden community detection
    print("Applying Leiden community detection...")
    communities, val_score = detect_leiden_communities(G)
    print(f"Modularity score: {val_score:.4f}") # Print modularity score
    tqdm.write("Leiden community detection complete.")

    # Step 4: Calculate mean partisanship for the largest communities
    print("Calculating mean partisanship for largest communities...")
    largest_communities_partisanship = calculate_partisanship_distribution(
        user_partisanship, communities
    )
    tqdm.write("Mean partisanship calculation complete.")

    print("Analysis complete.")
    return largest_communities_partisanship


def calculate_node_partisanship(folder_path):
    """
    Calculate the average partisanship score for each author (node) by reading data from files in the specified folder.

    Parameters:
    - folder_path (str): Path to the folder containing 'post_to_author.json' and 'post_labels.json'.

    Returns:
    - node_partisanship (dict): Dictionary mapping each author (node) to their average partisanship score.
    """
    # Define label to score mapping
    label_to_score = {
        "left": -1,
        "lean left": -0.5,
        "center": 0,
        "lean right": 0.5,
        "right": 1,
    }

    # Define file paths within the folder
    post_to_author_path = os.path.join(folder_path, "author_to_post.json")
    post_labels_path = os.path.join(folder_path, "post_to_label.json")

    # Load data from JSON files
    with open(post_to_author_path, "r") as file:
        post_to_author = json.load(file)

    with open(post_labels_path, "r") as file:
        post_labels = json.load(file)

    # Initialize a dictionary to store each author's scores
    author_scores = defaultdict(list)

    # Map each post's label to its author and aggregate scores
    for post, author in post_to_author.items():
        if post in post_labels:
            score = label_to_score.get(
                post_labels[post], 0
            )  # Default to 0 if label is missing
            author_scores[author].append(score)

    # Calculate the average score per author (node)
    node_partisanship = {
        author: np.mean(scores) for author, scores in author_scores.items()
    }

    return node_partisanship


def calculate_beta(G, nu=0.2, target_R0=3.0):
    """
    Calculate an appropriate transmission probability (beta) for a given network G based on its average degree.

    Parameters:
    - G (networkx.Graph): The interaction network.
    - nu (float): The recovery rate for the SIR model.
    - target_R0 (float): Desired basic reproduction number (average number of infections caused by one infected node).

    Returns:
    - beta (float): Calculated transmission probability.
    """
    # Calculate the average degree of the network
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # Calculate beta based on the target R0
    beta = target_R0 * nu / avg_degree
    return beta


def run_sir_model(G, node_partisanship, nu=0.2, target_R0=3.0, iterations=1000):
    """
    Run the SIR model on the network G to simulate information spreading with dynamic beta calculation.

    Parameters:
    - G (networkx.Graph): The interaction network.
    - node_partisanship (dict): Dictionary mapping each node to its leaning score.
    - nu (float): Recovery rate.
    - target_R0 (float): Desired reproduction number.
    - iterations (int): Number of simulations to run with different initial nodes.

    Returns:
    - influence_leanings (pd.DataFrame): DataFrame containing initial leanings and the average leanings of influence sets.
    """
    # Calculate beta dynamically based on the network's average degree
    beta = calculate_beta(G, nu, target_R0)
    print(f"Calculated beta: {beta:.4f}")

    results = []

    # Run the model for each initial infected node
    for initial_node in random.sample(list(G.nodes()), iterations):
        # Initialize states
        states = {node: "S" for node in G.nodes()}
        states[initial_node] = "I"  # Start with initial_node as infected

        infected = [initial_node]
        influence_set = set(infected)

        # Run the SIR process until no more nodes are infectious
        while infected:
            new_infected = []
            for node in infected:
                # Spread to neighbors based on transmission probability beta
                for neighbor in G.neighbors(node):
                    if states[neighbor] == "S" and random.random() < beta:
                        states[neighbor] = "I"
                        new_infected.append(neighbor)
                        influence_set.add(neighbor)

                # Recover the node based on recovery probability nu
                if random.random() < nu:
                    states[node] = "R"

            infected = new_infected  # Update the list of currently infected nodes

        # Calculate average leaning of influence set
        influence_leaning = np.mean(
            [node_partisanship.get(node, 0) for node in influence_set]
        )
        initial_leaning = node_partisanship.get(initial_node, 0)

        results.append(
            {"Initial Leaning": initial_leaning, "Influence Leaning": influence_leaning}
        )

    return pd.DataFrame(results)
    return pd.DataFrame(results)


def categorize_leaning(leaning):
    """Categorize leanings into groups."""
    if leaning < -0.5:
        return "Extreme Left"
    elif -0.5 <= leaning < -0.25:
        return "Left"
    elif -0.25 <= leaning <= 0.25:
        return "Center"
    elif 0.25 < leaning <= 0.5:
        return "Right"
    else:
        return "Extreme Right"
