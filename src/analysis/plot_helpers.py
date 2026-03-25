import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# Set Roboto as the default font family for all plots
# plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.sans-serif'] = "Droid Sans"


def plot_user_to_post_sankey(
    users,
    filter_df,
    user_rank,
    ax,
    max_posts_per_user=20,
    platform_label="",
    user_x=0.2,
    is_leftmost=True,
):
    """
    Visualizes the flow of users to their posts using smooth Bezier curves in a specified subplot.

    Parameters:
    - users (list): List of users.
    - filter_df (DataFrame): DataFrame containing post information, including the authors and posts.
    - user_rank (dict): Dictionary mapping users to their rank.
    - ax (matplotlib.axes): The axis to plot on.
    - max_posts_per_user (int, optional): Max number of posts per user to display (default is 20).
    - platform_label (str, optional): Label for the platform to be displayed as the title.
    """

    # Sort users by PageRank from highest to lowest
    users = sorted(users, key=lambda u: user_rank[u])

    # Map each user to their posts
    user_to_posts = {
        user: filter_df[filter_df["author"] == user]["post"].tolist()[
            :max_posts_per_user
        ]
        for user in users
    }
    user_to_posts = {k: v for k, v in user_to_posts.items() if v}

    # Set uniform block height for each user from top (1st rank) to bottom (20th rank)
    num_users = len(users)
    user_block_height = 1 / num_users  # Each user gets equal height
    user_positions = np.linspace(1 - user_block_height, 0, num_users)  # Top to bottom

    # Calculate post percentiles for curve plotting
    all_posts = filter_df["post"].tolist()
    num_all_posts = len(all_posts)

    def get_post_percentile(post, all_posts):
        return 1 - (all_posts.index(post) / num_all_posts)

    posts_to_plot = [post for posts in user_to_posts.values() for post in posts]
    percentiles = [get_post_percentile(post, all_posts) for post in posts_to_plot]

    # Vertical line for posts with reduced spacing
    post_x = 0.50  # Reduced spacing for a compact layout
    ax.vlines(post_x, 0, 1, colors="gray", lw=2)

    # Vertical lines for each user with minimal spacing

    for i, user_pos in enumerate(user_positions):
        ax.vlines(user_x, user_pos, user_pos + user_block_height, colors="gray", lw=2)

    # Adding usernames, rank, and frst post text next to the user line
    user_centers = user_positions + user_block_height / 2
    for user_center, user in zip(user_centers, users):
        if is_leftmost:
            ax.text(
                0.07,
                user_center,
                f"{user_rank[user]}",
                va="center",
                ha="center",
                fontsize=14,
                color="black",
                fontweight="bold",
            )

        ax.text(
            user_x - 0.01,
            user_center,
            user,
            va="center",
            ha="right",
            fontsize=14,
            color="black",
            fontweight="bold",
        )

        # Display the first post on the right side of the username
        # first_post_text = user_to_posts[user][0] if user in user_to_posts else ""
        # ax.text(0.25, user_center, first_post_text, va="center", ha="left", fontsize=8, color="black")

    # Bezier curves for posts
    for user_idx, user_pos in enumerate(user_positions):
        user = users[user_idx]
        if user in user_to_posts:
            user_center = user_pos + user_block_height / 2
            post_percentiles_for_user = [
                percentiles[posts_to_plot.index(post)] for post in user_to_posts[user]
            ]

            for post_percentile in post_percentiles_for_user:
                post_center = post_percentile
                start = (user_x, user_center)
                end = (post_x, post_center)
                control1 = (user_x + 0.1, user_center)
                control2 = (post_x - 0.05, post_center)

                # Determine color based on percentile
                curve_color = get_color_for_percentile(post_center)
                path = bezier_curve(start, end, control1, control2)
                patch = PathPatch(
                    path, facecolor="none", edgecolor=curve_color, lw=2, alpha=0.7
                )
                ax.add_patch(patch)

    # Hide axes
    ax.set_axis_off()

    # Column labels
    if is_leftmost:
        ax.text(
            0.07,
            1.02,
            "PageRank",
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color="black",
        )
    ax.text(
        user_x - 0.05,
        1.02,
        "User",
        va="center",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color="black",
    )
    ax.text(
        0.50,
        1.02,
        "FastLexRank",
        va="center",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color="black",
    )

    # Set the title for the platform
    ax.set_title(
        platform_label,
        fontsize=20,
        fontweight="bold",
        y=0,
    )

    # Helper functions


def bezier_curve(start, end, control1, control2):
    """Generate a Bezier curve path."""
    verts = [start, control1, control2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def get_color_for_percentile(post_center):
    """Determine color based on the percentile of the post."""
    percentiles = [0.25, 0.5, 0.75, 1.0]
    if post_center <= percentiles[0]:
        return "#403990"
    elif post_center <= percentiles[1]:
        return "#80A6E2"
    elif post_center <= percentiles[2]:
        return "#fbdd85"
    else:
        return "#cf433e"


def add_lexrank_legend(fig):
    """
    Adds a legend to the figure indicating the color scheme for LexRank percentiles.

    Parameters:
    - fig (matplotlib.figure): The figure to add the legend to.
    """
    # Define the legend elements
    legend_elements = [
        mpatches.Patch(color="#403990", label="75-100%"),
        mpatches.Patch(color="#80A6E2", label="50-75%"),
        mpatches.Patch(color="#fbdd85", label="25-50%"),
        mpatches.Patch(color="#cf433e", label="0-25%"),
    ]

    # Add the legend to the figure
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        title="LexRank Percentile",
        ncol=4,
        fontsize=15,
        title_fontsize=16,
    )

    # Main function to plot the three platforms with separate subplots


def plot_sankey_for_three_platforms(
    users_list,
    filter_dfs,
    user_ranks,
    platform_labels,
    max_posts_per_user=20,
    dpi=500,
    save_path=None,
):
    """
    Plots three Sankey diagrams for different platforms in a single row with a LexRank legend.

    Parameters:
    - users_list (list of lists): List of user lists for each platform.
    - filter_dfs (list of DataFrames): List of DataFrames for each platform.
    - user_ranks (list of dicts): List of dictionaries containing ranks for each platform.
    - platform_labels (list of str): Labels for each platform.
    - max_posts_per_user (int, optional): Max number of posts per user to display (default is 20).
    - dpi (int, optional): Resolution of the plot in dots per inch (default is 300).
    """

    # Create a row of 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(22, 10), dpi=dpi)

    # Plot each platform's Sankey diagram in its respective subplot
    for i, ax in enumerate(axs):
        plot_user_to_post_sankey(
            users_list[i],
            filter_dfs[i],
            user_ranks[i],
            ax=ax,
            max_posts_per_user=max_posts_per_user,
            platform_label=platform_labels[i],
            is_leftmost=(i == 0),
            user_x=0.05 if platform_labels[i] != "Mastodon" else 0.22,
        )

    # Add a global legend for LexRank percentiles
    add_lexrank_legend(fig)

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for the legend at the top
    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_platform_partisanship_box_strip(
    platform_df, ax, platform_name, jitter_x=0.02, alpha=1
):
    """
    Plots a box plot overlaid with a strip plot for each platform's partisanship distribution on a specified axis,
    with dynamic sampling based on the network size.

    Parameters:
    - platform_df (pd.DataFrame): DataFrame containing 'community_id', 'size', and 'partisanship_scores' for each platform.
    - ax (matplotlib.axes): The axis to plot on.
    - platform_name (str): The title to be displayed for each platform subplot.
    - jitter_x (float): Amount of jitter to apply along the x-axis for the strip plot.
    - alpha (float): Transparency level for the strip plot points. Default is 0.6.
    """
    # Determine sample size based on network size

    # Expand platform data into a DataFrame for Seaborn compatibility
    plot_data = []
    for i, (community_id, scores, size) in enumerate(
        zip(
            platform_df["community_id"],
            platform_df["partisanship_scores"],
            platform_df["size"],
        )
    ):
        if size > 10000:
            sample_size = int(0.01 * size + 500)
        elif size > 1000:
            sample_size = int(0.1 * size + 200)
        elif size > 100:
            size = int(0.5 * size + 100)
        else:
            sample_size = None  # No sampling if the network size is small
        # Apply dynamic sampling if sample_size is determined
        if sample_size and size > sample_size:
            scores = np.random.choice(scores, sample_size, replace=False)

        for score in scores:
            # Apply jitter along the x-axis
            score_with_jitter = score + np.random.uniform(-jitter_x, jitter_x)

            # Assign a partisanship category based on the score
            if score < -0.5:
                partisanship_category = "Left"
            elif -0.5 <= score < -0.1:
                partisanship_category = "Lean Left"
            elif -0.1 <= score <= 0.1:
                partisanship_category = "Center"
            elif 0.1 < score < 0.5:
                partisanship_category = "Lean Right"
            else:
                partisanship_category = "Right"

            plot_data.append(
                {
                    "community_id": f"{community_id} (n={size})",
                    "partisanship_score": score_with_jitter,
                    "partisanship_category": partisanship_category,
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # Define colors for each category
    color_map = {
        "Left": "#7895C1",
        "Lean Left": "#A8CBDF",
        "Center": "#F5EBAE",
        "Lean Right": "#E3625D",
        "Right": "#992224",
    }

    # Box plot on the specified axis
    sns.boxplot(
        x="partisanship_score",
        y="community_id",
        data=plot_df,
        color="lightgray",
        showcaps=True,
        fliersize=0,
        width=0.6,
        orient="h",
        ax=ax,
    )

    # Strip plot on the specified axis with a smaller dot size
    sns.swarmplot(
        x="partisanship_score",
        y="community_id",
        data=plot_df,
        hue="partisanship_category",
        palette=color_map,
        dodge=False,
        size=3,  # Smaller dot size for swarm plot
        alpha=alpha,
        ax=ax,
    )

    # Set title for the platform
    ax.set_title(platform_name, fontsize=18, fontweight="bold")

    # Adjust aesthetics for the subplot
    ax.set_ylabel("Communities (Ranked by Size)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_xlabel("")

    # Remove individual legend to avoid duplicates
    ax.get_legend().remove()


def plot_all_platforms(platform_dfs, platform_names, dpi=500):
    """
    Plots three platform box plots in a single figure with shared x-axis label and legend.

    Parameters:
    - platform_dfs (list of pd.DataFrame): List of platform DataFrames for each plot.
    - platform_names (list of str): List of platform names to use as titles for each subplot.
    - dpi (int, optional): Resolution of the plot.
    """
    # Create a 3x1 grid for subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), dpi=dpi, sharex=True)

    # Plot each platform dataframe in its own subplot
    for i, (platform_df, platform_name) in enumerate(zip(platform_dfs, platform_names)):
        plot_platform_partisanship_box_strip(
            platform_df, ax=axs[i], platform_name=platform_name
        )

    # Set a shared x-label and title
    fig.text(
        0.5,
        0.04,
        "Political Ideology Score (Left: -1 to Right: 1)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    # fig.suptitle("Platform Partisanship Distributions", fontsize=16, fontweight="bold")

    # Create a single legend outside the plot
    handles, labels = axs[1].get_legend_handles_labels()
    desired_order = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
    sorted_handles_labels = sorted(
        zip(handles, labels), key=lambda x: desired_order.index(x[1])
    )
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # Create a single legend outside the plot in the specified order
    fig.legend(
        sorted_handles,
        sorted_labels,
        title="Political Ideology",
        bbox_to_anchor=(0.90, 0.5),
        loc="center",
        fontsize=14,
        title_fontsize=14,
    )
    # Adjust layout for better spacing
    plt.tight_layout(
        rect=[0, 0.05, 0.8, 0.95]
    )  # Adjust layout to leave space for the legend on the right
    plt.savefig("community.png")
    plt.show()


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


def plot_swarm_plot_by_leaning_category(df):
    """
    Swarm plot showing distribution of influence leaning grouped by initial leaning categories.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Initial Leaning' and 'Influence Leaning' columns.
    """
    # Add the leaning category column
    df["Leaning Category"] = df["Initial Leaning"].apply(categorize_leaning)

    # Set the desired category order
    category_order = ["Extreme Left", "Left", "Center", "Right", "Extreme Right"]

    plt.figure(figsize=(12, 8), dpi=300)
    sns.swarmplot(
        x="Leaning Category",
        y="Influence Leaning",
        data=df,
        order=category_order,
        size=5,  # Adjust point size for better visibility
    )

    # Plot title and labels
    plt.title("Distribution of Influence Set Leaning by Initial Leaning Category")
    plt.xlabel("Initial Leaning Category")
    plt.ylabel("Average Leaning of Influence Set")
    plt.show()
