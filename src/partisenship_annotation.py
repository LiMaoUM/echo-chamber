import re

import pandas as pd
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
# Regular expression to match the value after "Trump":
pattern = r'"political_leaning":\s*"(.*?)"'
section = 1
target = "partisan"
df = pd.read_parquet(f"../data/annotation/df_subset_{section}.parquet")


# Define a function to create the prompt, considering the context of the reply
def create_prompt(post, reply_to_post=None):
    if reply_to_post:
        prompt = f"""
        Analyze the following post for the user's political leaning, based on these detailed guidelines:

        1. **Left**:
           - The post praises or supports Joe Biden, Democrats, liberal values, or progressive policies.
           - The post criticizes or opposes Donald Trump, conservatives, or right-leaning positions.
        
        2. **Lean Left**:
           - The post expresses moderate or qualified support for left-leaning positions (e.g., support for Biden or Democratic policies but with some reservations).
           - The post mildly criticizes right-leaning positions or Trump but does not express strong or radical views.

        3. **Center**:
           - The post takes a neutral or balanced stance without strong support or opposition for either side.
           - The post avoids taking a clear left or right stance and may present both sides of an issue equally.

        4. **Lean Right**:
           - The post expresses moderate or qualified support for right-leaning positions (e.g., support for Trump or conservative values but with some reservations).
           - The post mildly criticizes left-leaning positions or Biden without expressing extreme or radical views.

        5. **Right**:
           - The post praises or supports Donald Trump, conservatives, right-wing ideologies, or conservative policies.
           - The post criticizes or opposes Joe Biden, Democrats, or liberal values.

        Consider that this post is a response to the following context:
        
        Replying to: "{reply_to_post}"
        
        Post: "{post}"
        
        Please classify the political leaning using one of the following options:
        - Left
        - Lean Left
        - Center
        - Lean Right
        - Right

        ONLY provide the output in the following JSON format without any additional text:
        {{
            "Political_Leaning": "Left/Lean Left/Center/Lean Right/Right"
        }}
        """
    else:
        # Same prompt logic without reply context
        prompt = f"""
        Analyze the following post for the user's political leaning, based on these detailed guidelines:

        1. **Left**:
           - The post praises or supports Joe Biden, Democrats, liberal values, or progressive policies.
           - The post criticizes or opposes Donald Trump, conservatives, or right-leaning positions.
        
        2. **Lean Left**:
           - The post expresses moderate or qualified support for left-leaning positions (e.g., support for Biden or Democratic policies but with some reservations).
           - The post mildly criticizes right-leaning positions or Trump but does not express strong or radical views.

        3. **Center**:
           - The post takes a neutral or balanced stance without strong support or opposition for either side.
           - The post avoids taking a clear left or right stance and may present both sides of an issue equally.

        4. **Lean Right**:
           - The post expresses moderate or qualified support for right-leaning positions (e.g., support for Trump or conservative values but with some reservations).
           - The post mildly criticizes left-leaning positions or Biden without expressing extreme or radical views.

        5. **Right**:
           - The post praises or supports Donald Trump, conservatives, right-wing ideologies, or conservative policies.
           - The post criticizes or opposes Joe Biden, Democrats, or liberal values.

        Post: "{post}"
        
        Please classify the political leaning using one of the following options:
        - Left
        - Lean Left
        - Center
        - Lean Right
        - Right

        ONLY provide the output in the following JSON format without any additional text:
        {{
            "Political_Leaning": "Left/Lean Left/Center/Lean Right/Right"
        }}
        """
    return prompt


# Function to run batch inference
def annotate_posts_in_batch(posts, replies, llm, sampling_params):
    # Generate the prompt for each post, including reply context if available
    prompts = [
        create_prompt(post, reply) if reply else create_prompt(post)
        for post, reply in zip(posts, replies)
    ]

    # Run the inference in batch
    results = llm.generate(prompts, sampling_params)

    # Parse the batch results
    parsed_results = []
    for result in results:
        match = re.search(pattern, result.outputs[0].text.lower())
        if match:
            value = match.group(1)  # Extract the value (e.g., "Neutral")
            parsed_results.append(value)
        else:
            parsed_results.append("error")

    return parsed_results


# Function to process large dataset in batches and save the annotations to the DataFrame
# Function to process large dataset in batches and save the annotations to CSV incrementally
def process_large_dataset(posts, reply_to_posts, batch_size, df, output_file):
    # Track how many posts are processed
    total_processed = 0
    df["stance"] = None

    # Open the CSV file for the first time to write header
    with open(output_file, "w") as f:
        # Write header
        f.write("Trump Stance\n")

    # Process posts in batches
    for i in tqdm(range(0, len(posts), batch_size)):
        batch_posts = posts[i : i + batch_size]
        batch_replies = reply_to_posts[i : i + batch_size]

        # Annotate stance for the current batch
        batch_results = annotate_posts_in_batch(
            batch_posts, batch_replies, llm, sampling_params
        )

        # Append results to the CSV file after each batch
        with open(output_file, "a") as f:  # Append mode ('a') to add new results
            for post, reply, result in zip(batch_posts, batch_replies, batch_results):
                f.write(f'"{result}"\n')

        # Update the DataFrame with the annotations
        df.loc[i : i + batch_size - 1, "stance"] = batch_results

        total_processed += len(batch_posts)
        print(f"Processed {total_processed} out of {len(posts)} posts")

    return df


# Example usage
# Assuming you have a DataFrame `df` with columns 'post' and 'reply_to_post'
posts = list(df["post"])
reply_to_posts = list(df["reply_to_post"])

# Define batch size (you can experiment with this value for optimal performance)
batch_size = 32

# Output file for saving results incrementally
output_file_csv = (
    f"../data/annotation/partial_stance_analysis_results_{target}{section}.csv"
)

# Process the dataset in batches and save results incrementally

df_with_annotations = process_large_dataset(
    posts, reply_to_posts, batch_size, df, output_file_csv
)

# Optionally save the DataFrame as Parquet for efficient storage as well
output_file_parquet = (
    f"../data/annotation/annotated_stance_analysis_{target}{section}.parquet"
)
df_with_annotations.to_parquet(output_file_parquet, index=False)

print(
    f"Annotated DataFrame saved to {output_file_parquet} and partial results saved to {output_file_csv}"
)
