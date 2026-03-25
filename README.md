# Echo Chamber

Code for the ICWSM 2026 paper, *Who Talks to Whom: Quantifying Echo Chamber Effects in Emerging Social Media Platforms*.

DOI: to be added after publication.

Author: Mao Li  
Email: maolee@umich.edu

## Overview

This repository contains code for analyzing interaction structure and political alignment in emerging social media platforms. Based on the current codebase, the project focuses on:

- constructing author reply networks from platform-specific data
- computing basic network statistics and centrality
- detecting communities in interaction networks
- estimating post- and user-level political leaning
- visualizing influence and representativeness patterns across platforms

The analysis currently references multiple platforms, including Mastodon, Bluesky, Truth Social, and Reddit.

## Repository Structure

```text
.
├── data/
└── src/
    ├── partisenship_annotation.py
    └── analysis/
        ├── network_basic.ipynb
        ├── network_helpers.py
        └── plot_helpers.py
```

## What the Code Does

### `src/partisenship_annotation.py`

Runs batched LLM-based annotation of post political leaning using `vllm` and a Llama 3.1 instruct model. The script reads a parquet subset, generates prompts with optional reply context, and writes both CSV and parquet outputs for downstream analysis.

### `src/analysis/network_helpers.py`

Provides helper functions for:

- loading platform-specific identifier mappings
- constructing directed reply networks
- computing basic network statistics
- ranking users with PageRank
- estimating user partisanship from post labels
- detecting communities with Leiden
- preparing community-level partisanship summaries

### `src/analysis/plot_helpers.py`

Contains plotting utilities for visualizing user-to-post relationships and influence/representativeness patterns across platforms.

### `src/analysis/network_basic.ipynb`

Notebook for exploratory and comparative network analysis across platforms, including basic structural statistics, community-level stance analysis, and influence-representativeness alignment.

## Data Availability

The data used in this project are not distributed in this repository. Researchers who are interested in access to the underlying data should contact the author directly.

## Environment Notes

The current code imports the following major Python packages:

- `pandas`
- `numpy`
- `networkx`
- `cdlib`
- `sentence-transformers`
- `matplotlib`
- `seaborn`
- `tqdm`
- `vllm`

Depending on your environment, additional setup may be required for model inference and notebook execution.

## Citation

If you use this repository, please cite the ICWSM 2026 paper:

Mao Li. *Who Talks to Whom: Quantifying Echo Chamber Effects in Emerging Social Media Platforms*. ICWSM 2026. DOI forthcoming.
