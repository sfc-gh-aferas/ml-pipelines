# Time-of-day email communications
This project aims to optimize the time-of-day at which users receive emails from January. It does so via a clustered-bandit approach. That is, in the training phase, a large set of debtors are clustered based on various debt-, debtor-, and client-related features. Then, in the inference phase, new debtors are first assigned to the nearest cluster based on their features, after which a multi-armed bandit (using UCB) is applied to each cluster separately. This folder contains the following files.

## Training
This file builds the clusters of debtors using K-means. It stores the K-means object in Snowflake's model registry and stores the formed clusters (alongside some metadata) in a Snowflake table.

## Inference
This file fetches new debtors that have not been assigned to a cluster yet, uses the stored K-means object to assign each debtor to its nearest cluster, and stores the assignments to a Snowflake table (alongside some metadata). 

## Clustered MAB
This file contains the logic for applying multi-armed bandits to the clustered debtors. More specifically, it applies the upper confidence bound (UCB) approach on email open rates to figure out the optimal time-of-day to send emails at for each cluster separately. All of this is controlled via a Snowflake SQL query.

## Utils
This file contains helper functions used for the time-of-day clustered bandits. Most notably, it contains functions to fetch, transform, and clean the data that is used for the clustering and cluster assignments. 

## Config
This file contains the run configuration for the time-of-day clustered bandits model. 

## Project Requirements
Simple file with packages required to run the code in this folder.