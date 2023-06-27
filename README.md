# SICCS-majority_illusion-project

This repository contains the code to replicate a project by Jeongho Choi,
Diletta Goglia, Christina Cano, Emelie Karlsson, Matias Piqueras, Sophie Mainz.
This was part of the course work of SICCS summer school in Norrköping 2023.

## Instructions

To replicate, you can run the following steps:

1. Start by collecting the relevant threads using the bulk-downloader-for-reddit
([`bdfr`](https://github.com/aliparlakci/bulk-downloader-for-reddit)).

``` bash
bdfr archive data/ --subreddit svenskpolitik --search "valet"
```

2. Next, create the edge list of user replies by running

```
python create_network.py --output data/edgelist.txt  
```

3. Perform the network analysis part by running

```
python network_analysis.py
```

All the generated figure will be stored under `figs/`

4. To parse and process the text data of each thread, run:

```
python parse_text.py
```


