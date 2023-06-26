# SICCS-majority_illusion-project

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


