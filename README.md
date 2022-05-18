# Temporality- and Frequency-aware Graph Contrastive Learning for temporal networks


## Requirement
+ Python (>=3.8)
+ PyTorch (>=1.9.0)
+ NumPy (>=1.22.3)
+ Pandas (>=1.4.1)
+ Scikit-Learn (>=1.0.2)
+ Scipy (>=1.9.0)
+ Networkx (>=2.6.3)
+ DGL (>=0.8.1)


## Dataset
DBLP [Michael Ley. 2009. DBLP - Some Lessons Learned.Proc. VLDB Endow.2, 2 (2009),1493–1500. ]

sx-mathoverflow: http://snap.stanford.edu/data/sx-mathoverflow.html

CollegeMsg: http://snap.stanford.edu/data/CollegeMsg.html

ia-retweet-pol: https://networkrepository.com/ia-retweet-pol.php

Facebook: https://networkrepository.com/fb-messages.php

Email: http://snap.stanford.edu/data/email-Eu-core-temporal.html

|  Dataset   | Nodes  | Edges | Timespan |
|  -------------------  | ----  | ----- | ---------- |
| DBLP  | 26,584 | 78K | 5 years |
| sx-mathoverflow  | 24,184 | 503K | 6 years |
| CollegeMsg | 1,854 | 59K | 191 days |
| ia-retweet-pol | 17,336 | 59K | 49 days |
| Facebook | 1,862 | 61K | 215 days |
| Email | 979 | 332K | 2 years |

## Usage
Here we provide the implementation of TF-GCL.

+ To train and evaluate, just run:
```python
python run_retweet.py
```

or

```python
python main_dynamicCL_ver1.py --h_feats_list 256 --data=ReTweet --epochs=2000 --emb_size=256 --lr=5e-3 --cl_loss_weight=1.0 --cl_batch_size=256 --neg_class_weight=0.4 --temporal_weight_loss=true --cl_loss=true --decay_by=day --ts_decay_rate=0.01 --concat=false --temporal_cl_loss=true --neighbor_as_pos=true --symmetric_cl_loss=true --eval_epochs=1 --fea_drop_percent=0.6 --edge_pert_percent=0.6 --ts_mask_percent=0.6
```
`h_feats_list` construct the GCNs encoder, for example `--h_feats_list 256 128` construct a two layer GCNs encoder, which have 256 and 128 hidden dimension.
`cl_batch_size` is the balance weight of TF-GCL
`fea_drop_percent` is the drop rate of augmentation feature masking
`edge_pert_percent` is the edge ratio of augmentation edge perturbation
`ts_mask_percent` is the edge ratio of augmentation timestamp masking


## Input
### Input format
G.edges

where G.edges is the edgelist file with timestamps. column 1 is the start node, column 2 is the destination node, and column 3 is the timestamp of the edge.
### edges file
```
3 9 0
1 3 13
2 4  50
...
```


