# Temporality- and Frequency-aware Graph Contrastive Learning for temporal networks


## Dependencies
+ Python (>=3.8)
+ PyTorch (>=1.9.0)
+ NumPy (>=1.22.3)
+ Pandas (>=1.4.1)
+ Scikit-Learn (>=1.0.2)
+ Scipy (>=1.9.0)
+ Networkx (>=2.6.3)
+ DGL (>=0.8.1)

To install all dependencies:
```
pip install -r requirements.txt
```

## Usage
Here we provide the implementation of MERIT along with Cora and Citeseer dataset.

+ To train and evaluate:
```python
python run_retweet.py
```

or

`python main_dynamicCL_ver1.py --h_feats_list 256 --data=ReTweet --epochs=2000 --emb_size=256 --lr=5e-3 --cl_loss_weight=1.0 --cl_batch_size=256 --neg_class_weight=0.4 --temporal_weight_loss=true --cl_loss=true --decay_by=day --ts_decay_rate=0.01 --concat=false --temporal_cl_loss=true --neighbor_as_pos=true --symmetric_cl_loss=true --eval_epochs=1 --fea_drop_percent=0.6 --edge_pert_percent=0.6 --ts_mask_percent=0.6`



