# ELCRec

NeurIPS 2024

### Requirements

- Python >= 3.7
- Pytorch >= 1.2.0
- tqdm == 4.26.0
- faiss-gpu==1.7.1



### Quick Start

For evaluation:
```
cd ./src
bash ./scripts/run_sports.sh
```

```
ELCRec-Sports_and_Outdoors-1
{'Epoch': 0, 'HIT@5': '0.0286', 'NDCG@5': '0.0185', 'HIT@20': '0.0648', 'NDCG@20': '0.0286'}
```



```
bash ./scripts/run_beauty.sh
```

```
ELCRec-Beauty-1
{'Epoch': 0, 'HIT@5': '0.0529', 'NDCG@5': '0.0355', 'HIT@20': '0.1079', 'NDCG@20': '0.0509'}
```



```
bash ./scripts/run_toys.sh
```

```
ELCRec-Toys_and_Games-1
{'Epoch': 0, 'HIT@5': '0.0585', 'NDCG@5': '0.0403', 'HIT@20': '0.1138', 'NDCG@20': '0.0560'}
```



```
bash ./scripts/run_yelp.sh
```

```
ELCRec-Yelp-1
{'Epoch': 0, 'HIT@5': '0.0236', 'NDCG@5': '0.0150', 'HIT@20': '0.0653', 'NDCG@20': '0.0266'}
```

