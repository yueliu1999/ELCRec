<div align="center">
<h2><a href="https://arxiv.org/pdf/2410.02832">End-to-end Learnable Clustering for Intent Learning in Recommendation</a></h2>

[Yue Liu](https://yueliu1999.github.io/), Shihao Zhu, [Jun Xia](https://junxia97.github.io/), [Yingwei Ma](https://yingweima2022.github.io/), Jian Ma, [Xinwang Liu](https://xinwangliu.github.io/), Shengju Yu, Kejun Zhang, Wenliang Zhong


<p align="center">  <a href="https://pytorch.org/" alt="PyTorch">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" />  </a> <a href="https://icml.cc/Conferences/2023" alt="Conference"> <img src="https://img.shields.io/badge/NeurIPS'24-brightgreen" /> </a>
</p>


</div>

<p align = "justify">
Intent learning, which aims to learn users' intents for user understanding and item recommendation, has become a hot research spot in recent years. However, existing methods suffer from complex and cumbersome alternating optimization, limiting performance and scalability. To this end, we propose a novel intent learning method termed <u>ELCRec</u>, by unifying behavior representation learning into an <u>E</u>nd-to-end <u>L</u>earnable <u>C</u>lustering framework, for effective and efficient <u>Rec</u>ommendation. Concretely, we encode user behavior sequences and initialize the cluster centers (latent intents) as learnable neurons. Then, we design a novel learnable clustering module to separate different cluster centers, thus decoupling users' complex intents. Meanwhile, it guides the network to learn intents from behaviors by forcing behavior embeddings close to cluster centers. This allows simultaneous optimization of recommendation and clustering via mini-batch data. Moreover, we propose intent-assisted contrastive learning by using cluster centers as self-supervision signals, further enhancing mutual promotion Experimental results demonstrate the superiority of ELCRec from six perspectives. Compared to the runner-up, ELCRec improves NDCG@5 by 8.9% and reduces computational costs by 22.5% on the Beauty dataset. Due to the scalability and universal applicability, we deploy ELCRec on the industrial recommendation system with 130 million page views and achieve promising results.
</p>

<img src="./assets/intent_learning.png" alt="radar_plot" style="zoom:1%;" />
<p align="center">
Figure 1. Intent learning in recommendation.
</p>

<img src="./assets/elcrec.png" alt="radar_plot" style="zoom:61%;" />
<p align="center">
Figure 2. ELCRec framework.
</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>


# Usage



### Requirements

- Python >= 3.7
- Pytorch >= 1.2.0
- tqdm == 4.26.0
- faiss-gpu==1.7.1



### Quick Start

For evaluation:

```
cd ./src
```


```
bash ./scripts/run_sports.sh

ELCRec-Sports_and_Outdoors-1
{'Epoch': 0, 'HIT@5': '0.0286', 'NDCG@5': '0.0185', 'HIT@20': '0.0648', 'NDCG@20': '0.0286'}
```



```
bash ./scripts/run_beauty.sh

ELCRec-Beauty-1
{'Epoch': 0, 'HIT@5': '0.0529', 'NDCG@5': '0.0355', 'HIT@20': '0.1079', 'NDCG@20': '0.0509'}
```



```
bash ./scripts/run_toys.sh

ELCRec-Toys_and_Games-1
{'Epoch': 0, 'HIT@5': '0.0585', 'NDCG@5': '0.0403', 'HIT@20': '0.1138', 'NDCG@20': '0.0560'}
```



```
bash ./scripts/run_yelp.sh

ELCRec-Yelp-1
{'Epoch': 0, 'HIT@5': '0.0236', 'NDCG@5': '0.0150', 'HIT@20': '0.0653', 'NDCG@20': '0.0266'}
```


## Acknowledgements

- [SASRec](https://github.com/kang205/SASRec): the official implement of SASRec model.
- [ICLRec](https://github.com/salesforce/ICLRec): the official implement of ICLRec model.

# Citation

```
@inproceedings{liuyue_ELCRec,
  title={End-to-end Learnable Clustering for Intent Learning in Recommendation},
  author={Liu, Yue and Zhu, Shihao and Xia, Jun and Ma, Yingwei and Ma, Jian and Zhong, Wenliang and Liu, Xinwang and Yu, Shengju and Zhang, Kejun},
  booktitle={Proc. of NeurIPS},
  year={2024}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>