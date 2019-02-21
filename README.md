# DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System

This is our official implementation for the paper:

Zhi-Hong Deng, Ling Huang, Chang-Dong Wang, Jian-Huang Lai, Philip S. Yu. [DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System.](https://arxiv.org/abs/1901.04704v1) In AAAI '19(**Oral**), Honolulu, Hawaii, USA, January 27 – February 1, 2019.

In this work, we explored the possibility of fusing representation learning-based CF methods and matching function learning-based CF methods. We have devised a general framework DeepCF and proposed its MLP implementation, i.e., CFNet. The DeepCF framework is simple but effective. Although we have implemented the two components with MLP in this paper, different types of representation learning-based methods and matching function learning-based methods can be integrated under the DeepCF framework. 

**Please cite our AAAI'19 paper if you use our codes. Thanks!** 

## Environment Settings
We use Keras with tensorflow as the backend. 
- Keras version: '2.1.4'
- tensorflow-gpu version:  '1.7.0'

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run CFNet-rl:
```
python DMF.py --dataset ml-1m --epochs 20
```

Run CFNet-ml:
```
python MLP.py --dataset ml-1m --epochs 20
```

Run CFNet (without pre-training): 
```
python CFNet.py --dataset ml-1m --epochs 20 --lr 0.01
```

Run CFNet (with pre-training):
```
python CFNet.py --dataset ml-1m --epochs 20 --lr 0.0001  --learner sgd  --dmf_pretrain Pretrain/ml-1m_DMF_1533478530.h5 --mlp_pretrain Pretrain/ml-1m_MLP_1533344127.h5
```

### Dataset
We provide two processed datasets: MovieLens 1 Million (ml-1m) and Amazon Music (AMusic). 

**train.rating**
- Train file.
- Each Line is a training instance: `userID\t itemID\t rating\t timestamp (if have)`

**test.rating**
- Test file (positive instances). 
- Each Line is a testing instance: `userID\t itemID\t rating\t timestamp (if have)`

**test.negative**
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: `(userID,itemID)\t negativeItemID1\t negativeItemID2 ...`

## Citation
```
@article{deng2019deepcf,
  title={DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System},
  author={Deng, Zhi-Hong and Huang, Ling and Wang, Chang-Dong and Lai, Jian-Huang and Yu, Philip S},
  booktitle={AAAI},
  year={2019}
}
```
If the code helps you in your research, please also cite:
```
@misc{deepcf2019,
  author =       {Deng, Zhi-Hong and Huang, Ling and Wang, Chang-Dong and Lai, Jian-Huang and Yu, Philip S},
  title =        {DeepCF},
  howpublished = {\url{https://github.com/familyld/DeepCF}},
  year =         {2019}
}
```
