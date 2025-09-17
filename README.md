# DMON-ARO: Neighbor-Based Feature and Index Enhancement for Person Re-Identification

<div align='center'>
    <a href='https://arxiv.org/abs/2504.11798'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

A ***Plug-and-Play*** method designed Not only for ReID task, but all retrival tasks.

## ⚒️ Usage
```bash
# get the features qf and gf from query and gallery images/text or some others modality

from DMON import dmon
qf = dmon(qf, k1 = 3)
gf = dmon(gf, k1 = 3)

from ARO import aro
dist = aro(qf, gf, k2=2)

# your rank options
```

## 📒 Citation

If you find our work useful for your research, please consider citing the paper:

```bash
@inproceedings{yuan2025neighbor,
  title={Neighbor-Based Feature and Index Enhancement for Person Re-Identification},
  author={Yuan, Chao and Zhang, Tianyi and Niu, Guanglin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5762--5769},
  year={2025}
}
```
