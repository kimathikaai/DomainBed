# Domain Generalization for Domain-Linked Classes
Code release for "Domain Generalization for Domain-Linked Classes" NeurIPS 2023 submission.
This repostitory is a modification of "DomainBed". DomainBed is a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in ["In Search of Lost Domain Generalization"](https://arxiv.org/abs/2007.01434).

## Evaluation Datasets
Here are the evaluation datasets and their [interfaces](domainbed/datasets.py):

* VLCS  ([Fang et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf)) ([Download](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077)) ([Download](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8))
* Office-Home ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522)) ([Download](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC))

## Requirements
Refer to [chai_requirements.txt](chai_requirements.txt) and [chai_conda.txt](chai_conda.txt)

## Algorithms
Refer to the algorithm implementations in [algorithms.py](domainbed/algorithms.py):
#### Baselines
* Empirical Risk Minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Deep CORAL (CORAL, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719))
* Meta Learning Domain Generalization (MLDG, [Li et al., 2017](https://arxiv.org/abs/1710.03463))
* Adaptive Risk Minimization (ARM, [Zhang et al., 2020](https://arxiv.org/abs/2007.02931)), contributed by [@zhangmarvin](https://github.com/zhangmarvin)
* Self-supervised Contrastive Regularization (SelfReg, [Kim et al., 2021](https://arxiv.org/abs/2104.09841))
* Optimal Representations for Covariate Shift (CAD & CondCAD, [Ruan et al., 2022](https://arxiv.org/abs/2201.00057)), contributed by [@ryoungj](https://github.com/ryoungj)
* Quantifying and Improving Transferability in Domain Generalization (Transfer, [Zhang et al., 2021](https://arxiv.org/abs/2106.03632)), contributed by [@Gordon-Guojun-Zhang](https://github.com/Gordon-Guojun-Zhang)
* Invariant Causal Mechanisms through Distribution Matching (CausIRL with CORAL or MMD, [Chevalley et al., 2022](https://arxiv.org/abs/2206.11646)), contributed by [@MathieuChevalley](https://github.com/MathieuChevalley)
* Empirical Quantile Risk Minimization (EQRM, [Eastwood et al., 2022](https://arxiv.org/abs/2207.09944)), contributed by [@cianeastwood](https://github.com/cianeastwood)
* PGrad: Learning Principal Gradients For Domain Generalization (Grad, [Wang, Z., 2023](https://openreview.net/forum?id=CgCmwcfgEdH))
#### Variants
* FOND
* FOND_F
* FOND_FB
* FOND_FBA

## Hyper-Parameter Search
Refer to [hparams_registry.py](domainbed/hparams_registry.py) for hyper-parameter search spaces for each algorithm

## Quick Start
Download the datasets:

```bash
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

Train a model:

```bash
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/PACS/\
       --algorithm FOND\
       --dataset PACS\
       --test_env 2\
       --overlap high
```

Launch a sweep:

```bash
python3 -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher local\
       --algorithms ERM FOND CAD Transfer\
       --datasets PACS VLCS OfficeHome\
       --overlaps low high\
       --steps 5001\
       --single_test_envs\
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation
```

Visualize the results of your sweep:

```bash
python3 -m domainbed.scripts.collect_results\
        --input_dir=/my/sweep/output/path\
        --eval_metric=nacc\
        --selec_metric=nacc\
        --overlap=high\
        --latex
```
