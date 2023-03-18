#!/bin/bash

overlap=0
dataset='PACS'
input_dir='/pub2/podg/'

selec_metric='acc'
eval_metric='acc'
python -m domainbed.scripts.collect_results\
	--input_dir=$input_dir \
    --eval_metric=$eval_metric \
    --selec_metric=$selec_metric \
    --overlap=$overlap \
    --dataset=$dataset \
	--latex

selec_metric='acc'
eval_metric='f1'
python -m domainbed.scripts.collect_results\
	--input_dir=$input_dir \
    --eval_metric=$eval_metric \
    --selec_metric=$selec_metric \
    --overlap=$overlap \
    --dataset=$dataset \
	--latex

selec_metric='acc'
eval_metric='oacc'
python -m domainbed.scripts.collect_results\
	--input_dir=$input_dir \
    --eval_metric=$eval_metric \
    --selec_metric=$selec_metric \
    --overlap=$overlap \
    --dataset=$dataset \
	--latex

selec_metric='acc'
eval_metric='nacc'
python -m domainbed.scripts.collect_results\
	--input_dir=$input_dir \
    --eval_metric=$eval_metric \
    --selec_metric=$selec_metric \
    --overlap=$overlap \
    --dataset=$dataset \
	--latex
