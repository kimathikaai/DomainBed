#!/bin/bash

echo "*****************************0"
python -m domainbed.scripts.collect_results\
	--input_dir=/pub2/tmp/0 \
	--latex

echo "*****************************33"
python -m domainbed.scripts.collect_results\
	--input_dir=/pub2/tmp/33 \
	--latex

echo "*****************************66"
python -m domainbed.scripts.collect_results\
	--input_dir=/pub2/tmp/66 \
	--latex
