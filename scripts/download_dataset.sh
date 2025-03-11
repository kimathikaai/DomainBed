#!/bin/bash

base_dir=/mnt/pub0/projects/fond

python3 -m domainbed.scripts.download \
    --data_dir=$base_dir/data \
    --pacs --vlcs --office_home --camelyon17
