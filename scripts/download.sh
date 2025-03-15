#!/bin/bash

base_dir=/mnt/pub0/projects/fond

# Download datasets
python3 -m domainbed.scripts.download \
    --data_dir=$base_dir/data \
    --pacs --vlcs --office_home --camelyon17

# Download resnet pretrained weights
curl https://download.pytorch.org/models/resnet18-f37072fd.pth \
    --output $base_dir/saved/resnet18-f37072fd.pth
