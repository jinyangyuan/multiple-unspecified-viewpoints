#!/bin/bash

folder_src='src'
folder_downloads='downloads'
if [ ! -d $folder_downloads ]; then mkdir $folder_downloads; fi

name='clevr'
path_config='config_'$name'_multi.yaml'
folder_base='../clevr-dataset-gen/output_viewpoint_1/images_2'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='.'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out \
    --multiview
python $folder_src'/create_viewpoint.py' \
    --name $name'_viewpoint' \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out

name='shop'
path_config='config_'$name'_multi.yaml'
folder_base='../shop-vrb-gen/output_multi_1/images_2'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='.'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out \
    --multiview
python $folder_src'/create_viewpoint.py' \
    --name $name'_viewpoint' \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out
