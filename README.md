## Object-Centric Learning with Object Constancy (OCLOC)

This is the code repository of the paper "Unsupervised Learning of Compositional Scene Representations from Multiple Unspecified Viewpoints".

### Dependencies

- pytorch == 2.0
- pytorch_lightning == 2.0
- hydra == 1.3
- omegaconf == 2.3
- numpy == 1.24
- h5py == 3.9
- pyyaml == 6.0
- tensorboard == 2.14
- scipy == 1.10
- scikit-learn == 1.3

### Datasets

Synthesize images in the CLEVR dataset using Blender. The tested version of Blender is 2.79.

```bash
cd ../image_generation_viewpoint_1
./create_blend.sh
./create_pngs.sh
cd ../..
```

Synthesize images in the SHOP dataset using Blender. The tested version of Blender is 2.83.

```bash
cd ../image_generation_multi_1
./create_blend.sh
./create_pngs.sh
cd ../..
```

Create datasets in the HDF5 format.

```bash
cd data
./create_datasets.sh
cd ..
```

### Experiments

Train and test models.

```bash
cd exp_multi
./run.sh
cd ../exp_multi_no_shadow
./run.sh
cd ..
```

Run `exp_multi/evaluate.ipynb` and `exp_multi_no_shadow/evaluate.ipynb` to evaluate the performance.
