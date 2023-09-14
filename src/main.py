import hydra
from omegaconf import DictConfig

import os
import shutil

import h5py
import pytorch_lightning as pl
import torch

from dataset import (
    DummyDataModule,
    CustomDataModule,
)

from model import Model


@pl.utilities.rank_zero.rank_zero_only
def resume_cwd(cfg: DictConfig) -> None:
    folder_cwd = os.getcwd()
    folder_cwd_split = os.path.split(folder_cwd)
    folder_search = os.path.sep.join(folder_cwd_split[:-1])
    folder_list = sorted([val for val in os.listdir(folder_search) if val != folder_cwd_split[-1]], reverse=True)
    for folder in folder_list:
        folder_resume = os.path.join(folder_search, folder)
        if os.path.exists(os.path.join(folder_resume, cfg.sub_folder_ckpt, 'last.ckpt')):
            os.chdir(folder_resume)
            shutil.rmtree(folder_cwd)
            break
        else:
            shutil.rmtree(folder_resume)
    return


@pl.utilities.rank_zero.rank_zero_only
def save_results(results_all):
    for phase, results in results_all.items():
            with h5py.File(f'{phase}.h5', 'w') as f:
                for key, val in results.items():
                    f.create_dataset(key, data=val.numpy(), compression='gzip')
    return


@hydra.main(config_path='config', config_name='config_dummy', version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=None, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if cfg.debug:
        datamodule = DummyDataModule(cfg)
    else:
        datamodule = CustomDataModule(cfg)
    model = Model(cfg)
    if 'folder_pretrain' in cfg:
        folder_pretrain = hydra.utils.to_absolute_path(cfg.folder_pretrain)
        filenames = [val for val in os.listdir(folder_pretrain) if val.startswith('best_')]
        filenames = sorted(filenames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        path_pretrain = os.path.join(folder_pretrain, filenames[-1])
        state_dict = torch.load(hydra.utils.to_absolute_path(path_pretrain))['state_dict']
        model.load_state_dict(state_dict)
    if cfg.resume and cfg.use_timestamp:
        resume_cwd(cfg)
    folder_ckpt = os.path.join(os.getcwd(), cfg.sub_folder_ckpt)
    ckpt_best_callback = pl.callbacks.ModelCheckpoint(
        dirpath=folder_ckpt,
        filename='best_step-{step}',
        monitor='val/loss_opt',
        mode='min',
        auto_insert_metric_name=False,
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )
    ckpt_last_callback = pl.callbacks.ModelCheckpoint(
        dirpath=folder_ckpt,
        monitor=None,
        save_last=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
        every_n_epochs=0,
        save_on_train_epoch_end=False,
    )
    ckpt_path = os.path.join(folder_ckpt, 'last.ckpt')
    resume = cfg.resume and os.path.exists(ckpt_path)
    if cfg.training:
        if resume:
            global_step = torch.load(ckpt_path)['global_step']
            logger = pl.loggers.TensorBoardLogger('.', version=0, purge_step=global_step - 1)
        else:
            logger = pl.loggers.TensorBoardLogger('.', version=0)
    else:
        logger = False
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=-1,
        num_nodes=1,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=[ckpt_best_callback, ckpt_last_callback],
        fast_dev_run=False,
        max_epochs=None,
        max_steps=cfg.trainer.max_steps,
        max_time=None,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        deterministic=False,
        benchmark=True,
        inference_mode=True,
        use_distributed_sampler=True,
        profiler=None,
        detect_anomaly=False,
        barebones=False,
        plugins=None,
        sync_batchnorm=False,
        reload_dataloaders_every_n_epochs=0,
        default_root_dir=None,
    )
    if cfg.training:
        datamodule.setup('fit')
        if resume:
            trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, datamodule)
    if cfg.predicting:
        folder_ckpt = 'checkpoints'
        filenames = [val for val in os.listdir(folder_ckpt) if val.startswith('best_')]
        filenames = sorted(filenames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_path = os.path.join(folder_ckpt, filenames[-1])
        datamodule.setup('predict')
        results_all_list = None
        for _ in range(cfg.num_tests):
            results_all = trainer.predict(model, datamodule, ckpt_path=ckpt_path)
            results_all = [
                {
                    key: torch.cat([sub_results[key] for sub_results in results], dim=0)
                    for key in results[0]
                }
                for results in results_all
            ]
            if results_all_list is None:
                results_all_list = [[val] for val in results_all]
            else:
                for idx, results in enumerate(results_all):
                    results_all_list[idx].append(results)
        results_test, results_general = [
            {key: torch.stack([sub_val[key] for sub_val in val], dim=0) for key in val[0]}
            for val in results_all_list
        ]
        results_all = {'test': results_test, 'general': results_general}
        save_results(results_all)
    return


if __name__ == '__main__':
    main()
