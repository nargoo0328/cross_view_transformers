import torch
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
# from pytorch_lightning.utilities.distributed import group as _group
import json

class ModelModule(pl.LightningModule):
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None, nusc_metric=None, val_only=False):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['loss_func', 'metrics', 'scheduler_args','nusc_metric','val_only'],
            logger=False,
        )

        self.backbone = backbone
        self.loss_func = loss_func
        self.metrics = metrics
        self.nusc_metric = nusc_metric
        self.val_only = val_only
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):
        return self.backbone(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        loss, loss_details, weights = self.loss_func(pred, batch)
        if self.metrics is not None:
            if prefix == 'train':
                if not self.val_only:
                    self.metrics.update(pred, batch)
           
            else:
                self.metrics.update(pred, batch)
                
        if self.nusc_metric is not None:
            batch.update({'current_state':prefix})
            self.nusc_metric.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True, logger=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True,logger=True)
            if self.training and weights:
                self.log_dict({f'{prefix}/weights/{k}': v.detach() for k, v in weights.items()},
                        on_step=on_step, on_epoch=True,logger=True)
        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', on_step = True,
                                return_output = batch_idx % self.hparams.experiment.log_image_interval == 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', on_step = False,
                                return_output = batch_idx % self.hparams.experiment.log_image_interval == 0)

    def on_validation_start(self) -> None:
        if not self.val_only:
            self._log_epoch_metrics('train')
        # self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def on_validation_epoch_end(self): # validation_epoch_end
        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """ 
        if self.metrics is None:
            if self.nusc_metric is not None:
                self.compute_nusc_metric(prefix)
            return
        metrics = self.metrics.compute()
        ious = list()
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    # print(f'{prefix}/metrics/{key}{subkey}: {val}')
                    self.log(f'{prefix}/metrics/{key}{subkey}', val, on_epoch=True, logger=True)
            else:
                if 'IoU' in key:
                    ious.append(value)
                self.log(f'{prefix}/metrics/{key}', value, on_epoch=True, logger=True)

        self.log(f'{prefix}/metrics/mIoU', torch.stack(ious).mean(), on_epoch=True, logger=True)
        self.metrics.reset()


    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        weighting_param = [x for x in self.loss_func.parameters() if x.requires_grad]

        optimizer = torch.optim.AdamW(parameters+weighting_param, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def compute_nusc_metric(self, prefix):
        list_gather_obj = self._gather_objects(obj=self.nusc_metric.results_dict)
        if not self.trainer.is_global_zero:
            self.nusc_metric.reset()
            return
        else:
            results_dict = list_gather_obj[0]
            for d in list_gather_obj[1:]:
                results_dict['results'].update(d['results'])
            path = './tmp_result.json'
            with open(path, 'w') as f:
                json.dump(results_dict, f)
            metrics = self.nusc_metric.compute()

            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, val in value.items():
                        # print(f'{prefix}/metrics/{key}{subkey}: {val}')
                        self.log(f'{prefix}/metrics/{key}{subkey}', val,logger=True)
                else:
                    self.log(f'{prefix}/metrics/{key}', value,logger=True)
            self.nusc_metric.reset()

    def _gather_objects(self, obj):
        if not self.trainer.is_global_zero:
            print("\n\nNon-zero rank")
            dist.gather_object(obj=obj, object_gather_list=None, dst=0, group=_group.WORLD)
            return None
        else: # global-zero only
            print("\n\nGlobal-zero rank")
            list_gather_obj = [None] * self.trainer.world_size   # the container of gathered objects.
            dist.gather_object(obj=obj, object_gather_list=list_gather_obj, dst=0, group=_group.WORLD)
            return list_gather_obj
