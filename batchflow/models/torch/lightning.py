""" Utilities to seamlessly convert `BatchFlow` model and data generation pipeline to
`Pytorch Lightning` compatible modules to leverage all of its features.

>>> lightning_model = batchflow_model.to_lightning()
>>> dataloader = ppl.to_dataloader('images', 'labels', batch_size=64)

>>> trainer = Trainer(gpus=7, max_steps=100, amp_backend='native', auto_lr_find=True)
>>> trainer.fit(lightning_model, dataloader)
"""
import numpy as np

try:
    from pytorch_lightning import LightningModule
except ImportError:
    LightningModule = object

try:
    from torch.utils.data import DataLoader, IterableDataset
except ImportError:
    IterableDataset = object


class PipelineDataset(IterableDataset):
    """ Wrap pipeline into `PyTorch` interface. """
    def __init__(self, pipeline, components=None, batch_size=64):
        if IterableDataset == object:
            raise ImportError('Install `Pytorch` module first')
        self.pipeline = pipeline
        self.components = components
        self.batch_size = batch_size

    def get_data(self):
        """ Yield batch items endlessly. """
        while True:
            batch = self.pipeline.next_batch(self.batch_size)
            yield [getattr(batch, comp).astype(np.float32) for comp in self.components]

    def __iter__(self):
        return self.get_data()


class PipelineDataLoader(DataLoader):
    """ Create dataloader from `Pipeline` or `PipelineDataset`. """
    def __init__(self, dataset, components=None, batch_size=None, pin_memory=True, **kwargs):
        from ...pipeline import Pipeline #pylint: disable=relative-beyond-top-level
        if isinstance(dataset, Pipeline):
            if components is not None and isinstance(batch_size, int):
                dataset = PipelineDataset(dataset, components=components, batch_size=batch_size)
                batch_size = None
            else:
                raise ValueError('Provide `components` and `batch_size` parameters to dataloader initialization.')

        if not isinstance(dataset, PipelineDataset):
            raise TypeError('`PipelineDataLoader` is supposed to be used with `Pipeline` or `PipelineDataset`')

        super().__init__(dataset, batch_size=batch_size, pin_memory=pin_memory, **kwargs)



class LightningModel(LightningModule):
    """ Convert `TorchModel` from `BatchFlow` to `PyLightning`.
    Falcon WA, et al "`PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_"
    """
    def __init__(self, batchflow_model):
        if LightningModule == object:
            raise ImportError('Install `PyTorch Lightning` module first')
        super().__init__()
        self.batchflow_model = batchflow_model

    def forward(self, x):
        return self.batchflow_model.model(x)

    def configure_optimizers(self):
        return self.batchflow_model.train_steps['']['optimizer']

    def training_step(self, batch, batch_idx):
        """ Define custom training iteration on given data. """
        _ = batch_idx
        inputs, targets = batch
        predictions = self(inputs)

        loss_func = self.batchflow_model.train_steps['']['loss'][0]
        loss = loss_func(predictions, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}
