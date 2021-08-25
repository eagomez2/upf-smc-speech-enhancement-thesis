import os
import glob
import torch
import numpy as np
from abc import ABC
from tqdm import tqdm
from typing import List, Union, Optional
from datetime import datetime
from prettytable import PrettyTable
from collections.abc import Iterable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Uncomment the next line to enable a Nan detector for debugguing
# torch.autograd.set_detect_anomaly(True)

class TrainingProcessProgressBar:
    def __init__(self,
                 iterator: Iterable,
                 description: str = None,
                 leave: bool = True,
                 unit: str = "step"):
        super().__init__()
        self.iterator = iterator
        self._leave = leave
        self.description = description
        self.unit = unit
        self._postfix_dict = {}
        self._pbar = self._init_pbar()

    def __call__(self):
        return self._pbar

    def _init_pbar(self):
        pbar = tqdm(self.iterator,
                    leave=self._leave,
                    postfix=self._postfix_dict,
                    unit=self.unit)

        return pbar
    
    def set_description(self, description: str):
        self._pbar.set_description(description)

    def set_value(self, k: str, v):
        self._postfix_dict[k] = v
        self._pbar.set_postfix(self._postfix_dict)

    def close(self):
        self._pbar.close()
        

class TrainingProcessRunningDict:
    def __init__(self):
        super().__init__()
        self._data = {}

    def set_value(self, 
                  k: Union[str, dict],
                  v: Optional[Union[str, int, float, None]] = None):
        if isinstance(k, dict):
            for k_, v_ in k.items():
                self.set_value(k_, v_)
        else:
            if k not in self._data.keys():
                self._data[k] = [v]
            else:
                self._data[k].append(v)

    def update_value(self, k: str, v: str, strict: bool = True):
        if k not in self._data.keys():

            if strict:
                raise KeyError(f"Key key={k} not found. "
                               f"Available keys={list(self._data.keys())}")
            else:
                self.remove_key(k, strict=False)
                self.set_value(k, v)

    def get_value(self, k: str, strict : bool = True):
        # an error is raised if strict, otherwise None is returned
        try:
            v = self._data[k]
        except KeyError:
            if strict:
                raise KeyError(f"Key key={k} not found. "
                               f"Available keys={list(self._data.keys())}")
            else:
                v = None

        return v

    def get_last_value(self, k: str, strict: bool = True):
        try:
            last_value = self.get_value(k, strict)[-1]
        except TypeError:
            if strict:
                raise TypeError
            else:
                last_value = None

        return last_value

    def remove_key(self, k: str, strict: bool = False):
        if strict:
            del self._data[k]  # raises error if key does not exist
        else:
            self._data.pop(k, None)

    def flush(self, exceptions: list = []):
        # copy keys to avoid 'changed size' error during iteration
        for k in list(self._data.keys()):
            if k not in exceptions:
                self.remove_key(k)


class TrainingProcessCallback(ABC):
    def __init__(self):
        super().__init__()

    def on_val_epoch_start(self, training_process):
        ...

    def on_val_epoch_end(self, training_process):
        ...


class TrainingProcess:
    """ Base class to create a training cycle of a network. """
    def __init__(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader = None,
                 max_epochs: int = 1000,
                 device: torch.device = torch.device("cpu"),
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 lr_scheduler_start_epoch: int = 0,
                 grad_norm_clipping: float = None,
                 resume_from_checkpoint: str = None,
                 overfit_single_batch: bool = False,
                 disable_callbacks_in_overfit_single_batch: bool = False,
                 run_name: str = "model",
                 run_name_prefix: str = "",
                 run_ts_fmt: str = "%Y-%m-%d %Hh%Mm%Ss",
                 logs_dir: str = "logs",
                 callbacks: List[TrainingProcessCallback] = []
                 ):
        super().__init__()

        # dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # checkpoint params
        self.resume_from_checkpoint = resume_from_checkpoint
        self.run_ts_fmt = run_ts_fmt
        self._base_run_name = run_name
        self.run_name_prefix = run_name_prefix
        self.run_name = self._init_run_name()
        self.logs_dir = logs_dir
        self.callbacks = callbacks

        # training params
        self.device = device
        self.max_epochs = max_epochs
        self.grad_norm_clipping = grad_norm_clipping
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_start_epoch = lr_scheduler_start_epoch

        # training mode
        self.resume_from_checkpoint = resume_from_checkpoint
        self.overfit_single_batch = overfit_single_batch
        self.disable_callbacks_in_overfit_single_batch = \
                disable_callbacks_in_overfit_single_batch

        # training process preparation
        self.logger = self._init_logger()
        self.checkpoints_dir = self._init_checkpoints_dir()
        self.current_epoch = None  # None if training hasn't started yet

        # training running objects (exist only during fit)
        # holds data in training and validation
        self.running_dict = TrainingProcessRunningDict()
        self._train_pbar = None
        self._val_pbar = None

    def _init_run_name(self):
        # if a checkpoint is loaded, run_name is derived from the dir structure
        if self.resume_from_checkpoint is not None:
            run_name = os.path.basename(
                            os.path.dirname(
                                os.path.dirname(self.resume_from_checkpoint)
                            )
                        )

        # no checkpoint
        elif self.run_ts_fmt is not None:
            ts_str = datetime.now().strftime(self.run_ts_fmt)
            run_name = f"{self.run_name_prefix}{self._base_run_name} {ts_str}"
        # no checkpoint and no ts
        else:
            run_name = self._base_run_name

        return run_name

    def _init_logger(self) -> SummaryWriter:
        return SummaryWriter(os.path.join(self.logs_dir, self.run_name))

    def _init_checkpoints_dir(self) -> str:
        # checkpoints folder is fixed
        return os.path.join(self.logs_dir, self.run_name, "checkpoints")

    def _init_overfit_batch(self):
        return next(iter(self.train_dataloader))

    def _get_dataset_name_from_dataloader(self, dataloader: DataLoader):
        if dataloader is not None:
            dataset_name = dataloader.dataset.__class__.__name__
        else:
            dataset_name = None

        return dataset_name

    # TODO: decide how to get val_loss if reduced direct or how to add
    # more information to the saved checkpoint
    # TODO: Expand checkpoint dict to save multiple models
    def save_checkpoint(self, prefix="", ext=".tar"):
        # make checkpoint dir if it does not exist
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        # compose checkpoint filename
        ts = datetime.now().strftime(self.run_ts_fmt)

        # NOTE: overfit_single_batch_mode has no validation process
        if self.has_validation and not self.is_in_overfit_single_batch_mode:
            avg_val_loss = self.running_dict.get_last_value("avg_val_loss")

            checkpoint_file = (
                        f"{prefix}{self._base_run_name} {ts} "
                        f"epoch={self.current_epoch} "
                        f"val_loss={avg_val_loss:.4f}{ext}"
                    )
        else:
            checkpoint_file = (
                        f"{prefix}{self._base_run_name} {ts} "
                        f"epoch={self.current_epoch}{ext}"
                    )
        
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_file)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_epoch": self.current_epoch
            }, checkpoint_path)
        
        # checkpoint path is returned to be used by callbacks in case they need
        # to keep track of different checkpoints saved over time
        return checkpoint_path

    def load_checkpoint(self, model, optimizer):
        # corroborate checkpoint file exists
        if not os.path.isfile(self.resume_from_checkpoint):
            raise FileNotFoundError("Checkpoint file not found: "
                                   f"{self.resume_from_checkpoint}")

        checkpoint = torch.load(self.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # start from the next epoch since the saved epoch is assumed
        # as completed
        self.current_epoch = checkpoint["last_epoch"] + 1
        return model, optimizer

    def _run_callback_hook(self, hook_name: str):
        if (self.is_in_overfit_single_batch_mode and
            self.disable_callbacks_in_overfit_single_batch):
            return

        for callback in self.callbacks:
            hook_fn = getattr(callback, hook_name, False)

            if callable(hook_fn):
                hook_fn(self)

    @property
    def has_validation(self) -> bool:
        return self.val_dataloader is not None

    @property
    def is_in_overfit_single_batch_mode(self) -> bool:
        return self.overfit_single_batch

    @property
    def callback_names(self) -> list:
        callback_names = [
                callback.__class__.__name__ for callback in self.callbacks
                ] if len(self.callbacks) > 0 else None
        return callback_names

    @property
    def train_dataset_name(self) -> str:
        return self._get_dataset_name_from_dataloader(self.train_dataloader)

    @property
    def val_dataset_name(self) -> str:
        return self._get_dataset_name_from_dataloader(self.val_dataloader)

    @property
    def criterion_name(self) -> str:
        if type(self.criterion).__name__ == "function":
            criterion_name = self.criterion.__name__
        else:
            criterion_name = self.criterion.__class__.__name__

        return criterion_name

    @property
    def model_name(self) -> str:
        return self.model.__class__.__name__

    @property
    def optimizer_name(self) -> str:
        return self.optimizer.__class__.__name__

    @property
    def lr_scheduler_name(self) -> str:
        if self.lr_scheduler is not None:
            lr_scheduler_name = self.lr_scheduler.__class__.__name__
        else:
            lr_scheduler_name = None
        
        return lr_scheduler_name

    def _print_model_summary(self, model: torch.nn.Module):
        layers = []
        layer_types = []
        layer_params = []

        for idx, (name, module) in enumerate(model.named_modules()):
            # skip first entry that corresponds to the module itself
            if idx == 0:
                continue

            layers.append(name)
            layer_types.append(module.__class__.__name__)
            layer_params.append(
                    sum(params.numel() for params in module.parameters())
                    )

        trainable_params = []
        nontrainable_params = []

        for param in model.parameters():
            if param.requires_grad:
                trainable_params.append(param.numel())
            else:
                nontrainable_params.append(param.numel())

        trainable_params = sum(trainable_params)
        nontrainable_params = sum(nontrainable_params)
        
        summary_table = PrettyTable()
        summary_table.add_column("#", range(len(layers)))
        summary_table.add_column("Layer", layers)
        summary_table.add_column("Type", layer_types)
        summary_table.add_column("Parameters", layer_params)
        print(f"{model.__class__.__name__} summary:")
        print(summary_table)
        print(f"Total parameters: {trainable_params + nontrainable_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {nontrainable_params}")        

    def _print_training_process_summary(self):
        # get callback names
        if self.callback_names is not None:
            callback_names_str = ", ".join(self.callback_names)
        else:
            callback_names_str = None
        
        # get time stamp
        process_start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.is_in_overfit_single_batch_mode:
            print("****** RUNNING IN OVERFIT SINGLE BATCH MODE ******\n"
                  "In 'overfit single batch mode' only the first batch "
                  "of the training_dataloader will be used for training. "
                  "Validation will be skipped."
                    )

        print(
        f"Model: {self.model_name}\n"
        f"Run name: {self.run_name}\n"
        f"Resumed from checkpoint: {self.resume_from_checkpoint is not None}\n"
        f"CUDA available: {torch.cuda.is_available()}\n"
        f"Device: {self.device}\n"
        f"Training dataset: {self.train_dataset_name}\n"
        f"Validation dataset: {self.val_dataset_name}\n"
        f"Checkpoints folder: {self.checkpoints_dir}\n"
        f"Process start date: {process_start_date}\n"
        f"Optimizer: {self.optimizer_name}\n"
        f"Learning rate scheduler: {self.lr_scheduler_name}\n"
        f"Gradient norm clipping: {self.grad_norm_clipping}\n"
        f"Criterion: {self.criterion_name}\n"
        f"Maximum epochs: {self.max_epochs}\n"
        f"Callbacks: {callback_names_str}\n"
        )


    def _on_train_setup(self):
        # prevents inconsistency in current epoch vs last epoch
        if (self.current_epoch is not None and 
            self.current_epoch >= self.max_epochs):
            raise ValueError("Expected max_epochs > current_epoch but got "
                            f"max_epochs={self.max_epochs} and "
                            f"current_epoch={self.current_epoch}")

        # self.current_epoch is None when training hasn't started yet
        # if training has been started from checkpoint, self.current_epoch
        # will be set accordingly in self.load_checkpoint()
        if self.current_epoch is None:
            self.current_epoch = 0

        self._print_training_process_summary()
        self._print_model_summary(self.model)
        print("\nTraining progress:")

    def on_train_setup(self):
        ...

    def _on_train_epoch_start(self, epoch):
        self.current_epoch = epoch

        # prepare model for training
        self.model.to(self.device)
        self.model.train()

        # if overfit single batch is set, only a subset will be put in the
        # iterator
        self._train_pbar = TrainingProcessProgressBar(
                            self.train_dataloader,
                            leave=(self.current_epoch + 1 == self.max_epochs))


        self._train_pbar.set_description(f"Epoch {self.current_epoch}"
                                         f"/{self.max_epochs - 1}")
                                    

        # update running dict
        self.running_dict.set_value("epoch", self.current_epoch)

    def on_train_epoch_start(self):
        ...

    def _on_train_step(self, batch_idx, batch):
        ...

    def on_train_step(self, batch_idx, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_norm_clipping
                    )

        self.optimizer.step()
        self.running_dict.set_value("train_loss", loss.item())

    def _on_train_step_end(self):
        # set last train loss in progress bar
        last_train_loss = self.running_dict.get_last_value("train_loss")
        self._train_pbar.set_value("train_loss", f"{last_train_loss:.4f}")

        if self.has_validation:
            # set last average val loss in progress bar if exists
            last_avg_val_loss = self.running_dict \
                                    .get_last_value("avg_val_loss", 
                                                    strict=False)

            if last_avg_val_loss is not None:
                self._train_pbar.set_value("avg_val_loss",
                                          f"{last_avg_val_loss:.4f}")

    def on_train_step_end(self):
        ...

    def _on_train_epoch_end(self):
        # close train progress bar
        self._train_pbar.close()

        # compute and log train loss
        avg_train_loss = np.mean(self.running_dict.get_value("train_loss"))
        self.logger.add_scalar("Loss/train", avg_train_loss, self.current_epoch)

        # remove train_loss list from running dict
        self.running_dict.remove_key("train_loss")
        self.running_dict.set_value("avg_train_loss", avg_train_loss)

    def on_train_epoch_end(self):
        ...

    def _on_val_epoch_start(self):
        # set up progress bar
        self._val_pbar = TrainingProcessProgressBar(self.val_dataloader,
                leave=(self.current_epoch + 1 == self.max_epochs))
        self._val_pbar.set_description(
            f"Validation of epoch {self.current_epoch}/{self.max_epochs - 1}"
        )

        # set last train loss
        avg_train_loss = self.running_dict.get_last_value("avg_train_loss")
        self._val_pbar.set_value("avg_train_loss", 
                                f"{avg_train_loss:.4f}")

        # remove val loss from last epoch
        self.running_dict.remove_key("avg_val_loss", strict=False)

        # set model for evaluation
        self.model.eval()

    def on_val_epoch_start(self):
        ...

    def _on_val_step(self, batch_idx, batch):
        ...

    def on_val_step(self, batch_idx, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        # log to running dict
        self.running_dict.set_value("val_loss", loss.item())

    def _on_val_step_end(self):
        last_val_loss = self.running_dict.get_last_value("val_loss")
        self._val_pbar.set_value("val_loss", f"{last_val_loss:.4f}")

    def on_val_step_end(self):
        ...

    def _on_val_epoch_end(self):
        # close validation progress bar
        self._val_pbar.close()

        # compute avg val loss and log it
        avg_val_loss = np.mean(self.running_dict.get_last_value("val_loss"))
        self.logger.add_scalar("Loss/validation", 
                               avg_val_loss, self.current_epoch)
        self.running_dict.remove_key("val_loss")
        self._train_pbar.set_value("avg_val_loss", f"{avg_val_loss:.4f}")

        self.running_dict.update_value("avg_val_loss", avg_val_loss, 
                                       strict=False)

    def on_val_epoch_end(self):
        # get avg_val_loss and update scheduler base on that
        avg_val_loss = self.running_dict.get_last_value("avg_val_loss")

        if (self.lr_scheduler is not None and
           (self.current_epoch + 1) > self.lr_scheduler_start_epoch):
            self.lr_scheduler.step(avg_val_loss)

    def _on_fit_epoch_end(self):
        self.running_dict.flush(exceptions=["avg_val_loss"])

    def on_fit_epoch_end(self):
        # log histogram
        model_name = self.model.__class__.__name__

        for name, param in self.model.named_parameters():
            self.logger.add_histogram(f"{model_name}.{name}", 
                                      param, self.current_epoch)
            self.logger.add_histogram(f"{model_name}.{name}.grad", 
                                      param.grad, self.current_epoch)

    def _run_train_loop(self, model, criterion, optimizer):
        # references to make model, criterion and optimizer accesible by
        # passing a reference to this class in callback hooks
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # internal hooks begin with _, external hooks have the same name
        self._on_train_setup()
        self.on_train_setup()
        self._run_callback_hook("on_train_setup")

        for epoch in range(self.current_epoch, self.max_epochs):
            self._on_train_epoch_start(epoch)
            self.on_train_epoch_start()
            self._run_callback_hook("on_train_epoch_start")

            for batch_idx, batch in enumerate(self._train_pbar()):
                self._on_train_step(batch_idx, batch)
                self.on_train_step(batch_idx, batch)

                self._on_train_step_end()
                self.on_train_step_end()

            self._on_train_epoch_end()
            self.on_train_epoch_end()
            self._run_callback_hook("on_train_epoch_end")

            if self.has_validation:

                with torch.no_grad():
                    self._on_val_epoch_start()
                    self.on_val_epoch_start()
                    self._run_callback_hook("on_val_epoch_start")

                    for batch_idx, batch in enumerate(self._val_pbar()):
                        self._on_val_step(batch_idx, batch)
                        self.on_val_step(batch_idx, batch)

                        self._on_val_step_end()
                        self.on_val_step_end()

                    self._on_val_epoch_end()
                    self.on_val_epoch_end()
                    self._run_callback_hook("on_val_epoch_end")

            self._on_fit_epoch_end()
            self.on_fit_epoch_end()
            self._run_callback_hook('on_fit_epoch_end')

    def _on_overfit_train_setup(self):
        # model will never be saved in overfit train setup
        self.current_epoch = 0
        self._print_training_process_summary()
        self._print_model_summary(self.model)
        print("\nOverfitting in progress...")

        self.model.to(self.device)
        self.running_dict.set_value("epoch", self.current_epoch)

    def _on_overfit_train_epoch_start(self, epoch):
        self.current_epoch = epoch
        self.model.train()

    def on_overfit_train_epoch_start(self, epoch):
        ...

    def _on_overfit_train_step(self, batch_idx, batch):
        ...

    def on_overfit_train_step(self, batch_idx, batch):
        # an inherited class from TrainingProcess should be created and it
        # should implement this method if used with overfit single batch mode
        raise NotImplementedError

    def _on_overfit_train_epoch_end(self):
        # set last train loss in progress bar
        last_train_loss = self.running_dict.get_last_value("train_loss")

        # avoids accumulating several values on the running dict
        self.running_dict.remove_key("train_loss")

        # simple print to track the loss in overfit mode
        print(f"Epoch {self.current_epoch}/{self.max_epochs - 1}, "
              f"train_loss: {last_train_loss:.4f}")

    def on_overfit_train_epoch_end(self):
        ...

    def _on_overfit_val_step(self, batch_idx, batch):
        self.model.eval()

    def on_overfit_val_step(self, batch_idx, batch):
        ...

    def _on_overfit_val_epoch_end(self):
        ...

    def on_overfit_val_epoch_end(self):
        ...

    def _on_overfit_fit_epoch_end(self):
        self.running_dict.flush()

    def on_overfit_fit_epoch_end(self):
        ...

    def _run_overfit_single_batch_loop(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self._on_overfit_train_setup()
        
        overfit_batch = next(iter(self.train_dataloader))

        for epoch in range(self.max_epochs):
            self._on_overfit_train_epoch_start(epoch)
            self.on_overfit_train_epoch_start(epoch)

            # equivalent to args (batch_idx, batch) of regular train loop
            self._on_overfit_train_step(0, overfit_batch)
            self.on_overfit_train_step(0, overfit_batch)

            self._on_overfit_train_epoch_end()
            self.on_overfit_train_epoch_end()

            with torch.no_grad():
                self._on_overfit_val_step(0, overfit_batch)
                self.on_overfit_val_step(0, overfit_batch)

                self._on_overfit_val_epoch_end()
                self.on_overfit_val_epoch_end()
                self._run_callback_hook("on_overfit_val_epoch_end")

        self._on_overfit_fit_epoch_end()
        self.on_overfit_fit_epoch_end()
        self._run_callback_hook("on_overfit_fit_epoch_end")


    def fit(self, model, criterion, optimizer):
        if self.resume_from_checkpoint is not None:
            model, optimizer = self.load_checkpoint(model, optimizer)
            self._run_train_loop(model, criterion, optimizer)
        elif self.overfit_single_batch:
            self._run_overfit_single_batch_loop(model, criterion, optimizer)
        else:
            self._run_train_loop(model, criterion, optimizer)
