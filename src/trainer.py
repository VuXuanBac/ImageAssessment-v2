from optimizer import create_optimizer, LRUpdater
import torch
from torch import nn
import os
import time
from utils import *


def initialize_model_weight(model, weight_init_info: dict):
    if weight_init_info is None or "strategy" not in weight_init_info:
        return model
    strategy = weight_init_info.get("strategy", None)
    submodel_name = weight_init_info.get("submodel", None)
    strategy = str(strategy).lower()
    if strategy == "xavier":
        func = nn.init.xavier_uniform_
    elif strategy == "normal":
        func = nn.init.normal_
    elif strategy == "uniform":
        func = nn.init.uniform_
    else:
        func = lambda x: nn.init.kaiming_uniform_(x, nonlinearity="relu")
    with torch.no_grad():

        def init_weight(m):
            if type(m) == nn.Linear:
                func(m.weight)

        if submodel_name is None:
            model.apply(init_weight)
        else:
            getattr(model, submodel_name).apply(init_weight)
    return model

class Trainer(object):
    def __init__(
        self,
        optimizer_info: dict,
        criterion,
        dataloader,
        lr_updater_info: dict = None,
        weight_init_info: dict = None,
    ) -> None:
        self.optimizer_info = optimizer_info
        self.lr_updater_info = lr_updater_info
        self.weight_init_info = weight_init_info
        self.criterion = criterion
        self.dataloader = dataloader        

    def set_model(self, model, checkpoint_file: str = None, device = None):
        
        if checkpoint_file is not None:
            if not os.path.exists(checkpoint_file):
                raise Exception(f'...No file at {checkpoint_file}. Can not load pretrained model...')
            data = torch.load(checkpoint_file)
            if "model" in data:
                model.load_state_dict(data["model"])
                model.to(device)
                
            optimizer = create_optimizer(model, **self.optimizer_info)
            if "optim" in data:
                optimizer.load_state_dict(data["optim"])
            
            self.model = model
            self.optimizer = optimizer
            print(':: Load checkpoint successfully ::')
        else:
            model.to(device)
            self.model = initialize_model_weight(model, self.weight_init_info)
            self.optimizer = create_optimizer(self.model, **self.optimizer_info)
        self.device = device

    def train(
        self,
        n_epochs: int = 100,
        sample_weights=None,
        unfreeze_after: int = None,
        verbose=True,
        checkpoint_dir: str = None,
        state_file: str = None,
        validation_dataloader=None,
        start_epoch: int = 0,
    ):

        model = self.model
        device = self.device
        optimizer = self.optimizer
        lr_updater = (
            LRUpdater(self.optimizer, **self.lr_updater_info, start_epoch=start_epoch) if self.lr_updater_info else None
        )
        dataloader = self.dataloader
        criterion = self.criterion

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if unfreeze_after and start_epoch > unfreeze_after:
            unfreeze(model)

        if verbose:
            _N, _B = len(dataloader.dataset), dataloader.batch_size
            _T, _U = get_parameters_info(model)
            print(f":: Parameters       : {_T} [Trainable], {_U} [Not Trainable]")
            print(f":: Data             : {_N} samples [batch {_B}]")
            print(f":: Epochs           : {n_epochs} [continue from {start_epoch + 1}]")
            print(f":: Device           : {device}")
            print(f":: Optimizer        :", optimizer_info(optimizer))

        print_in_step = verbose and (_N / _B) > 1000


        model.train()
        for epoch in range(start_epoch, n_epochs):
            if unfreeze_after and epoch == unfreeze_after:
                unfreeze(model)
                print(f":: after unfreezing ::", get_parameters_info(model))

            ss = time.time()
            losses = []
            for ind, (_, images, labels) in enumerate(dataloader, 1):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # reset all weights' gradients to 0 -> not accumulate gradient

                outputs = model(images)

                bloss = criterion(
                    outputs.view(len(labels), -1, 1), labels
                )  # Loss on one batch
                #print(outputs.shape, labels.shape, bloss.shape)
                losses.append(bloss.detach())

                mean_bloss = bloss.mean()

                mean_bloss.backward()  # Back propagation
                optimizer.step()  # Update parameters

                if print_in_step and ind % 100 == 0:
                    print(
                        f"Epoch {epoch + 1:>2} | Step {ind:>4} | Training loss: {mean_bloss:.4f} [Min: {min(bloss):.4f}, Max: {max(bloss):.4f}]: Time {(time.time() - ss)/60.:.2f} mins."
                    )
            ############### DONE ONE EPOCH ###############
            with torch.no_grad():
                epoch_losses = torch.cat(
                    losses, dim=0
                )  # does not follow grad on phase_losses from this

                if sample_weights is not None:
                    _losses = epoch_losses * sample_weights
                    _losses /= _losses.count_nonzero() / len(_losses)
                    train_avg_loss = _losses.sum().item()
                else:
                    train_avg_loss = epoch_losses.mean().item()

                print(
                    f"====> Epoch {epoch + 1:>2}:   Training loss: {train_avg_loss:.4f} [Min: {min(epoch_losses):.4f}, Max: {max(epoch_losses):.4f}]: Time {(time.time() - ss)/60.:.2f} mins."
                )

            if validation_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    validation_avg_loss = 0.0
                    for ids, images, labels in validation_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        bloss = criterion(outputs.view(len(labels), -1, 1), labels)
                        validation_avg_loss += bloss.sum().item()
                    validation_avg_loss /= len(validation_dataloader.dataset)
                    print(
                        f"====> Epoch {epoch + 1:>2}: Validation loss: {validation_avg_loss:.4f}"
                    )

                if lr_updater is not None:  # Learning rate decay
                    lr_updater.step(validation_avg_loss)
                if state_file:
                    write_state_on_file(state_file, train_avg_loss, validation_avg_loss)
                model.train()

            else:
                if lr_updater is not None:  # Learning rate decay
                    lr_updater.step(train_avg_loss)
                if state_file:
                    write_state_on_file(state_file, train_avg_loss)

            if checkpoint_dir:
                set_checkpoint(epoch + 1, model, optimizer, checkpoint_dir)
