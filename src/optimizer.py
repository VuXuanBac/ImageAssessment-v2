import torch.optim as optim
import torch.optim.lr_scheduler as lr_updater

__all__ = ["create_optimizer", "LRUpdater"]


def create_optimizer(
    model, name: str, param_groups: list[dict], **kwargs
) -> optim.Optimizer:
    params_info = []
    for group in param_groups:
        _info = {}
        for k, v in group.items():
            if k == "params":
                _info[k] = (
                    model.parameters() if v == "" else getattr(model, v).parameters()
                )
            else:
                _info[k] = v
        params_info.append(_info)
    return getattr(optim, name)(params=params_info, **kwargs)


class LRUpdater(object):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        strategy: str,
        verbose: bool = False,
        update_freq: int = 1,
        start_epoch: int = 0,
        **kwargs
    ) -> None:
        self.update_freq = max(update_freq, 1)
        self.use_metric = False
        if strategy == "onplateau":
            self.lr_updater = lr_updater.ReduceLROnPlateau(
                optimizer, verbose=verbose, **kwargs
            )
            self.use_metric = True
        elif strategy == "exponential":
            self.lr_updater = lr_updater.ExponentialLR(
                optimizer, verbose=verbose, **kwargs
            )
        else:
            raise NotImplementedError(
                "Just support 'onplateau', 'exponential' for [strategy] argument"
            )
        self.epoch_count = start_epoch

    def step(self, error: float):
        self.epoch_count += 1
        if self.epoch_count % self.update_freq == 0:
            if self.use_metric:
                self.lr_updater.step(error)
            else:
                self.lr_updater.step()
