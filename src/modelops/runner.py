from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

__all__ = ['TrainConfig', 'ModelRunner']


M = TypeVar('M', bound=torch.nn.Module)     # Model
X = TypeVar('X')                            # Input
Y = TypeVar('Y')                            # Output


@dataclass(frozen=True)
class TrainConfig(Generic[Y]):
    optimizer: Optimizer
    loss: Callable[[Y, Y], torch.Tensor]
    metrics: Mapping[str, Callable[[Y, Y], float]]

    num_epochs: int
    writer: SummaryWriter
    checkpoint_dir: Path

@dataclass
class TrainResult:
    loss: float
    metrics: Mapping[str, float]

    epoch: int
    mode: Literal['train', 'valid']

@dataclass
class EvalConfig(Generic[Y]):
    metrics: Mapping[str, Callable[[Y, Y], float]]

@dataclass
class EvalResult:
    metrics: Mapping[str, float]


class ModelRunner(Generic[M, X, Y], ABC):
    def __init__(self, model: M) -> None:
        super().__init__()

        self.model = model

    def train(
        self,
        *,
        train_data: Iterable[tuple[X, Y]],
        val_data: Iterable[tuple[X, Y]],
        config: TrainConfig[Y],
    ) -> Iterable[TrainResult]:
        print('[Model Training]')

        self._before_train(config=config)

        try:
            history: list[TrainResult] = []

            for epoch in range(1, config.num_epochs + 1):
                self._before_epoch_train(epoch=epoch, history=history, config=config)

                try:
                    train_result = self._run_epoch(
                        data=train_data,
                        epoch=epoch,
                        config=config,
                        mode='train',
                    )
                    history.append(train_result)
                    yield train_result
                finally:
                    self._after_epoch_train(epoch=epoch, history=history, config=config)

                val_result = self._run_epoch(
                    data=val_data,
                    epoch=epoch,
                    config=config,
                    mode='valid',
                )
                history.append(val_result)
                yield val_result
        finally:
            self._after_train(config=config)

    def _before_train(
        self,
        *,
        config: TrainConfig[Y],
    ) -> None:
        pass

    def _after_train(
        self,
        *,
        config: TrainConfig[Y],
    ) -> None:
       pass

    def _before_epoch_train(
        self,
        *,
        epoch: int,
        history: Sequence[TrainResult],
        config: TrainConfig[Y],
    ) -> None:
       pass

    def _run_epoch(
        self,
        *,
        data: Iterable[tuple[X, Y]],
        epoch: int,
        config: TrainConfig[Y],
        mode: Literal['train', 'valid'],
    ) -> TrainResult:
        running_loss = 0.0
        running_metrics = {name: 0.0 for name in config.metrics.keys()}
        running_count = 0

        prev_mode = self.model.training
        self.model.train(mode == 'train')

        try:
            progbar = tqdm(data, unit='batch', desc=f'Epoch {epoch}/{config.num_epochs} ({mode})')
            for x, y_true in progbar:
                if mode == 'train':
                    config.optimizer.zero_grad()

                y_pred = self._forward(x)

                loss = config.loss(y_true, y_pred)

                if mode == 'train':
                    loss.backward()
                    config.optimizer.step()

                running_loss += loss.item()

                for name, metric in config.metrics.items():
                    running_metrics[name] += metric(y_true, y_pred)

                running_count += 1

                progbar.set_postfix(dict(
                    loss=running_loss / running_count,
                    **{name: metric / running_count for name, metric in running_metrics.items()},
                ))

            result = TrainResult(
                loss=running_loss / running_count,
                metrics={name: metric / running_count for name, metric in running_metrics.items()},
                epoch=epoch,
                mode=mode,
            )

            config.writer.add_scalars(
                mode,
                asdict(result),
                global_step=epoch,
            )
            config.writer.flush()
        finally:
            self.model.train(prev_mode)

        return result

    @abstractmethod
    def _forward(
        self,
        x: X,
    ) -> Y:
        raise NotImplementedError

    def _after_epoch_train(
        self,
        *,
        epoch: int,
        history: Sequence[TrainResult],
        config: TrainConfig[Y],
    ) -> None:
       train_results = [result for result in history if result.mode == 'train']
       valid_results = [result for result in history if result.mode == 'valid']

       valid_losses = [result.loss for result in valid_results]
       if valid_losses[-1] == min(valid_losses):
           torch.save(self.model.state_dict(), config.checkpoint_dir / f'epoch_{epoch}')

    @torch.no_grad()
    def eval(
        self,
        *,
        data: Iterable[tuple[X, Y]],
        config: EvalConfig[Y],
    ) -> EvalResult:
        print('[Model Evaluation]')

        running_metrics = {name: 0.0 for name in config.metrics.keys()}
        running_count = 0

        progbar = tqdm(data, unit='batch')
        for x, y_true in progbar:
            y_pred = self._forward(x)

            for name, metric in config.metrics.items():
                running_metrics[name] += metric(y_true, y_pred)

            running_count += 1

            progbar.set_postfix(dict(
                **{name: metric / running_count for name, metric in running_metrics.items()},
            ))

        return EvalResult(
            metrics={name: metric / running_count for name, metric in running_metrics.items()},
        )

    @torch.no_grad()
    def infer(
        self,
        *,
        data: Iterable[X],
    ) -> Iterable[Y]:
        print('[Model Inference]')

        for batch in tqdm(data, unit='batch'):
            yield self._forward(batch)
