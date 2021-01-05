import torch
import torch.utils.data as data
from typing import Protocol, Any, KeysView, Sized, Iterator


class StrMapProto(Protocol, Sized):
    def __getitem__(self, key: str) -> Any:
        ...

    def keys(self) -> KeysView:
        ...


class _BaseStrSampler(data.Sampler[str]):
    data_source: StrMapProto

    def __init__(self, data_source: StrMapProto) -> None:
        self.data_source = data_source

    def __len__(self) -> int:
        return len(self.data_source)


class SequentialStrSampler(_BaseStrSampler):
    def __iter__(self) -> Iterator[str]:
        return iter(self.data_source.keys())


class RandomStrSampler(_BaseStrSampler):
    def __iter__(self) -> Iterator[str]:
        # I would like the random source to come from pytorch
        keys = list(self.data_source.keys())
        yield from (keys[i] for i in torch.randperm(len(keys)))
