from typing import MutableMapping, Hashable, List, Iterable, Optional
from itertools import chain
from collections import defaultdict


class Vocab:
    """This vocab can hold any hashable value, so imma say this contains
    `elements` instead of narrowing it down to string.
    """

    etoi: MutableMapping[Hashable, int]
    itoe: List[Hashable]
    unk_token: Optional[Hashable]

    def __init__(
        self, el_source: Iterable[Hashable], unk_token: Optional[Hashable] = None
    ):
        if unk_token:
            self.unk_token = unk_token
            self.etoi = defaultdict(int) 
        else:
            self.etoi = {}
        self.itoe = []
        it = chain([unk_token], el_source) if unk_token else el_source
        for i, el in enumerate(it):
            self.etoi[el] = i
            self.itoe.append(el)

    @property
    def unk_index(self):
        try:
            return self.etoi[self.unk_token]
        except AttributeError:
            raise AttributeError("This instance doesn't define an unk_token") from None

    def __contains__(self, e):
        return e in self.etoi

    def __iter__(self):
        return iter(self.etoi)

    def __len__(self):
        return len(self.etoi)


class EmbeddedVocab(Vocab):
    pass


class GloVe(EmbeddedVocab):
    pass
