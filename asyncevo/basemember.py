__all__ = ['BaseMember']


from abc import ABC, abstractmethod


class BaseMember(ABC):
    @abstractmethod
    def appropriate_lineage(self, lineage):
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
