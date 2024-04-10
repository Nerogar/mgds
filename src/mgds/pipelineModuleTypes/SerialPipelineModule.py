from abc import ABCMeta, abstractmethod
from typing import Any


class SerialPipelineModule(
    metaclass=ABCMeta,
):
    def __init__(self):
        super(SerialPipelineModule, self).__init__()
        self.current_variation = -1
        self.current_index = -1

    @abstractmethod
    def approximate_length(self) -> int:
        """
        Returns the approximate number of items this module can return.
        The number may not be exact if the exact number of items is not known.
        """
        pass

    def start(self, variation: int, start_index: int):
        """
        Called once before each variation, starting with the first variation.

        :param variation: the variation that is started
        :param start_index: the index to start iteration from
        """
        pass

    def get_meta(self, name: str) -> Any:
        """
        Called to return meta information about this module.

        :param name: the requested meta key
        :return: meta information
        """
        return None

    @abstractmethod
    def get_next_item(self) -> dict:
        """
        Called to return the next item from this module.

        :return: an item
        """
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """
        Returns True if this module can return at least one more item.
        """
        pass
