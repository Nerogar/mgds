from abc import ABCMeta, abstractmethod
from typing import Any


class RandomAccessPipelineModule(
    metaclass=ABCMeta,
):
    def __init__(self):
        super(RandomAccessPipelineModule, self).__init__()
        self.started = False

    @abstractmethod
    def length(self) -> int:
        """
        Returns the number of items this module can return.
        """
        pass

    def start(self, variation: int):
        """
        Called once when the pipeline is started.

        :param variation: the variation that is started
        """
        pass

    def get_meta(self, variation: int, name: str) -> Any:
        """
        Called to return meta information about this module.

        :param variation: the variation index to return
        :param name: the requested meta key
        :return: meta information
        """
        return None

    @abstractmethod
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        """
        Called to return an item or partial item from this module.
        If `requested_name` is None, the entire item should be returned.
        If `requested_name` is a string, only the specified key needs to be returned,
        but the whole item can be returned if it improves performance to return everything at once.

        :param variation: the variation index to return
        :param index: the item index to return
        :param requested_name: the requested item key
        :return: an item or partial item
        """
        pass
