from abc import ABCMeta, abstractmethod
from typing import Any


class SerialPipelineModule(
    metaclass=ABCMeta,
):
    def __init__(self):
        super(SerialPipelineModule, self).__init__()
        self.current_variation = -1

    def start(self, variation: int):
        """
        Called once before each variation, starting with the first variation.

        :param variation: the variation that is started
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
    def get_item(self, index: int, requested_name: str = None) -> dict:
        """
        Called to return an item or partial item from this module.
        If `requested_name` is None, the entire item should be returned.
        If `requested_name` is a string, only the specified key needs to be returned,
        but the whole item can be returned if it improves performance to return everything at once.

        :param index: the item index to return
        :param requested_name: the requested item key
        :return: an item or partial item
        """
        pass
