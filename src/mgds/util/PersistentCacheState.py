import os
import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import torch

from mgds.util.FileUtil import safe_write_torch_file

_CACHE_VERSION_FIELD = '__version'
"""
The special key name for our data version.
"""
_CACHE_UUID_FIELD = '__uuid'
"""
The special key name for our data's Unique ID.
"""

_RESERVED_CACHE_FIELD_KEY_LIST = [_CACHE_VERSION_FIELD, _CACHE_UUID_FIELD]
"""
Cache keys that are reserved by the cache system and are not allowed to be used by callers.
"""

_CACHE_VERSION_INITIAL = 1
"""
Initial cache version.
"""

_CACHE_VERSION_LATEST = _CACHE_VERSION_INITIAL
"""
Our latest cache version.
"""

_CACHE_VERSION_MINIMUM_SUPPORTED = _CACHE_VERSION_INITIAL
"""
The oldest version of a cache that we still support. As cache's are only a best-effort, this version
can be bumped to force a regeneration of cache files in the case where a bug/issue calls for caches
to be regenerated, or even if migrating cache becomes to cumbersome.
"""

class PersistentCacheState:
    """
    A light-weight class to map a particular DiskCache group variation's split-cache and aggregate
    cache data to persistent files on disk. This allows DiskCache to track files across multiple
    runs, allowing for files to be added or removed between runs without the need to regenerate an
    entire group's cache -- only new files need to be cached.
    
    This class works by mapping a unique `persistent_key` string to each index in a group (this is
    set in `PersistentCacheState.build_cache_file_mappings`). Each `persistent_key` is then
    assigned a unique ID called a `persistent_index`, an integer assigned to that key, and only that
    key. `persistent_index` values are generated using an incrementing integer when the
    `PersistentCacheState.build_cache_file_mappings()` function is called. Each `persistent_index`
    which will correspond to a specific file on disk for split files, and a specific index for
    aggregate caches.
    
    Users of this class should create an instance of this class for each cache group variation. Once
    the class has been initialized, `PersistentCacheState.build_cache_file_mappings()` must be
    called which accepts a dict mapping of the local instances `group_index` to their corresponding
    `persistent_key`. Each `group_index` must have a unique and stable `persistent_key` string
    assigned to it at this time. It is important that this value be stable and recoverable for this
    value the next time the application runs, or we will not be able to find previously cached
    values.
    
    Good examples of stable and unique values for `persistent_key`s include: the `filename` for the
    associated index, or a `hash` of the contents of the file of the associated index. Bad examples
    would be the `item_index` or the `group_index` value for the item, as these values are not
    stable, meaning they aren't guaranteed to be the same value for an item across multiple runs of
    an application, especially if files on disk change.
    
    Once the transient cache mapping has been set, the class's updated state should be retrieved by
    calling `PersistentCacheState.to_dict()`, and then that data should be written to disk. To load
    the same data back into the class from the save file, the `PersistentCacheState.from_dict()`
    static method can bee used to create a new class instance. `PersistentCacheState.from_dict()`
    will raise `ValueError` exceptions if the provided data is missing fields, or is invalid in
    other ways. The `PersistentCacheState` support file versioning for forwards compatibility, but
    as caching is best-effort, it is possible that future changes will be incompatible and will
    invalidate the cache by returning an object with mostly default state (if possible).
    
    Once the persistent cache state file has been saved, the following functions will become
    available for calling:
        - `PersistentCacheState.get_aggregate_cache()`: returns the aggregate cache data list for
          this group, with list indices returned in `group_index` order. When the aggregate cache is
          loaded from disk, invalid values will be returned as `None` and are expected to be
          regenerated and resaved by calling applications using the `save_aggregate_cache()`
          function. By default, only "safe" types will be loaded.
        - `PersistentCacheState.save_aggregate_cache()`: saves the aggregate cache data list for
          this group to disk, automatically converting `group_index` order indices to their
          corresponding `persistent_index` values before writing to disk.
    """

    _cache_dir_path: Path
    """
    Base path of our cache directory.
    """

    _aggregate_file_path: Path
    """
    The path to aggregate cache file for this group.
    """

    _split_cache_filename_prefix: str
    """
    An optional prefix used by cache files, allowing multiple cache groups to share a directory
    safely. Must not be entirely numeric, contain slashes, or start with a period. May be empty,
    but not None.
    """

    _split_cache_filename_suffix: str
    """
    The suffix used by cache files, allowing the use of custom file extensions for cache files. Must
    start with a period and must not contain slashes, or start with a period. Must not be empty.
    """

    _next_persistent_index: int
    """
    The next persistent index to assign a new cache item.
    """

    _cache_uuid: UUID
    """
    Unique identifier for this generation of cache. Used to detect when a cache file does not belong
    to the current cache generation.
    """

    _persistent_key_to_persistent_index_map: dict[str, int]
    """
    Dictionary mapping each known `persistent_key` to the corresponding `persistent_index` it is
    mapped to. This allows lookup of the `persistent_index` value for each `group_index` using that
    index's `persistent_key`. These values should be saved to disk using the `to_dict()` function to
    retrieve this class's data.
    """

    _transient_group_index_to_persistent_index_map: dict[int, int]
    """
    Transient (not saved) mapping of the current group's `group_index` indices to their corresponding
    `persistent_index` indices. This dictionary will only contain values after
    `PersistentCacheState.build_cache_file_mappings` has been called.
    """
    _transient_persistent_index_to_group_index_map: dict[int, int]
    """
    Transient (not saved) mapping of the current group's `persistent_index` indices to their
    corresponding `group_index` indices. This dictionary will only contain values after
    `PersistentCacheState.build_cache_file_mappings` has been called.
    """

    def __init__(self,
                 cache_dir: str|os.PathLike,
                 aggregate_filename: str|os.PathLike = 'aggregate.pt',
                 split_cache_filename_prefix: str = '',
                 split_cache_filename_suffix: str = '.pt',
                 persistent_index_starting_value: int = 0,
                 cache_uuid: UUID|None = None,
    ):
        """
        Initialize a new instance of the `PersistentCacheState` class.
        
        :param persistent_index_starting_value: The starting value for assigned `persistent_index`s.
            Each `split_file`'s `persistent_index`, and any set `split_cache_filename_prefix` will
            form the filename for that `split_file`. Default value: `0`.
        :type persistent_index_starting_value: int
        
        :param cache_dir: The root path to this group's cache data. It is possible to share a cache
            directory between multiple groups by utilizing a unique `aggregate_filename` and a 
            unique `split_cache_filename_prefix`.
        :type cache_dir: str | os.PathLike
        
        :param aggregate_filename: The filename for this group's aggregate cache data. Must be unique
            for each group sharing data in `cache_dir`, if any. Default value: `'aggregate.pt'`
        :type aggregate_filename: str | os.PathLike
        
        :param split_cache_filename_prefix: An optional prefix for `split_cache` filenames, to allow
            multiple groups to share a single folder if needed. If set, the prefix must not be
            entirely numeric, must not start with a period, and all characters used must also be
            valid characters for a filename. Not all invalid values are checked or handled, and
            failure to use reasonable values may lead to malfunctions and/or loss of data. Default
            value: `''` (empty string).
        :type split_cache_filename_prefix: str
        
        :param split_cache_filename_suffix: The suffix for `split_cache` filenames. This value must
            start with a period and contain at least one other printable character that is valid for
            use in a filename. Not all invalid values are checked or handled, and failure to use
            reasonable values may lead to malfunctions and/or loss of data. Default value: `'.pt'`. 
        :type split_cache_filename_suffix: str
        
        :param cache_uuid: A unique value for this generation of cache. Prevents us from reading
            stale files that do not belong to this cache generation. If `None`, a unique value will
            be generated for this cache. Default value: `None`. 
        :type cache_uuid: UUID | None
        """

        if cache_dir is None:
            raise ValueError("`cache_dir` may not be None.")
        if aggregate_filename is None:
            raise ValueError("`aggregate_filename` may not be None.")
        if split_cache_filename_prefix is None:
            raise ValueError("`split_cache_filename_prefix` may not be None.")
        if split_cache_filename_suffix is None:
            raise ValueError("`split_cache_filename_suffix` may not be None.")
        if persistent_index_starting_value is None:
            raise ValueError("`persistent_index_starting_value` may not be None.")
        # If `cache_uuid` is `None``, generate a new uuid for it.
        if cache_uuid is None:
            cache_uuid = uuid4()

        self._persistent_key_to_persistent_index_map = dict()
        self._transient_group_index_to_persistent_index_map = dict()
        self._transient_persistent_index_to_group_index_map = dict()
        self.reset(persistent_index_starting_value,
                   cache_dir=cache_dir,
                   aggregate_filename=aggregate_filename,
                   split_cache_filename_prefix=split_cache_filename_prefix,
                   split_cache_filename_suffix=split_cache_filename_suffix,
                   cache_uuid=cache_uuid)

    def to_dict(self) -> dict:
        """
        Creates a dict with our persistent cache state.
        
        :return: The state of our persistent cache. Can be used to reconstruct our persistent state
            when used with the `PersistentCacheState.from_dict()` static method.
        :rtype: dict
        """
        # Write our cache object, rewriting our persistent indexes to be strings for numerical stability
        # and flexibility.
        data = {
            _CACHE_VERSION_FIELD: str(_CACHE_VERSION_LATEST),
            'aggregate_filename': self._aggregate_file_path.name,
            'split_cache_filename_prefix': self._split_cache_filename_prefix,
            'split_cache_filename_suffix': self._split_cache_filename_suffix,
            'next_persistent_index': str(self._next_persistent_index),
            'cache_uuid': self._cache_uuid.hex,
            'persistent_key_map': {
                persistent_key: str(persistent_index)
                for persistent_key, persistent_index in self._persistent_key_to_persistent_index_map.items()
            },
        }

        return data

    @staticmethod
    def from_dict(cache_dir: str|os.PathLike,
                  cache_state_data: dict) -> 'PersistentCacheState':
        """
        Loads the provided `cache_state_data` into our local state. Raises a ValueError if the
        provided data is invalid, malformed, or just too old to be migrated.
        
        :param cache_dir: The path to where our cache files are stored on disk. The directory will
            be created if it does not exist.
        :type cache_dir: str | os.PathLike
        
        :param cache_state_data: A state dictionary from a previous call to `PersistentCacheState.to_dict()`.
        :type cache_state_data: dict
        
        :return: A PersistentCacheState object if the provided input was valid, otherwise a ValueError
            exception will be raised.
        :rtype: PersistentCacheState
        """

        # Check that we have all of our data fields and they're validish.

        ## cache_version ##
        if _CACHE_VERSION_FIELD not in cache_state_data:
            raise ValueError('Invalid disk cache state, missing version field.')

        cache_version = cache_state_data[_CACHE_VERSION_FIELD]
        if not isinstance(cache_version, int):
            try:
                cache_version = int(cache_version)
            except ValueError as ex:
                print(f'Invalid disk cache state, unable to convert version "{cache_version}" into'
                      f' integer due to error: {str(ex)}.')
                raise ex
        if cache_version < _CACHE_VERSION_INITIAL:
            raise ValueError('Invalid disk cache state, version invalid (older than initial'
                             ' version).')
        if cache_version < _CACHE_VERSION_MINIMUM_SUPPORTED:
            raise ValueError('Invalid disk cache state, version unsupported (too old).')
        if cache_version > _CACHE_VERSION_LATEST:
            raise ValueError('Invalid disk cache state, version unsupported (too new).')


        ## aggregate_filename ##
        aggregate_filename_default = 'aggregate.pt'
        aggregate_filename = cache_state_data.get('aggregate_filename', aggregate_filename_default)
        if aggregate_filename is None:
            aggregate_filename = aggregate_filename_default
        aggregate_filename = Path(aggregate_filename).name
        if not isinstance(aggregate_filename, str):
            raise ValueError('Invalid disk cache state, `aggregate_filename` invalid'
                             ' (not None or string).')
        if len(aggregate_filename) == 0:
            raise ValueError('Invalid disk cache state, `aggregate_filename` invalid'
                             ' (empty string).')
        if '.' not in aggregate_filename:
            raise ValueError('Invalid disk cache state, `aggregate_filename` invalid'
                             ' (missing suffix / file extension).')
        if aggregate_filename.startswith('.'):
            raise ValueError('Invalid disk cache state, `aggregate_filename` invalid'
                             ' (starts with period).')
        aggregate_file_path = str(cache_dir / Path(aggregate_filename).name)


        ## split_cache_filename_prefix ##
        split_cache_filename_prefix_default = ''
        split_cache_filename_prefix = cache_state_data.get('split_cache_filename_prefix',
                                                           split_cache_filename_prefix_default)
        if split_cache_filename_prefix is None:
            split_cache_filename_prefix = split_cache_filename_prefix_default
        if not isinstance(split_cache_filename_prefix, str):
            raise ValueError('Invalid disk cache state, `split_cache_filename_prefix` invalid'
                             ' (not string).')
        split_cache_filename_prefix = split_cache_filename_prefix.lstrip()
        if split_cache_filename_prefix.startswith('.'):
            raise ValueError('Invalid disk cache state, `split_cache_filename_prefix` invalid'
                             ' (starts with period).')
        if split_cache_filename_prefix.isdigit():
            raise ValueError('Invalid disk cache state, `split_cache_filename_prefix` invalid'
                             ' (only digits).')
        if not split_cache_filename_prefix.isprintable():
            raise ValueError('Invalid disk cache state, `split_cache_filename_prefix` invalid'
                             ' contains unprintable characters).')
        if '/' in split_cache_filename_prefix or '\\' in split_cache_filename_prefix:
            raise ValueError('Invalid disk cache state, `split_cache_filename_prefix` invalid'
                             ' (must not contain any slashes).')
        # We don't validate everything that won't work here, just (hopefully) the "big" ones that
        # would cause issues, like path traversal or failure to create the file.


        ## split_cache_filename_suffix ##
        split_cache_filename_suffix_default = '.pt'
        split_cache_filename_suffix = cache_state_data.get('split_cache_filename_suffix',
                                                           split_cache_filename_suffix_default)
        if split_cache_filename_suffix is None:
            split_cache_filename_suffix = split_cache_filename_suffix_default
        if not isinstance(split_cache_filename_suffix, str):
            raise ValueError('Invalid disk cache state, `split_cache_filename_suffix` invalid'
                             ' (not string).')
        split_cache_filename_suffix = split_cache_filename_suffix.strip()
        if len(split_cache_filename_suffix) < 2:
            raise ValueError('Invalid disk cache state, `split_cache_filename_suffix` invalid'
                             ' (string not long enough).')
        if not split_cache_filename_suffix.startswith('.'):
            raise ValueError('Invalid disk cache state, `split_cache_filename_suffix` invalid'
                             ' (must start with a period).')
        if '..' in split_cache_filename_suffix:
            raise ValueError('Invalid disk cache state, `split_cache_filename_suffix` invalid'
                             ' string must not contain double-periods.')
        if not split_cache_filename_suffix.isprintable():
            raise ValueError('Invalid disk cache state, `split_cache_filename_suffix` invalid'
                             ' (contains unprintable characters).')
        if '/' in split_cache_filename_suffix or '\\' in split_cache_filename_suffix:
            raise ValueError('Invalid disk cache state, `split_cache_filename_suffix` invalid'
                             ' (must not contain any lashes).')
        # We don't validate everything that won't work here, just (hopefully) the "big" ones that
        # would cause issues, like path traversal or failure to create the file.


        ## next_persistent_index ##
        if 'next_persistent_index' not in cache_state_data:
            raise ValueError('Invalid disk cache state, missing next_persistent_index field.')
        next_persistent_index = cache_state_data['next_persistent_index']
        if not isinstance(next_persistent_index, int):
            try:
                next_persistent_index = int(next_persistent_index)
            except ValueError as ex:
                print(f'Invalid disk cache state, `next_persistent_index` invalid (could not'
                       ' convert "{next_persistent_index}" to integer: {str(ex)}).')
                raise ex
        if next_persistent_index < 0:
            raise ValueError('Invalid disk cache state, `next_persistent_index` invalid'
                             ' (less than zero).')


        ## cache_uuid ##
        if 'cache_uuid' not in cache_state_data:
            cache_uuid = uuid4()
        else:
            try:
                cache_uuid = UUID(hex=cache_state_data['cache_uuid'])
            except ValueError as ex:
                print(f'Invalid disk cache state, `cache_uuid` invalid (could not  convert "'
                      f'{cache_state_data['cache_uuid']}" into hex UUID: {str(ex)}).')
                raise ex


        ## persistent_key_map ##
        if 'persistent_key_map' not in cache_state_data:
            raise ValueError('Invalid disk cache state, missing persistent_key_map field.')
        persistent_key_map = cache_state_data['persistent_key_map']
        if not isinstance(persistent_key_map, dict):
            raise ValueError('Invalid disk cache state, persistent_key_map invalid (not Dict).')
        persistent_key_map = persistent_key_map.copy()
        for persistent_key, persistent_index in persistent_key_map.items():
            if not isinstance(persistent_key, str):
                raise ValueError('Invalid disk cache state, persistent_key invalid (not string).')
            if len(persistent_key) < 1:
                raise ValueError(f'Invalid disk cache state, persistent_key invalid (empty string).')

            if not isinstance(persistent_index, int):
                try:
                    persistent_index = int(persistent_index)
                    persistent_key_map[persistent_key] = persistent_index
                except ValueError as ex:
                    print(f'Failed to convert persistent_index "{persistent_index}" to integer: {str(ex)}.')
                    raise ex

            if persistent_index < 0:
                raise ValueError(f'Invalid disk cache state, persistent_index "{persistent_index}"'
                                 f' is invalid value, "{persistent_index}" is lower than minimum'
                                  ' value of "1".')
            if persistent_index >= next_persistent_index:
                raise ValueError(f'Invalid disk cache state, persistent_index "{persistent_index}"'
                                 f' for "{persistent_key}" is an invalid value, as it is higher than'
                                 f' the current maximum value "{next_persistent_index - 1}.')


        # We have validated everything now, build and return our object
        persistent_cache_state = PersistentCacheState(cache_dir=cache_dir,
                                                      aggregate_filename=aggregate_file_path,
                                                      split_cache_filename_prefix=split_cache_filename_prefix,
                                                      split_cache_filename_suffix=split_cache_filename_suffix,
                                                      persistent_index_starting_value=next_persistent_index,
                                                      cache_uuid=cache_uuid)
        persistent_cache_state._persistent_key_to_persistent_index_map = persistent_key_map

        return persistent_cache_state

    def build_cache_file_mappings(self,
                                  group_index_to_persistent_key_map: dict[int, str],
                                  remove_stale_cache: bool = False):
        """
        Sets and creates the mapping between our local run's `group_index` values to the persistent
        cache's `persistent_index` value using the provided `group_index_to_persistent_key_map`.
        
        NOTE: It is important to include all `group_index` values for this group in this call, as
        all previously-known values, that are now no longer present, will be removed from our lists,
        split files deleted, and need re-caching.
        
        :param group_index_to_persistent_key_map: Mapping of all currently valid `group_index` values
            to their `persistent_key` values. Values no longer present will be assumed to be invalid,
            and will have their data deleted and discarded.
        :type group_index_to_persistent_key_map: dict[int, str]
        
        :param remove_stale_cache: If True, cache split items we are aware of, that no longer exist
            in the current `group_index` list will be removed from disk. The corresponding aggregate
            cache item will also be removed.
            
            If False, we do not delete files that are no longer in our current `group_index` list. If
            these files are re-added at a future point, the existing cache items will be used.
        :type remove_stale_cache: bool
        """

        self._transient_group_index_to_persistent_index_map.clear()
        self._transient_persistent_index_to_group_index_map.clear()

        # Now that we have a list of valid persistent keys, make a list of items that are no longer
        # current so that we can remove the stale items from our cache.
        current_persistent_keys = set(group_index_to_persistent_key_map.values())
        inactive_cache_persistent_keys = [
            persistent_cache_key
            for persistent_cache_key in self._persistent_key_to_persistent_index_map.keys()
            if persistent_cache_key not in current_persistent_keys
        ]

        # Map all of the group-index keys with persistent keys, potentially adding new items to our
        # persistent index
        for group_index, persistent_key in group_index_to_persistent_key_map.items():
            if not isinstance(group_index, int):
                raise ValueError("Invalid cache mapping, expected `group_index` to be an integer.")
            if not isinstance(persistent_key, str):
                raise ValueError("Invalid cache mapping, expected `persistent_key` to be an string.")

            # Support multiple group indexes pointing at the same cache file
            if persistent_key in self._persistent_key_to_persistent_index_map:
                persistent_index = self._persistent_key_to_persistent_index_map[persistent_key]
            else:
                persistent_index = self._next_persistent_index
                self._next_persistent_index += 1

                self._persistent_key_to_persistent_index_map[persistent_key] = persistent_index

            self._transient_group_index_to_persistent_index_map[group_index] = persistent_index
            self._transient_persistent_index_to_group_index_map[persistent_index] = group_index

        # Ensure we have mappings for all of the input data, and that 
        assert len(self._transient_group_index_to_persistent_index_map) == len(group_index_to_persistent_key_map)

        if remove_stale_cache:
            # Remove any cache keys that no longer are active. We do this last so that any invalid data
            # errors above get raised above before we start thinking about deleting files.
            for inactive_persistent_key in inactive_cache_persistent_keys:
                inactive_persistent_index = self._persistent_key_to_persistent_index_map[inactive_persistent_key]
                print(f'Removing inactive persistent key "{inactive_persistent_key}" ({inactive_persistent_index})')

                inactive_cache_file = self._get_split_item_file_path_by_persistent_index(inactive_persistent_index)
                if inactive_cache_file is not None and inactive_cache_file.is_file():
                    inactive_cache_file.unlink(missing_ok=True)

                del self._persistent_key_to_persistent_index_map[inactive_persistent_key]

    def get_aggregate_cache(self,
                            torch_device: torch.device | str | dict[str, str] | None,
                            allow_unsafe_types: bool,
                            validate_against_split_items: bool = True
    ) -> 'list[dict[str, Any]] | None':
        """
        Gets the aggregate cache list for this cache state, automatically removing invalid or expired
        cache data. 
        
        :param torch_device: The torch device to load tensor data onto
        :type torch_device: torch.device | str | dict[str, str] | None
        
        :param allow_unsafe_types: If True, all data types can be deserialized into the returned data.
        
            If False, only safe and allow-listed types known to torch will be deserialized. See
            `torch.serialization.add_safe_globals` for more information.
        :type allow_unsafe_types: bool
        
        :param validate_against_split_items: If True, we will attempt to validate that all valid
            aggregate cache entries have a corresponding split item cache file. If the corresponding
            split item cache file is missing, we will remove any cached data for the aggregate cache
            entry at the same index. Additionally, if an aggregate cache has no data for a specific
            entry, the corresponding split cache file, if present, will be removed.

            If False, we will not check if the split item file exists, and we will not attempt to
            remove it in the case the aggregate file is invalid.
        :type validate_against_split_items: bool
        
        :return: A list of cache data in the same order of the `group_index` values set in the call
            to `PersistentCacheState.build_cache_file_mappings()`.
        :rtype: list[dict[str, Any]] | None
        """

        if not self._aggregate_file_path.is_file():
            return None

        aggregate_data: dict[str, dict] = torch.load(self._aggregate_file_path,
                                                    weights_only=(not allow_unsafe_types),
                                                    map_location=torch_device)

        # Remap legacy aggregate caches to dicts
        if isinstance(aggregate_data, list):
            aggregate_data = {
                index: cache_data
                for index, cache_data in enumerate(aggregate_data)
            }
        if not isinstance(aggregate_data, dict):
            print('Failed to load aggregate cache, data in file not in expected format.')
            return None

        aggregate_cache_version = None
        if _CACHE_VERSION_FIELD in aggregate_data:
            aggregate_cache_version = aggregate_data[_CACHE_VERSION_FIELD]

        if aggregate_cache_version is not None:
            if not isinstance(aggregate_cache_version, int):
                try:
                    aggregate_cache_version = int(aggregate_cache_version)
                except ValueError as ex:
                    print(f'Failed to convert aggregate cache version "{aggregate_cache_version}" to'
                          f' integer: {str(ex)}.')
                    return None

            if aggregate_cache_version < _CACHE_VERSION_INITIAL:
                print(f'Failed to load aggregate cache of unknown version "{aggregate_cache_version}"'
                       ' (cache too old).')
                return None
            elif aggregate_cache_version > _CACHE_VERSION_LATEST:
                print(f'Failed to load aggregate cache of unknown version "{aggregate_cache_version}"'
                       ' (cache too new).')
                return None

            # If this cache item is not the latest version, perform any needed updates in here.
            if aggregate_cache_version < _CACHE_VERSION_LATEST:
                # If it is discovered that a bug (or other issue of importance) requires caches to be
                # regenerated, bump the value of _CACHE_VERSION_MINIMUM_SUPPORTED to the desired minimum
                # version. If a cache is older than this version, we will ignore it and force users
                # to generate a new one.
                if aggregate_cache_version < _CACHE_VERSION_MINIMUM_SUPPORTED:
                    return None

                # Perform any necessary data upgrades/changes at this time, or return `None` if our
                # data cannot, or should not, be migrated to the latest version and should be
                # discarded. Try to avoid letting exceptions pass to the user, as data validation
                # issues usually should just be resolved by generating a new cache.
                pass

        if _CACHE_UUID_FIELD in aggregate_data:
            cache_aggregate_uuid = aggregate_data[_CACHE_UUID_FIELD]
            if cache_aggregate_uuid != self._cache_uuid.hex:
                print('Failed to load aggregate cache, cache data did not match our unique ID:'
                     f' "{self._cache_uuid.hex}" != "{cache_aggregate_uuid}"')
                return None

        # Rebuild our aggregate cache list to use `group_index` keys that callers expect, instead of
        # the `persistent_index` values we save to disk. Additionally, we will discard data when the
        # corresponding split file is missing (or vice versa).
        out_aggregate_data = [None] * len(self._transient_group_index_to_persistent_index_map)
        for group_index, persistent_index in self._transient_group_index_to_persistent_index_map.items():
            if validate_against_split_items:
                # Check if the split data cache for this index still exists
                split_item_file_path = self._get_split_item_file_path_by_persistent_index(persistent_index)
                if split_item_file_path is None or not split_item_file_path.is_file():
                    # Throw away our data for this index if the persistent_index is invalid or the file
                    # no longer exists on disk.
                    continue

            # Ensure if we actually have aggregate data for this index
            persistent_index_str = str(persistent_index)
            if persistent_index_str not in aggregate_data:
                if validate_against_split_items:
                    # Remove the split item for this cache entry if our aggregate data is missing.
                    split_item_file_path.unlink()
                continue

            # It is expected that each aggregate cache index will be the not-`None` dict.
            indexes_cache_data = aggregate_data[persistent_index_str]
            if indexes_cache_data is None or not isinstance(indexes_cache_data, dict):
                if validate_against_split_items:
                    # Remove the split item for this cache entry if our aggregate data is invalid.
                    split_item_file_path.unlink()
                continue

            # Add our aggregate data to the returned object.
            out_aggregate_data[group_index] = indexes_cache_data

        return out_aggregate_data

    def save_aggregate_cache(self, aggregate_cache_data_list: list[dict[str, Any]]):
        """
        Saves `aggregate_cache_data_list` for for this persistent cache to disk.
        `aggregate_cache_data_list` may not contain any keys with two leading underscores.
        
        :param aggregate_cache_data_list: The data for this persistent cache's aggregate cache.
        :type aggregate_cache_data_list: list[dict[str, Any]]
        """
        # Coerce no-data to be an empty list instead
        if aggregate_cache_data_list is None:
            aggregate_cache_data_list = []

        # Ensure our data is a list
        if not isinstance(aggregate_cache_data_list, list):
            raise ValueError("Expected aggregate cache to be a list.")

        if len(aggregate_cache_data_list) > 0:
            if len(self._transient_group_index_to_persistent_index_map) != len(aggregate_cache_data_list):
                raise ValueError("Expected aggregate cache to contain as many items as  list.")

        # Ensure our list of data is either None or a Dict (which may be empty or contain data)
        persistent_aggregate_data = {
            _CACHE_VERSION_FIELD: str(_CACHE_VERSION_LATEST),
            _CACHE_UUID_FIELD: str(self._cache_uuid.hex),
        }
        for group_index, aggregate_cache_data in enumerate(aggregate_cache_data_list):
            if aggregate_cache_data is not None and not isinstance(aggregate_cache_data, dict):
                raise ValueError('Expected aggregate cache data to be a dict or None')

            if group_index not in self._transient_group_index_to_persistent_index_map:
                raise ValueError('Expected aggregate cache data to have persistent_index mapping,'
                                f' "{group_index}" has no mapping.')

            persistent_index = self._transient_group_index_to_persistent_index_map[group_index]

            persistent_aggregate_data[str(persistent_index)] = aggregate_cache_data

        safe_write_torch_file(persistent_aggregate_data, self._aggregate_file_path)

    def get_split_item(self,
                       group_index: int,
                       torch_device: 'torch.device | str | dict[str, str] | None',
                       allow_unsafe_types: bool = False,
    ) -> 'dict[str, Any]|None':
        """
        Get the data for a particular `group_index` if it is valid. Returns `None` if cache file was
        missing or invalid.
        
        :param group_index: The `group_index` to request split item cache data for.
        :type group_index: int
        :param torch_device: The torch device to store Tensor data on
        :type torch_device: 'torch.device | str | dict[str, str] | None'
        :param allow_unsafe_types: If True, allows deserializing of any type, including potentially
            malicious classes or features.
            
            If False, will only deserialize "safe" types, such as `int`, `float`, `str`, `dict`,
            `list`, `torch.Tensor`, and other types allow-listed in Torch.
            
            See `torch.serialization.add_safe_globals` for more information.
        :type allow_unsafe_types: bool
        :return: A dictionary of the cached data for the requested `group_index` if valid data is
            cached, or `None` if there was no data, or if it was invalid.
        :rtype: dict[str, Any] | None
        """

        split_item_file_path = self._get_split_item_file_path_by_group_index(group_index)
        if split_item_file_path is None or not split_item_file_path.is_file():
            return None

        split_item_data = torch.load(split_item_file_path,
                          map_location=torch_device,
                          weights_only=bool(not allow_unsafe_types))
        if split_item_data is None or not isinstance(split_item_data, dict):
            return None

        # Check cache version
        if _CACHE_VERSION_FIELD in split_item_data:
            split_item_cache_version = split_item_data[_CACHE_VERSION_FIELD]

            if not isinstance(split_item_cache_version, int):
                try:
                    split_item_cache_version = int(split_item_cache_version)
                except ValueError as ex:
                    print(f'Failed to convert split item cache version "{split_item_cache_version}"'
                          f' to integer: {str(ex)}.')
                    return None

            if split_item_cache_version < _CACHE_VERSION_INITIAL:
                print(f'Failed to load split item cache of unknown version "{split_item_cache_version}"'
                       ' (cache too old).')
                return None
            elif split_item_cache_version > _CACHE_VERSION_LATEST:
                print(f'Failed to load split item cache of unknown version "{split_item_cache_version}"'
                       ' (cache too new).')
                return None

            # If this cache item is not the latest version, perform any needed updates in here.
            if split_item_cache_version < _CACHE_VERSION_LATEST:
                # If it is discovered that a bug (or other issue of importance) requires caches to be
                # regenerated, bump the value of _CACHE_VERSION_MINIMUM_SUPPORTED to the desired minimum
                # version. If a cache is older than this version, we will ignore it and force users
                # to generate a new one.
                if split_item_cache_version < _CACHE_VERSION_MINIMUM_SUPPORTED:
                    return None

                # Perform any necessary data upgrades/changes at this time, or return `None` if our
                # data cannot, or should not, be migrated to the latest version and should be
                # discarded. Try to avoid letting exceptions pass to the user, as data validation
                # issues usually should just be resolved by generating a new cache.
                pass

            # Do not return the private version field to callers
            del split_item_data[_CACHE_VERSION_FIELD]

        if _CACHE_UUID_FIELD in split_item_data:
            cache_item_uuid = split_item_data[_CACHE_UUID_FIELD]
            if cache_item_uuid != self._cache_uuid.hex:
                print('Split item cache data did not match our Unique ID:'
                     f' "{self._cache_uuid.hex}" != "{cache_item_uuid}"')
                return None

            # Do not return the private UUID field to callers
            del split_item_data[_CACHE_UUID_FIELD]

        return split_item_data
    
    def save_split_item(self,
                        group_index: int,
                        split_item_data: dict[str, Any]):
        """
        Saves `split_item_data` for `group_index` to disk. `split_item_data` may not contain any
        keys with two leading underscores.
        
        A `ValueError` exception is raised if `split_item_data` is invalid or contains any specially
        reserved fields.
        
        :param group_index: The `group_index` to save `split_item_data` for.
        :type group_index: int
        :param split_item_cache_data: The data for this `group_index`'s split item cache.
        :type split_item_cache_data: dict[str, Any]
        """

        ## Validate group_index ##
        if group_index not in self._transient_group_index_to_persistent_index_map:
            raise ValueError('Split-item cache data error: expected group_index to have'
                            f' persistent_index mapping, "{group_index}" has no mapping.')

        split_item_file_path = self._get_split_item_file_path_by_group_index(group_index)
        if split_item_file_path is None:
            raise ValueError('Split-item cache data error: expected group_index to have'
                            f' valid file path, "{group_index}" has none.')

        ## Validate split_item_cache_data ##
        # Coerce no-data to be an empty dict instead
        if split_item_data is None:
            split_item_data = {}

        # Ensure our data is a dict
        if not isinstance(split_item_data, dict):
            raise ValueError('Split-item cache data error: expected split item cache data to be a'
                             ' dict.')

        ## Write our new data array
        split_file_validated_data = {
            _CACHE_VERSION_FIELD: str(_CACHE_VERSION_LATEST),
            _CACHE_UUID_FIELD: str(self._cache_uuid.hex),
        }
        for key, value in split_item_data.items():
            # Ensure our cache keys do not use any of our reserved field names.
            if key in _RESERVED_CACHE_FIELD_KEY_LIST:
                raise ValueError('Split-item cache data error: split item cache data has'
                                f' unexpected reserved field "{key}". This key is not allowed'
                                    ' to be used by cache items.')
            if not isinstance(key, str):
                raise ValueError('Split-item cache data error: split item cache data has'
                                f' unexpected non-string key name. All cache key names must'
                                ' only be strings.')

            split_file_validated_data[key] = value

        safe_write_torch_file(split_file_validated_data, split_item_file_path)

    def reset(self,
              persistent_index_starting_value: int,
              *,
              cache_dir: str|os.PathLike|None = None,
              aggregate_filename: str|os.PathLike|None = None,
              split_cache_filename_prefix: str|None = None,
              split_cache_filename_suffix: str|None = None,
              cache_uuid: UUID|None = None):
        """
        Reset the state of the class, and optionally reinitialize our options to new values.
        
        :param persistent_index_starting_value: The persistent_index_starting_value to reset our
            cache to.
        :type persistent_index_starting_value: `int`
        
        :param cache_dir: The new cache directory to use, or `None` to keep the existing value.
        :type cache_dir: `str | os.PathLike | None`
        
        :param aggregate_filename: The new aggregate filename to use, or `None` to keep the existing
            value.
        :type aggregate_filename: `str | os.PathLike | None`
        
        :param split_cache_filename_prefix: The new split item cache filename prefix to use, or
            `None` to keep the existing value.
        :type split_cache_filename_prefix: `str | None`
        
        :param split_cache_filename_suffix: The new split item cache filename suffix to use, or
            `None` to keep the existing value.
        :type split_cache_filename_suffix: `str | None`
        
        :param cache_uuid: The new cache UUID to use, or `None` to keep the existing value.
        :type cache_uuid: `UUID | None`
        """

        if not isinstance(persistent_index_starting_value, int):
            raise ValueError(f'Invalid `persistent_index_starting_value` "{persistent_index_starting_value}",'
                              ' must be type of integer.')
        if persistent_index_starting_value < 0:
            raise ValueError(f'Invalid `persistent_index_starting_value` "{persistent_index_starting_value}",'
                              ' must be positive integer.')
        if persistent_index_starting_value >= sys.maxsize:
            raise ValueError(f'Invalid `persistent_index_starting_value` "{persistent_index_starting_value}",'
                             f' must be smaller than "{sys.maxsize}".')

        if cache_dir is not None:
            cache_dir = Path(cache_dir).resolve()
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
            if not cache_dir.is_dir():
                raise ValueError(f'cache_dir "{str(cache_dir)}" is not a valid directory.')
  
        if aggregate_filename is not None:
            aggregate_filename = Path(aggregate_filename).name
            if len(aggregate_filename) < 1:
                raise ValueError(f'Invalid `aggregate_filename` "{aggregate_filename}",'
                                ' must be non-empty string.')

        if split_cache_filename_prefix is not None and len(split_cache_filename_prefix) > 0:
            split_cache_filename_prefix = split_cache_filename_prefix.lstrip()
            if split_cache_filename_prefix.startswith('.'):
                raise ValueError(f'Invalid `split_cache_filename_prefix` "{split_cache_filename_prefix}",'
                                ' string must not start with a period.')
            if split_cache_filename_prefix.isdigit():
                raise ValueError(f'Invalid `split_cache_filename_prefix` "{split_cache_filename_prefix}",'
                                ' string must not be entirely numerical.')
            if not split_cache_filename_prefix.isprintable():
                raise ValueError(f'Invalid `split_cache_filename_prefix` "{split_cache_filename_prefix}",'
                                ' string must not be entirely printable.')
            if '/' in split_cache_filename_prefix or '\\' in split_cache_filename_prefix:
                raise ValueError(f'Invalid `split_cache_filename_prefix` "{split_cache_filename_prefix}",'
                                ' string must not contain slashes.')
            # We don't validate everything that won't work here, just (hopefully) the "big" ones that
            # would cause issues, like path traversal or failure to create the file.

        if split_cache_filename_suffix is not None:
            split_cache_filename_suffix = split_cache_filename_suffix.strip()
            if len(split_cache_filename_suffix) < 2:
                raise ValueError(f'Invalid `split_cache_filename_suffix` "{split_cache_filename_suffix}",'
                                  ' string must contain a valid file extension.')
            if not split_cache_filename_suffix.startswith('.'):
                raise ValueError(f'Invalid `split_cache_filename_suffix` "{split_cache_filename_suffix}",'
                                  ' string must start with a period.')
            if '..' in split_cache_filename_suffix:
                raise ValueError(f'Invalid `split_cache_filename_suffix` "{split_cache_filename_suffix}",'
                                  ' string must not contain double-periods.')
            if not split_cache_filename_suffix.isprintable():
                raise ValueError(f'Invalid `split_cache_filename_suffix` "{split_cache_filename_suffix}",'
                                  ' string must not be entirely printable.')
            if '/' in split_cache_filename_suffix or '\\' in split_cache_filename_suffix:
                raise ValueError(f'Invalid `split_cache_filename_suffix` "{split_cache_filename_suffix}",'
                                  ' string must not contain slashes.')
            # We don't validate everything that won't work here, just (hopefully) the "big" ones that
            # would cause issues, like path traversal or failure to create the file.

        if cache_uuid is not None:
            cache_uuid_hex = cache_uuid.hex
            try:
                cache_uuid = UUID(hex=cache_uuid_hex)
            except ValueError as ex:
                print(f'Invalid `cache_uuid` "{cache_uuid_hex}", error: {str(ex)}.')
                raise ex

        # Write our updated values:
        self._next_persistent_index = persistent_index_starting_value
        if cache_dir is not None:
            self._cache_dir_path = cache_dir
        if aggregate_filename is not None:
            self._aggregate_file_path = self._cache_dir_path / Path(aggregate_filename).name
        if split_cache_filename_prefix is not None:
            self._split_cache_filename_prefix = split_cache_filename_prefix
        if split_cache_filename_suffix is not None:
            self._split_cache_filename_suffix = split_cache_filename_suffix
        if cache_uuid is not None:
            self._cache_uuid = cache_uuid

        self._persistent_key_to_persistent_index_map.clear()
        self._transient_group_index_to_persistent_index_map.clear()
        self._transient_persistent_index_to_group_index_map.clear()

        assert 0 <= self._next_persistent_index <= sys.maxsize
        assert self._cache_dir_path.is_dir() or not self._cache_dir_path.exists()
        assert isinstance(self._aggregate_file_path, Path)
        assert isinstance(self._split_cache_filename_prefix, str)
        assert isinstance(self._split_cache_filename_suffix, str)
        assert len(self._split_cache_filename_suffix) > 1
        assert len(self._cache_uuid.hex) > 1

    @staticmethod
    def _clone_for_cache_if_tensor(cache_item: Any) -> Any:
        """
        If the provided `cache_item` is a tensor, it will be cloned and returned, otherwise the
        original object is returned.
        """
        return (cache_item.clone()
                if isinstance(cache_item, torch.Tensor) else
                cache_item)

    def _get_split_item_file_path_by_group_index(self, group_index: int) -> Path|None:
        """
        Get the file path for a particular `group_index` if it is valid, whether it exists or not,
        or None if the `group_index` is out of our range.
        """
        group_index = int(group_index)
        if group_index < 0 or group_index not in self._transient_group_index_to_persistent_index_map:
            return None

        persistent_index = self._transient_group_index_to_persistent_index_map[group_index]
        return self._get_split_item_file_path_by_persistent_index(persistent_index)

    def _get_split_item_file_path_by_persistent_index(self, persistent_index: int) -> Path|None:
        """
        Get the file path for a particular `persistent_index` if it is valid, whether it exists or
        not, or `None` if the `persistent_index` is out of our range.
        """
        persistent_index = int(persistent_index)
        if persistent_index < 0 or persistent_index >= self._next_persistent_index:
            return None

        return self._cache_dir_path / Path(
                f'{self._split_cache_filename_prefix}{str(persistent_index)}'
            ).with_suffix(self._split_cache_filename_suffix)
