import json
import os
from typing import Any
from pathlib import Path

import torch

def safe_write_torch_file(data: Any,
                          save_file_path: str|os.PathLike):
    """
    Helper function to save torch files. Utilizes a temporary file before moving the temporary file
    into the final specified location, ensuring that files are completely written before they are
    moved into the `save_file_path` location. Attempts to create any missing parent directories
    before writing our file.

    If successful, will overwrite any file data that already exists.

    If a file fails to be written to, due to being canceled or another type of failure, the original
    file will be unmodified.

    Any raised exceptions are left to callers to handle; no exceptions are caught by this function.

    :param data: The data to save to the torch file specified in `save_file_path`.
    :type data: Any
    
    :param save_file_path: The file path to save the torch file to.
    :type save_file_path: str | os.PathLike
    """
    save_file_path = Path(save_file_path)

    if len(save_file_path.stem) < 1:
        raise ValueError(f'Invalid filename for file path "{str(save_file_path)}".')
    if len(save_file_path.suffix) < 2:
        raise ValueError(f'Invalid file extension for file path "{str(save_file_path)}".')

    # Attempt to make our parent directories if they don't already exist.
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to tmp file first to mitigate risks of partial-writes.
    save_tmp_file_path = save_file_path.with_suffix(f'{save_file_path.suffix}_tmp')

    torch.save(data, save_tmp_file_path)

    # Rename our tmp file and replace any existing file in its final location
    save_tmp_file_path.replace(save_file_path)

def safe_write_json_file(data: Any,
                         json_file_path: str|os.PathLike):
    """
    Helper function to save json files. Utilizes a temporary file before moving the temporary file
    into the final specified location, ensuring that files are completely written before they are
    moved into the `save_file_path` location. Attempts to create any missing parent directories
    before writing our file.

    If successful, will overwrite any file data that already exists.

    If a file fails to be written to, due to being canceled or another type of failure, the original
    file will be unmodified.

    Any raised exceptions are left to callers to handle; no exceptions are caught by this function.

    :param data: The data to save to the json file specified in `save_file_path`.
    :type data: Any
    
    :param save_file_path: The file path to save the torch file to.
    :type save_file_path: str | os.PathLike
    """
    json_file_path = Path(json_file_path)

    if len(json_file_path.stem) < 1:
        raise ValueError(f'Invalid filename for file path "{str(json_file_path)}".')
    if len(json_file_path.suffix) < 2:
        raise ValueError(f'Invalid file extension for file path "{str(json_file_path)}".')

    # Attempt to make our parent directories if they don't already exist.
    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to tmp file first to mitigate risks of partial-writes.
    json_temp_file_path = json_file_path.with_suffix(f'{json_file_path.suffix}_tmp')

    # Write our data to the file
    with open(json_temp_file_path, 'w') as json_temp_file_stream:
        json.dump(data, json_temp_file_stream)

    # Rename our tmp file and replace any existing file in its final location
    json_temp_file_path.replace(json_file_path)

