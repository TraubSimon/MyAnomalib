"""Helper fnctions for downloading dataets wit progress bars and hash verification.

This module provides utilities for:
- Showing progress bars during downloads with ``urlretrieve``
- Safely extracting compressed files
"""
import os
import sys
import re
import io
import logging
import hashlib
from pathlib import Path
from collections.abc import Iterable
from tqdm import tqdm
from dataclasses import dataclass
from urllib.request import urlretrieve
from zipfile import ZipFile
from tarfile import TarFile, TarInfo

logger = logging.getLogger(__name__)

@dataclass
class DownloadInfo:
    """Information needed to download a dataset from a URL
    
    Args:
        name (str): Name of the dataset
        url (str): URL to download the dataset from
        hashsum (str): Expected hash value of the downloaded file
        filename (str): Optional filenam to save as. If not provided, extracts 
            from URL 
        
    """
    name: str
    url: str 
    hashsum: str
    filename: str | None = None 

class DownloadProgressBar(tqdm):
    """Progress bar tor ``urlretrieve`` downloads.
    
    Subclasses ``tqdm`` to rovide a progress bar during file downloads.

    Example: 
        >>> url = "https://example.com/file.zip"
        >>> output_path = "file.zip"
        >>> with DownloadProgressBar(
                unit='B', 
                unit_scale=True, 
                miniters=1, 
                desc=url.split('/')[-1]
            ) as p_bar:
                urlretrieve(url, filename=output_path, reporthook=p_bar.update_to)
    Args: 
        iterable (Iterable | None): Iterable to decorate with progressbar
            Defaults to ``None``
        desc (str | None): Prefix for the progressbar
            Defaults to ``None``
        total (int |float): Expected number of iterations
            Defaults to ``None``
        leave (bool| None): Whether to leave the progressbar after completion
            Defaults to ``True``
        files (io.TextIOWrapper | io.StringIO | None): Output stram for progress messages
            Defaults to ``None``
        ncol (int | None): width of the progress bar
            Defaults to ``None``
        mininterval (float | None): Minimum update intervals in seconds
            Defaults to ``0.1``
        maxinterval (float | None): Maximum update interval in seconds
            Defaults to 10.0
        miniters (int | float): Minimum progress display update interval in iterations
            Defaults to ``None``
        use_asci (bool | None): Whteher to use ASCII characters for the progressbar
            Defaults to ``None``
        disable (bool | None): Whether to disable the progressbar
            Defaults to ``None``
        unit (str | None): unit of measurement
            Defaults to ``it``
        unit_scale (bool | int | float): Whether to scale unites automatically
            Defaults to ``False``
        dynamic_ncols (bool | None): Whether to adapt the terminal resizes
            Defaults to ``False``
        smoothing (float | None): Exponential moving average smoothing factor
            Defaults to ``0.3``
        bar_format (str | None): Custom progressbar format string
            Defaults to ``None``
        initial (int | float | None): Initial counter value
            Defaults to ``0``
        position(int | None): Line offset for printing
            Defaults to ``None``
        postfix (dict | None): Additional stats to display
            Defaults to ``None``
        unit_divisor (float | None): Unit divisor for scaling
            Defaults to ``1000``
        write_bytes(bool | None): Whether to write bytes
            Defaults to ``None``
        lock_args (tuple | None): Arguments passed to refresh
            Defaults to ``None``
        nrow (int | None): Screen height
            Defaults to ``None``
        color (str | None): Bar color
            Defaults to ``None``
        delay (float | Noen): Display delay in seconds
            Defaults to ``0``
        gui (bool | None): Whether to use matplotlib animations
            Defaults to ``False``
    """
    def __init__(
        self, 
        iterable: Iterable | None,
        desc: str | None = None,
        total: int | float | None = None,
        leave: bool | None = True,
        file: io.TextIOWrapper | io.StringIO | None = None,
        ncols: int | None = None,
        mininterval: float | None = 0.1,
        maxinterval: float | None = 10.0,
        miniters: int | float | None = None,
        use_ascii: bool | str | None = None,
        disable: bool | None = False,
        unit: str | None = "it",
        unit_scale: bool | int | float | None = False,
        dynamic_ncols: bool | None = False,
        smoothing: float | None = 0.3,
        bar_format: str | None = None,
        initial: int | float | None = 0,
        position: int | None = None,
        postfix: dict | None = None,
        unit_divisor: float | None = 1000,
        write_bytes: bool | None = None,
        lock_args: tuple | None = None,
        nrows: int | None = None,
        colour: str | None = None,
        delay: float | None = 0,
        gui: bool | None = False,
        **kwargs,
    ) -> None:
        super().__init__(
            iterable=iterable, 
            desc=desc, 
            total=total, 
            leave=leave, 
            file=file,
            ncols=ncols,
            mininterval=mininterval, 
            maxinterval=maxinterval, 
            miniters=mininterval, 
            ascii=ascii, 
            disable=disable, 
            unit=unit, 
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, 
            bar_format=bar_format,
            initial=initial, 
            position=position, 
            postfix=postfix, 
            unit_divisor=unit_divisor,
            write_bytes=write_bytes, 
            lock_args=lock_args, 
            nrows=nrows, 
            colour=colour, 
            delay=delay,
            gui=gui, 
            **kwargs,
        )
        self.total: int | float | None 

    def update_to(
        self, 
        chunk_number: int = 1,
        max_chunk_size: int = 1,
        total_size: int | None = None
    ) -> None:
        """Update progressbar based on download progress
        
        This method is used as a callback for ``urlretrieve`` to update
        the progress bar during downloads.

        Args: 
            chunk_number (int): Current chunk being processed.
                Defauls to ``1``
            max_chunk_size (int): Maximum size of each chunk
                Defaults to ``1``
            total_size: Total download size
        """
        if total_size is not None:
            self.total = total_size
        self.update(chunk_number * max_chunk_size - self.n)

def is_file_potentially_dangerous(file_name: str) -> bool:
    """Chec if a file path contains potentially dnerous patterns.
    
    Args: 
        file_path (str): Path to check
        
    Returns: 
        ``True`` if the path matches unsafe patterns, ``False`` otherwise"""
    # Some example criteria. We could expand this
    unsafe_patterns = ["/etc/", "/root/"]
    return any(re.search(pattern, file_name) for pattern in unsafe_patterns)
    
def safe_extract(tar_file: TarFile, root: Path, members: list[TarInfo]) ->None:
    """Safely extracts members from a tar archive
    
    Args: 
        tar_file (TarFile): TarFile obkect to extract from
        root (Path): Root directory for extraction
        members: List of safe members to extract
    """
    for member in members:
        # check if the file already exists
        if not (root / member.name).exists():
            if sys.version_info[:3] > (3, 11, 4):
                # filter argument only works with python>=3.11.4
                tar_file.extract(member, root, filter="data")
            else:
                tar_file.extract(member, root)

def generate_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Generate a has of a file using the specified algorithm
    
    Args: 
        file_path (str | None): Path to the file to hash
        algorithm (str): Hashing algorithm to use (e.g. ``"sha256"`` , ``"sha3_512"``
            Defaults to ``"sha_256"``
    
    Returns: 
        Hexadecimal has string of the file

    Raises: 
        ValueError: If the specified hashing algorithm is not supported
    """
    # get the hashing algorithm
    try:
        hasher = getattr(hashlib, algorithm)()
    except AttributeError as err:
        msg = f"Unsupported hashing algorithm: {algorithm}"
        raise ValueError(msg) from err

    # Read the file in chunks to avoid loading it all into memory
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)

    # Return the computed hash value in hexadecimal format
    return hasher.hexadigets() 

def check_hash(file_path: Path, expected_hash: str, algorithm: str = "sha256") -> None:
    """Veify that a file's hash matches the expected value.
    
    Args: 
        file_path (Path): Path to file to check
        expected_hash (str): Expected hash value
        algorithm (str): Hashing algorithm to use
            Defaults to ``"sha256"``
    Raises: 
        ValueError: If the calculated has does not match the expected hash
    """
    # Compare the calculated ash with the expected hash
    calculated_hash = generate_hash(file_path, algorithm)
    if calculated_hash != expected_hash:
        msg = (
            f"Calculated has {calculated_hash} of downloaded file {file_path} "
            f"does not mach the required has {expected_hash}"
        )
        raise ValueError(msg)

def extract(file_name: Path, root: Path) -> None:
    """Extracts a compressed dataset file.
    
    Supports .zip, .tar, .gz, .xz and .tgz formats
    
    Args: 
        file_name (Path): Path of the file to extract
        root (Path): Root directory for extraction

    Raises: 
        ValueError: If the file format is not recognized    
    """
    logger.info(f"Extracting dataset into {root} folder.")

    # safely extract zip files
    if file_name.suffix == ".zip":
        with ZipFile(file_name, "r") as zip_file:
            for file_info in zip_file.infolist():
                if not is_file_potentially_dangerous(file_info.filename):
                    zip_file.extract(file_info, root)

    #safely extract tar files
    elif file_name.suffix in {".tar", ".gz", ".xz", ".tgz"}:
        with TarFile.open(file_name) as tar_file:
            members = tar_file.getmembers()
            safe_members = [member for member in members if not is_file_potentially_dangerous(member.name)]
            safe_extract(tar_file, root, safe_members)
    else:
        msg = f"Unrecognized file format {file_name}"
        raise ValueError(msg)
    logger.info("Cleaning up files.")
    file_name.unlink()
    

def download_and_extract(root: Path, info: DownloadInfo) -> None:
    """Dowload and extract a dataset.
    
    Args: 
        root (Path): Root directory where the dataset will be stored
        info: (DownloadInfo): Download information for the dataset

    Raises: 
        RuntimeError: If the URL scheme is not http(s)
    """

    root.mkdir(parents=True, exist_ok=True)

    # save the compresed file in the spcified root directory
    downloaded_file_path = root / info.filename if info.filename else root / info.url.split("/")[-1]

    if downloaded_file_path.exists():
        logger.info("Existing dataset archive found. Skipping download stage.")
    else:
        logger.info("Downloading the %s dataset.", info.name)
        # audit url. allowing only http:// and https://
        if info.url.startswith("http://") or info.url.startswith("https://"):
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=info.name) as progress_bar:
                urlretrieve( #noqa: S310 # nosec B31ß
                    url=f"{info.url}",
                    filename=downloaded_file_path, 
                    reprthook=progress_bar.update_to,
                )
            logger.info("Checking hash of the dowloadd file.")
            check_hash(downloaded_file_path, info.hashsum)
        else:
            msg = f"Invalid URL to download dataset. Supports only 'http://' or 'https://' but {info.url} is requested"
            raise RuntimeError(msg)
        
    extract(downloaded_file_path, root)

def is_within_directory(directory: Path, target: Path) -> bool:
    """Check if a target path is located within a given directory.
    
    Args:
        directory: Path of the parent directory
        target: Path to check
    
    Returns: 
        ``True`` if target is within directory, ``False`` otherwise
    """

    abs_directory = directory.resolve()
    abs_target = target.resolve() 

    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == str(abs_directory)



