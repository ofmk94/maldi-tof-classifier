from dataclasses import dataclass
from numpy.typing import NDArray

import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FileLocationConfig:
    """
    Configuration class for measurement file location.

    Defines the file type and the regex for extracting the spot names
    from filepaths.
    """
    file_type: str = ".txt"
    spot_pattern: str = r"_([A-Z][0-9]+)_"
    undefined_spot_label = "n/a"

@dataclass(frozen=True)
class FileParsingConfig:
    """
    Configuration class for parsing measurement files.

    Defines indices of the mz and intensity column, comment marker, the value
    separation symbol and metadata rows to skip.
    """
    mz_col_idx: int = 0
    intensity_col_idx: int = 1
    comment: str = "#"
    sep: str = r"\s+"
    # Row #5 is an uncommented metainfo in Shimadzu 8300 measurement files.
    skiprows: tuple[int, ...] = (5,)


class FullSpectraExtractor:
    """
    Encapsulates the logic for extracting data from measurements produced by
    a Shimadzu 8300 MALDI TOF mass spectrometer.

    It locates all measurement files and allows to extract the raw spectra,
    associated spot and class_label and the m/z axis shared across
    measurements.
    """
    def __init__(
        self,
        file_location: FileLocationConfig = FileLocationConfig(),
        file_parsing: FileParsingConfig = FileParsingConfig(),
        use_mz_cutoff: bool = False,
        mz_cutoff_mass: float | None = 20_000.0
    ) -> None:
        """
        Initializes the configuration variables for file location and parsing.

        Parameters
        -----------
        file_location
            Configuration variables for file location (e.g. filetype
            spot regex pattern)
        file_parsing
            Configuration variables for parsing measurement files (e.g.
            column indices, comment marker)
        use_mz_cutoff
            Defines whether the intensity-values are cut off after a certain
            mass-value.
        mz_cutoff_mass
            The mass-value at which the intensity-values are cut off.
        """
        self.file_location = file_location
        self.file_parsing = file_parsing
        self.use_mz_cutoff = use_mz_cutoff
        self.mz_cutoff_mass = mz_cutoff_mass


    def _locate_train_files(self, root_dir: Path) -> tuple[list[Path], list[str], list[str]]:
        """
        Locates all measurement files in the root directory and finds the
        filepaths, class labels (parent directory name) and spot (extracted
        from filename via regex).

        Parameters
        -----------
        root_dir:
            Base directory for initializing file location.

        Returns
        --------
        filepaths
            The filepaths to all measurement files discovered.
        spots
            The corresponding spots to all measurement files.
        class_labels
            The class labels (parent directory names) to all measurement files.
        """

        filepaths = []
        class_labels = []
        spots = []

        for class_dir in sorted(
                    root_dir.glob("*")
                ):
                    if class_dir.is_dir():
                        label = class_dir.name
                        for file in sorted(
                            class_dir.glob(f"*{self.file_location.file_type}")
                        ):
                            filepaths.append(file)
                            class_labels.append(label)

                            res = re.search(self.file_location.spot_pattern, file.name)
                            if res is not None:
                                spots.append(res.group(1))
                            else:
                                spots.append(self.file_location.undefined_spot_label)

        if not filepaths:
            raise FileNotFoundError(
                f"""
                No measurement files of the specified criteria found in {root_dir}
                """
            )

        return filepaths, class_labels, spots


    def _locate_predict_files(self, root_dir: Path) -> list[Path]:
        """
        Retrieves the filepaths of all files in root_dir matching the filetype
        defined in the file location config parameters.

        Parameters
        -----------
        root_dir
            Base directory for initializing file location.

        Returns
        --------
        Filepaths encountered in the root directory.
        """
        filepaths = sorted(root_dir.glob(f"*{self.file_location.file_type}"))

        if not filepaths:
            raise FileNotFoundError(
                f"""
                No measurement files of the specified criteria found in {root_dir}
                """
            )

        return filepaths

    def _read_file_col(self, filepath: Path, col_idx: int
                       ) -> NDArray[np.floating]:
        """
        Returns column with specified column index from file.

        Parameters
        -----------
        filepath:
            Filepath of measurement file.
        col_idx:
            Index of column to be read from file.

        Returns
        --------
        The values from the file and column specified.
        """
        vals = pd.read_csv(
            filepath,
            header=None,
            usecols=[col_idx],
            comment=self.file_parsing.comment,
            sep=self.file_parsing.sep,
            skiprows=list(self.file_parsing.skiprows)
        ).values.flatten()

        return vals

    def _extract_mz_axis(self, filepath: Path) -> NDArray[np.floating]:
        """
        Extracts mz_axis from measurement file. Assumes same mz axis is shared
        by all measurements.

        Parameters
        -----------
        filepath
            Filepath to a measurement file.

        Returns
        --------
        mz axis of the measurement file.
        """
        return self._read_file_col(filepath, self.file_parsing.mz_col_idx)

    def _extract_spectrum(self, filepath: Path) -> NDArray[np.floating]:
        """
        Extracts the spectrum (intensity values) from a measurement file up
        to the m/z-axis cutoff index, if specified.

        Returns
        --------
        Spectrum extracted from file.
        """
        spectrum = self._read_file_col(
            filepath,
            self.file_parsing.intensity_col_idx
        )

        if self.use_mz_cutoff:
            if (not hasattr(self, "mz_cutoff_idx_")):
                raise ValueError(
                    """
                    No m/z-axis cutoff index available!
                    First, initialize the m/z-axis cutoff by extracting the
                    training data by calling extractor.extract_train_data!
                    """
                )
            spectrum = spectrum[:self.mz_cutoff_idx_]

        return spectrum


    def extract_train_data(self, root_dir: Path
                           ) -> tuple[
                                    NDArray[np.floating],
                                    list[str], list[str],
                                    NDArray[np.floating]
                               ]:
        """
        Extracts spectra (intensity values), class labels, spots and the sets
        the m/z-axis cutoff index as an instance variable for spectra returned.

        Parameters
        -----------
        root_dir:
            Base directory for initializing file location.

        Returns
        --------
        spectra
            A 2D np.array where the rows are the spectra from the measurement
            files.
        class_labels
            The class labels (parent directory names) to all measurement files.
        spots
            The corresponding spots to all measurement files.
        """

        filepaths, class_labels, spots = self._locate_train_files(root_dir)

        # Set m/z-axis cutoff index.
        mz_axis = self._extract_mz_axis(filepaths[0])
        if self.use_mz_cutoff:
            if self.mz_cutoff_mass > mz_axis.max():
                raise ValueError(
                    """
                    m/z-axis cutoff-mass can not be larger than the m/z-axis
                    limit!
                    """
                )
            self.mz_cutoff_idx_ = np.searchsorted(mz_axis, self.mz_cutoff_mass,
                                                  side="left")

        spectra = []
        for fp in tqdm(filepaths, desc="Extracting training spectra"):
            spectrum = self._extract_spectrum(fp)
            spectra.append(spectrum)

        spectra = np.vstack(spectra)

        return spectra, class_labels, spots

    def extract_predict_data(self, root_dir: Path
                             ) -> tuple[NDArray[np.floating], list[str]]:
        """
        Extracts spectra (intensity values) for all test measurement files.

        Parameters
        -----------
        root_dir
            Base directory for initializing file location.

        Returns
        --------
        spectra
            A 2D np.array where the rows are the spectra from the measurement
            files.
        filenames
            Names of all files found in root_dir.
        """
        filenames = []
        spectra = []
        filepaths = self._locate_predict_files(root_dir)
        for file in tqdm(filepaths, desc="Extracting prediction spectra"):
            filenames.append(file.name)
            spectrum = self._extract_spectrum(file)
            spectra.append(spectrum)

        spectra = np.vstack(spectra)

        return spectra, filenames
