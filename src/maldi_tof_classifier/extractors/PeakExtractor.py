from dataclasses import dataclass

import numpy as np

import re
from pathlib import Path

from numpy.typing import NDArray
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects.methods import RS4
from rpy2.robjects import FloatVector, r

from rpy2.robjects.packages import importr

from rpy2.robjects.methods import RS4
from rpy2.robjects import FloatVector, ListVector

MALDIquant = importr("MALDIquant")


@dataclass(frozen=True)
class FileLocationConfig:
    """
    Configuration class for measurement file location.

    Defines the file type and the regex for extracting the spot names
    from filepaths.
    """
    file_type: str = ".csv"
    spot_pattern: str = r"_([A-Z][0-9]+)_"
    undefined_spot_label = "n/a"

@dataclass(frozen=True)
class FileParsingConfig:
    """
    Configuration class for parsing measurement files.

    Defines the names of the columns with the mass, intensity value, and  S/N
    ratio for the peak data to be extracted.
    """
    mass_label: str = "Mass (Da)"
    intensity_label: str = "Intensity (mV)"
    snr_label: str = "S / N"

class PeakExtractor(object):
    """
    Encapsulates the logic for extracting peak data from measurements produced
    by a Shimadzu 8030 MALDI TOF mass spectrometer.

    It locates all measurement files containing peak data and allows to extract
    the desired features from it.
    """
    def __init__(
        self,
        file_location: FileLocationConfig = FileLocationConfig(),
        file_parsing: FileParsingConfig = FileParsingConfig(),
        snr_thresh: float = 3.0,
        rel_shift_tolerance: float = 0.002,
        min_peak_freq: float = 0.25
    ) -> None:
        """
        Initializes the configuration variables for file location and parsing.

        Parameters
        -----------
        file_location
            Configuration variables for file location (e.g. filetype,
            spot regex pattern).
        file_parsing
            Configuration variables for parsing measurement files (column names,
            S/N ratio threshold).
        snr_thresh
            Signal to noise ratio threshold for keeping a peak in the data extracted.
        rel_shift_tolerance
            Relative peak shift tolerance for generating the intensity matrix
            from training peak data and for mapping new peak data on consensus
            masses from training. A relative shift tolerance of 0.002 means a
            tolerance of e.g. +-20 Da at m/z 10,000.
        min_peak_freq
            Minimum required frequency of a peak in training data to be
            considered for the intensity matrix generation.
        """
        self.file_location = file_location
        self.file_parsing = file_parsing
        self.snr_thresh = snr_thresh
        self.rel_shift_tolerance = rel_shift_tolerance
        self.min_peak_freq = min_peak_freq

    def _locate_train_files(self, root_dir: Path
                            ) -> tuple[list[Path], list[str], list[str]]:
        """
        Locates all measurement files in the root directory and finds the
        filepaths, class labels (parent directory name) and spots (extracted
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

    def _read_peaks(self, filepath: Path)-> pd.DataFrame:
        """
        Reads a peak measurement file and extracts required features of peaks with
        minimum S/N ratio.

        Parameters
        -----------
        filepath
            Filepath of measurement files.
        Returns
        --------
        A dataframe with the required features of all peaks fulfilling min. S/N ratio.
        """
        return (
            pd.read_csv(filepath)
            .loc[lambda df: df[self.file_parsing.snr_label] >= self.snr_thresh]
            [[self.file_parsing.mass_label, self.file_parsing.intensity_label,
              self.file_parsing.snr_label]]
            )

    def _df_to_mass_peaks(self, peaks_df: pd.DataFrame) -> RS4:
        """
        Converts a pandas dataframe with peak data to a MALDIquant mass peaks object.

        Parameters
        -----------
        peaks_df
            A dataframe containing peak data.

        Returns
        --------
        A MALDIquant mass peaks object containing the same peak data.
        """
        return MALDIquant.createMassPeaks(
            mass = FloatVector(peaks_df[self.file_parsing.mass_label].tolist()),
            intensity = FloatVector(peaks_df[self.file_parsing.intensity_label]
                                    .tolist()),
            snr = FloatVector(peaks_df[self.file_parsing.snr_label].tolist())
        )

    def _generate_intensity_matrix(self, mass_peaks: ListVector
                                   ) -> NDArray[np.floating]:
        """
        Find the consensus masses and generates the intensity matrix for the
        peaks data in mass_peaks. Stores the consensus masses as a object
        instance variable.

        Parameters
        -----------
        mass_peaks
            List of MALDI quant mass peaks objects containing peak data.

        Returns
        --------
        The intensity matrix based on the consensus peak masses.
        """
        binned = MALDIquant.binPeaks(mass_peaks, tolerance=self.rel_shift_tolerance)

        filtered = MALDIquant.filterPeaks(binned, minFrequency=self.min_peak_freq)

        matrix = MALDIquant.intensityMatrix(filtered)

        intensities = np.array(matrix, dtype=float)
        intensities[np.isnan(intensities)] = 0.0

        self.consensus_masses_ = np.array(r["attr"](matrix, "mass"), dtype=float)

        return intensities

    def _map_peaks_to_reference(self, peaks_df: pd.DataFrame) -> NDArray[np.floating]:
        """
        Maps a new instance of peak data onto the consensus masseses.

        Parameters
        -----------
        peaks_df
            A df containing new peak data

        Returns
        --------
        A vector containing the peak intensities of the data passed, mapped
        onto the consensus peak masses.
        """
        mapping = np.zeros(len(self.consensus_masses_))

        sample_masses = peaks_df[self.file_parsing.mass_label].to_numpy()
        sample_intensities = peaks_df [self.file_parsing.intensity_label].to_numpy()
        # Array to keep track which sample masses have been matched already.
        matched_sample_masses = np.zeros(len(sample_masses), dtype=bool)

        for i, consensus_mass in enumerate(self.consensus_masses_):
            relative_differences = np.abs(sample_masses - consensus_mass)/consensus_mass
            # Indices of all sample masses that could match mass and have not been
            # matched yet.
            match_idx = np.where((relative_differences <= self.rel_shift_tolerance) &
                                 (~matched_sample_masses))[0]

            # If at least one sample mass matches, take the closest one.
            if len(match_idx) > 0 :
                best_fit_idx = match_idx[np.argmin(relative_differences[match_idx])]
                mapping[i] = sample_intensities[best_fit_idx]
                matched_sample_masses[best_fit_idx] = True

        return mapping


    def extract_train_data(self, root_dir: Path) -> tuple[
                                                        NDArray[np.floating],
                                                        list[str],
                                                    ]:
        """
        Extracts the dataframes and corresponding class labels for all training
        peak data files in root_dir.

        Parameters
        -----------
        root_dir
            Base directory for extracting the peak data files.

        Returns
        --------
        A list of dataframes with exatracted peak data and corresponding class
        labels.
        """
        filepaths, class_labels, _ = self._locate_train_files(root_dir)
        peaks_dfs = [self._read_peaks(fp) for fp in filepaths]

        return peaks_dfs, class_labels

    def extract_predict_data(self, root_dir: Path) -> NDArray[np.floating]:
        """
        Extracts the dataframes for all peak data files to be classifie in
        root_dir.

        Parameters
        -----------
        root_dir
            Base directory for extracting the peak data files.

        Returns
        --------
        peak_dfs
            A list of dataframes with exatracted peak data.
        filenames
            Names of all files found in root_dir.
        """
        filepaths = self._locate_predict_files(root_dir)

        filenames = [fp.name for fp in filepaths]
        peak_dfs = [self._read_peaks(fp) for fp in filepaths]

        return peak_dfs, filenames

    def transform_train_data(self, peaks_dfs: list[pd.DataFrame]
                             ) -> NDArray[np.floating]:
        """
        Builds the consensus peak intensity matrix based on the peak data
        extracted for training.

        Parameters
        -----------
        peaks_dfs
            List of dataframes containing the training peak data.

        Returns
        --------
        The consensus peak intensity matrix computed on the training peak data.
        """
        mass_peaks = [self._df_to_mass_peaks(df) for df in peaks_dfs]
        return self._generate_intensity_matrix(mass_peaks)

    def transform_predict_data(self, peaks_dfs: list[pd.DataFrame]
                               ) -> NDArray[np.floating]:
        """
        Transforms a list of dataframes with peak data onto a matrix of rows
        with intensity values mapped onto the consensus peak masses.

        Parameters
        -----------
        peaks_dfs
            List of dataframes containing the training peak data.

        Returns
        --------
        A matrix of row vectors with intensity values mapped to the consensus
        peak masses.
        """
        if (not hasattr(self, "consensus_masses_")):
            raise ValueError(
                """
                No consensus masses available to map new peak data onto!
                First, generate the intensity matrix and consensus masses by
                calling extractor.transform_train_data!
                """
            )

        return (
            np.vstack([self._map_peaks_to_reference(df)
                       for df in peaks_dfs])
        )



