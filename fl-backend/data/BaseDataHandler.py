import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from .time_series_utils import temporal_grouped_split, create_test_windows

#this is supposed to be an interface for all the data handlers
# so each Data handler has these methods implemented, 
# and the rest of the code can call these methods without worrying about the implementation details of each data handler
class BaseDatasetHandler(ABC):
    """
    Abstract base class for all dataset handlers.
    Defines the interface every handler must implement, and provides
    shared implementations of run_split and load_test_set using the
    Template Method pattern.

    The Template Method pattern means the base class defines the
    structure of an algorithm (run_split, load_test_set), while
    subclasses fill in the dataset-specific details (_preprocess,
    _prepare_data). This guarantees that every handler applies
    identical preprocessing to both training and test data, since
    both paths call the same _preprocess method.

    Subclass responsibilities:
      - _prepare_data  : load, clean, feature-engineer, populate self.df
                         and self.feature_cols
      - _preprocess    : apply dataset-specific cleaning to any DataFrame
      - get_dataloaders: return train/val DataLoaders for a given partition
      - get_metadata   : return dict with input_dim, num_classes etc.
      - get_num_partitions: return number of clients/partitions
    """

    # ------------------------------------------------------------------ #
    #  Dataset-specific constants — override in subclass if needed         #
    # ------------------------------------------------------------------ #

    _node_col    = "building_id"
    _time_col    = "timestamp"
    _window_size = 168   # 1 week of hourly data
    _stride      = 168    # one window per day
    _gap_hours   = 73     # override if lag features require a gap

    def __init__(self, config):
        self.config       = config
        self.df           = None  # populated by _prepare_data
        self.feature_cols = None  # populated by _prepare_data

    # ------------------------------------------------------------------ #
    #  Abstract methods — must be implemented by every subclass            #
    # ------------------------------------------------------------------ #
   
    @abstractmethod
    def _prepare_data(self):
        """
        Load, clean and feature-engineer the dataset.
        Must populate self.df and self.feature_cols before returning
        X, y as numpy arrays.
        """
        pass

    @abstractmethod
    def _preprocess(self, df):
        """
        Apply dataset-specific preprocessing to a raw DataFrame.
        Called by both _prepare_data (training) and load_test_set
        (testing) so both paths use identical feature engineering.
        """
        pass

    @abstractmethod
    def get_dataloaders(self, partition_id: int):
        """
        Return train and validation DataLoaders for the given client
        partition. Flower calls this during client initialisation.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Return a dictionary of dataset metadata used by the model
        config's build() method to construct the correct architecture.
        Must include at least:
          - input_dim  : number of features per timestep — passed to
                         build(input_dim=...) as the model's in_channels
          - num_classes: number of output classes
          - num_samples: total number of raw rows after preprocessing
          - task_type  : e.g. "binary_classification"
          - data_format: e.g. "tabular"
        """
        pass

    @abstractmethod
    def get_num_partitions(self) -> int:
        """Return the number of client partitions."""
        pass

    # ------------------------------------------------------------------ #
    #  Shared implementations — available to all subclasses                #
    # ------------------------------------------------------------------ #

    def run_split(self):
        """
        Perform a temporal train/val split on the processed DataFrame.
        Uses self.df and self.feature_cols populated by _prepare_data.
        Split parameters are read from self.config.
        
        Returns
        -------
        X_train, y_train, X_val, y_val : np.ndarray
        """
        return temporal_grouped_split(
            self.df,
            feature_cols=self.feature_cols,
            node_col=self._node_col,
            time_col=self._time_col,
            train_ratio=1.0 - self.config.test_split,
            gap_hours=self._gap_hours,
            window_size=self._window_size,
            stride=self._stride,
            target=self.config.target,
        )

    def load_test_set(self, test_path: Path):
        """
        Load and preprocess a held-out test CSV, then window it without
        any further splitting. Uses the same feature columns computed
        during _prepare_data to guarantee consistency with training data.
        """
        
        test_df = self._preprocess(pd.read_csv(test_path))
        return create_test_windows(
            test_df,
            feature_cols=self.feature_cols,
            window_size=self._window_size,
            stride=self._stride,
            node_col=self._node_col,
            time_col=self._time_col,
            target=self.config.target,
        )
