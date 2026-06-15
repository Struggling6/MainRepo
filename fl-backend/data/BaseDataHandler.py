import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.preprocessing import StandardScaler
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
    _log_prefix  = "[DATA]"  # override in subclass for dataset-specific log tags

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
            train_ratio=1.0 - self.config.data.test_split,
            gap_hours=self._gap_hours,
            window_size=self._window_size,
            stride=self._stride,
            target=self.config.data.target,
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
            target=self.config.data.target,
        )

    # ------------------------------------------------------------------ #
    #  Shared windowed-array helpers                                       #
    #  Used by every CSV/window-based handler so they all scale, balance,  #
    #  and batch data identically. Subclasses only set self.batch_size,    #
    #  self.seed, and (optionally) self.normalize, plus _log_prefix.       #
    # ------------------------------------------------------------------ #

    def _scale_temporal_arrays(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize windowed arrays using training-set statistics only.

        Honours self.normalize when present (defaults to True for handlers
        that always scale, e.g. LEAD).
        """
        if not getattr(self, "normalize", True):
            return X_train.astype(np.float32), X_val.astype(np.float32)

        scaler = StandardScaler()
        train_shape = X_train.shape
        val_shape = X_val.shape

        X_train_scaled = scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(train_shape)
        X_val_scaled = scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(val_shape)

        print(f"{self._log_prefix} Applied StandardScaler using training data only")

        return (
            X_train_scaled.astype(np.float32),
            X_val_scaled.astype(np.float32),
        )

    def _build_dataloaders_from_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        return trainloader, valloader

    def _undersample_normals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normal_to_anomaly_ratio: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y = y.astype(np.int64)

        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        n_anomalies = len(anomaly_idx)

        if n_anomalies == 0:
            print(f"{self._log_prefix} WARNING: No anomalies found; skipping undersampling")
            return X, y

        n_normals_to_keep = int(n_anomalies * normal_to_anomaly_ratio)

        if len(normal_idx) <= n_normals_to_keep:
            print(
                f"{self._log_prefix} WARNING: Not enough normal samples for "
                "undersampling; keeping original data"
            )
            return X, y

        sampled_normal_idx = rng.choice(
            normal_idx,
            size=n_normals_to_keep,
            replace=False,
        )
        selected_idx = np.concatenate([anomaly_idx, sampled_normal_idx])
        rng.shuffle(selected_idx)

        X_under = X[selected_idx]
        y_under = y[selected_idx]

        print(
            f"{self._log_prefix} Applied undersampling: "
            f"ratio={normal_to_anomaly_ratio}:1, "
            f"before={len(y)}, after={len(y_under)}, "
            f"anomaly_rate={y_under.mean():.4f}"
        )

        return X_under, y_under

    def _oversample_anomalies(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        method: str,
        target_ratio: float,
        smote_k_neighbors: int,
        seed: int,
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = y.astype(np.int64, copy=False)

        if method in [None, "none"]:
            return X, y

        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if positives == 0 or negatives == 0:
            print(
                f"{self._log_prefix} WARNING: Oversampling skipped for {split_name}; "
                "requires both classes."
            )
            return X, y

        current_ratio = positives / negatives

        if target_ratio <= current_ratio:
            print(
                f"{self._log_prefix} Oversampling skipped for {split_name}: "
                f"current positive:negative ratio={current_ratio:.4f} "
                f"is already >= target={target_ratio:.4f}"
            )
            return X, y

        original_shape = X.shape
        X_flat = X.reshape(original_shape[0], -1)

        if method == "random_over":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(
                sampling_strategy=target_ratio,
                random_state=seed,
            )

        elif method == "smote":
            if positives < 2:
                print(
                    f"{self._log_prefix} WARNING: SMOTE skipped for {split_name}; "
                    "requires at least two positive samples."
                )
                return X, y

            from imblearn.over_sampling import SMOTE

            effective_k = min(smote_k_neighbors, positives - 1)
            sampler = SMOTE(
                sampling_strategy=target_ratio,
                random_state=seed,
                k_neighbors=effective_k,
            )

        elif method in ["borderline_smote", "borderlinesmote"]:
            if positives < 2:
                print(
                    f"{self._log_prefix} WARNING: BorderlineSMOTE skipped for "
                    f"{split_name}; requires at least two positive samples."
                )
                return X, y

            from imblearn.over_sampling import BorderlineSMOTE

            effective_k = min(smote_k_neighbors, positives - 1)
            sampler = BorderlineSMOTE(
                sampling_strategy=target_ratio,
                random_state=seed,
                k_neighbors=effective_k,
                m_neighbors=min(10, max(1, len(y) - 1)),
                kind="borderline-1",
            )

        elif method in ["time_series_augment", "ts_augment"]:
            rng = np.random.default_rng(seed)

            anomaly_idx = np.where(y == 1)[0]
            normal_idx = np.where(y == 0)[0]

            target_positives = int(target_ratio * len(normal_idx))
            n_to_generate = max(0, target_positives - len(anomaly_idx))

            if n_to_generate <= 0:
                print(
                    f"{self._log_prefix} Time-series augmentation skipped for "
                    f"{split_name}; target ratio already reached."
                )
                return X, y

            source_idx = rng.choice(
                anomaly_idx,
                size=n_to_generate,
                replace=True,
            )

            X_new = X[source_idx].copy()

            # 1. Jittering: small Gaussian noise
            noise_std = 0.02
            X_new = X_new + rng.normal(
                loc=0.0,
                scale=noise_std,
                size=X_new.shape,
            ).astype(np.float32)

            # 2. Magnitude scaling: slightly scale each window
            scale = rng.normal(
                loc=1.0,
                scale=0.05,
                size=(n_to_generate, 1, 1),
            ).astype(np.float32)
            X_new = X_new * scale

            # 3. Time shift: roll the sequence slightly forward/backward
            max_shift = 3
            for i in range(n_to_generate):
                shift = rng.integers(-max_shift, max_shift + 1)
                X_new[i] = np.roll(X_new[i], shift=shift, axis=0)

            y_new = np.ones(n_to_generate, dtype=np.int64)

            X_resampled = np.concatenate([X, X_new], axis=0)
            y_resampled = np.concatenate([y, y_new], axis=0)

            shuffle_idx = rng.permutation(len(y_resampled))
            X_resampled = X_resampled[shuffle_idx].astype(np.float32, copy=False)
            y_resampled = y_resampled[shuffle_idx].astype(np.int64, copy=False)

            print(
                f"{self._log_prefix} Applied {split_name} time-series augmentation: "
                f"target_pos_neg={target_ratio}:1, "
                f"before={len(y)}, pos_before={positives}, "
                f"generated={n_to_generate}, "
                f"after={len(y_resampled)}, pos_after={int(y_resampled.sum())}, "
                f"anomaly_rate={float(y_resampled.mean()):.4f}"
            )

            return X_resampled, y_resampled

        else:
            raise ValueError(
                f"Unknown oversampling_method='{method}'. Expected 'none', "
                "'random_over', 'smote', 'borderline_smote', or 'time_series_augment'."
            )

        X_resampled, y_resampled = sampler.fit_resample(X_flat, y)

        X_resampled = X_resampled.reshape((-1, *original_shape[1:])).astype(
            np.float32,
            copy=False,
        )
        y_resampled = y_resampled.astype(np.int64, copy=False)

        print(
            f"{self._log_prefix} Applied {split_name} oversampling: "
            f"method={method}, "
            f"target_pos_neg={target_ratio}:1, "
            f"before={len(y)}, pos_before={positives}, "
            f"after={len(y_resampled)}, pos_after={int(y_resampled.sum())}, "
            f"anomaly_rate={float(y_resampled.mean()):.4f}"
        )

        return X_resampled, y_resampled
