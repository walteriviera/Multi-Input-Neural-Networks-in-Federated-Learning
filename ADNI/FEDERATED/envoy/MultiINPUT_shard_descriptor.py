# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MultiINPUT Shard Descriptor."""

import logging
import os
from typing import Any, List, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class MultiINPUTShardDataset(ShardDataset):
    """MultiINPUT Shard dataset class."""

    def __init__(self, input_df, data_type: str = 'train', rank: int = 1, worldsize: int = 1) -> None:
        """Initialize MultiINPUTDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.adni_data = input_df.iloc[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an item by the index."""
        return self.adni_data.iloc[index]

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.adni_data)


class MultiINPUTShardDescriptor(ShardDescriptor):
    """MultiINPUT Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            adni_num: str = '',
            data_dir: str = '',
            img_dir: str = '',
            csv_path: str = 'ADNI_csv',
            csv_filename: str = 'ADNI_ready.csv',
            data_seed: str = '0',
            **kwargs

    ) -> None:
        """Initialize MultiINPUTShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.adni_num = adni_num
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.csv_filename = csv_filename
        self.seed = int(data_seed)
        print(f"Seed is {self.seed}, and has type {type(self.seed)}")

        adni_train, adni_val = self.load_data()
        self.data_by_type = {
            'train': adni_train,
            'val': adni_val
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train') -> MultiINPUTShardDataset:
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return MultiINPUTShardDataset(
            self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return ['90', '109', '90']

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return ['1', '1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'MultiINPUT dataset, shard number {self.rank}'
                f' out of {self.worldsize}')


    def load_data(self) -> Tuple[Any, Any, Any]:
        """Download prepared dataset."""

        # Load tabular data from csv file
        csv_file = os.path.join(self.csv_path, self.csv_filename)

        if not csv_file:
            logger.info(f"Dataset {csv_filename} not found at:{csv_path}.\n\t")
            logger.info(f"Aborting.")
            exit

        adni_tabular = pd.read_csv(csv_file)
        adni_tabular = adni_tabular[adni_tabular['SRC']==f"ADNI{self.adni_num}"]

        # Load img paths and details from stored dataframe
        img_df_filename = f"adni{self.adni_num}_paths.pkl"
        img_df_file = os.path.join(self.data_dir, img_df_filename)
        adni_imgs = pd.read_pickle(img_df_file)

        # Combine dataframes adni_tabular and adni_images
        adni = pd.merge( left=adni_imgs, right=adni_tabular, how="inner", on="PTID",
                            suffixes=("_x", "_y"),copy=False, indicator=False, validate="one_to_one")

        # Logging
        logger.info(f"Class distribution is organized as follow:")
        logger.info(f"Final:\n {adni['labels'].value_counts()}")

        # Datasplit
        labels = adni['labels'].tolist()
        train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                              shuffle=True, stratify=labels,
                                              random_state = self.seed)

        adni_train = adni.iloc[train_idx]
        adni_val = adni.iloc[val_idx]
        adni_test = adni_val
        logger.info('MultiINPUT data was loaded!')

        return adni_train, adni_val
