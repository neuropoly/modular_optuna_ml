import os

from copy import copy as shallow_copy
from logging import Logger
from typing import Optional, Self

from config.utils import default_as, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import DataHook


@registered_data_hook("dump")
class DumpHook(DataHook):
    """
    A hook which will dump the data. Useful when you want to save data after encoding, imputation, etc.
    The output file format is determined by the output_dest file extension (either 'tsv' or 'csv').

    Example usage:
            {
                "type": "dump",
                "output_dest": ".output_dumped.tsv"
            },

    Note: depending on the position of the hook in data_config.json, the data may be in a different state. For example,
    if you want to dump the data after encoding, make sure the hook is placed after the encoding hook(s).
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.output_dest = parse_data_config_entry(
            "output_dest", config,
            default_as('./dumped.tsv', self.logger)
        )
        output_folder = '/'.join(self.output_dest.split('/')[:-1])  # strip the filename
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Extract the output file suffix (tsv or csv); raise an error if it's not one of the two
        self.output_type = self.output_dest.split('.')[-1]
        if self.output_type not in ['tsv', 'csv']:
            raise ValueError(f"Invalid output type '{self.output_type}' for DumpHook! "
                             f"Please make sure the output type is one of the following: ['tsv', 'csv']")

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self,
            x: BaseDataManager,
            y: Optional[BaseDataManager] = None
        ) -> BaseDataManager:
        # Dump the results to a tabular output
        df_out = shallow_copy(x.data)
        df_out[y.data.columns[0]] = y.data.iloc[:, 0]
        # Save the dataframe, depending on the output type
        if self.output_type == 'tsv':
            df_out.to_csv(self.output_dest, sep='\t')
        elif self.output_type == 'csv':
            df_out.to_csv(self.output_dest, sep=',')

        # Return the result, unchanged
        return x
