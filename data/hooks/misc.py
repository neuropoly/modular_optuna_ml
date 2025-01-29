from logging import Logger
from typing import Optional, Self

from config.utils import default_as, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import DataHook
from data.mixins import MultiFeatureMixin


@registered_data_hook("dump")
class DumpHook(DataHook):
    """
    A hook which will dump the data. Useful when you want to save data after encoding, imputation, etc.

    Example usage:
            {
                "type": "dump",
                "output_dest": ".output_dumped.tsv"
            },
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.output_dest = parse_data_config_entry(
            "output_dest", config,
            default_as('./dumped.tsv', self.logger)
        )

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self,
            x: BaseDataManager,
            y: Optional[BaseDataManager] = None
        ) -> BaseDataManager:
        # We can only dump tabular datasets currently
        if isinstance(x, MultiFeatureMixin):
            df_out = x.data
            df_out['target'] = y.as_array()
            df_out.to_csv(self.output_dest, sep='\t')
        # Return the result, unchanged
        return x
