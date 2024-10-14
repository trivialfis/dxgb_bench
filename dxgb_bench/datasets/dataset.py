import argparse
import os
from typing import Any, Dict, Optional, Tuple
from urllib.request import urlretrieve

from ..utils import DType, fprint, show_progress


class DataSet:
    uri: Optional[str] = None

    def retrieve(self, local_directory: str) -> str:
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
        assert self.uri
        filename = os.path.join(local_directory, os.path.basename(self.uri))
        if not os.path.exists(filename):
            fprint(
                "Retrieving from {uri} to {filename}".format(
                    uri=self.uri, filename=filename
                )
            )
            urlretrieve(self.uri, filename, show_progress)
        return filename

    def extra_args(self) -> Dict[str, Any]:
        return {}

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        raise NotImplementedError()
