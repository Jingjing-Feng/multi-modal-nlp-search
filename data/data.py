import dataclasses
from typing import Optional
from datetime import datetime

import numpy as np


@dataclasses.dataclass
class Document:
    filename: str
    text: str
    extension: str
    created: datetime
    size: int
    path: str
    embedding: Optional[np.ndarray] = None

    def _get_metadata(self) -> dict:
        """
        Get the metadata of the document.
        """
        return {
            "filename": self.filename,
            "extension": self.extension,
            "created": self.created,
            "size": self.size,
            "path": self.path,
        }

    def to_index_format(self) -> dict:
        """
        Convert the document to the format required by Elasticsearch.
        """
        if self.embedding is None:
            raise ValueError("Embedding is None. Cannot convert to index format.")

        return {
            "filename": self.filename,
            "text": self.text,
            "extension": self.extension,
            "created": self.created,
            # "size": self.size,
            # "path": self.path,
            "embedding": self.embedding.tolist(),
            "metadata": self._get_metadata(),
        }
