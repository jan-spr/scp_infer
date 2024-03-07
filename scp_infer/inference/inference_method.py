"""class InferenceMethod: template for inference methods"""
import numpy as np
from anndata import AnnData


class InferenceMethod:
    """
    Template for Inference Methods

    Contains placeholders for all the utilities neccesary for implementing inference Methods.

    Wanted functionalities:
    - training for each Algorithm
    - test/val split
        - give to algorithm?
    class Methods:
    - infer (train)
        save trained parameters
        return: Graph / Adjacency Matrix
    - eval (test)
        return: Graph / Adjacency Matrix

    Attributes:
    adata_obj: AnnData
        Annotated expression data object from scanpy
        should be fully preprocessed and ready for inference
    verbose: bool
        default verbosity of the algorithm implementation

    """
    adata_obj: AnnData
    verbose: bool

    def __init__(
            self,
            adata_obj: AnnData,
            verbose: bool = False   # pylint: disable=unused-argument
    ) -> None:
        self.adata_obj = adata_obj

    def convert_data(self):
        """convert adata entries into respective format for algorithm"""
        raise NotImplementedError

    def infer(
        self,
        plot: bool = False,
        **kwargs
    ) -> np.array:
        """
        perform inference

        Returns:
        np.array: adjacency matrix
        """
        raise NotImplementedError
