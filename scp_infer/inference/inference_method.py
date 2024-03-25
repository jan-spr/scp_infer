"""class InferenceMethod: template for inference methods"""
from abc import ABC, abstractmethod
import numpy as np
from anndata import AnnData


class InferenceMethod(ABC):
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
    - add save/load data options?

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
            output_dir: str = None,
            verbose: bool = False   # pylint: disable=unused-argument
    ) -> None:
        self.adata_obj = adata_obj
        self.output_dir = output_dir
        self.verbose = verbose

    @abstractmethod
    def convert_data(self):
        """convert adata entries into respective format for algorithm"""
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        save_output: bool = True,
        **kwargs
    ) -> np.array:
        """
        perform inference

        Returns:
        np.array: adjacency matrix
        """
        raise NotImplementedError
