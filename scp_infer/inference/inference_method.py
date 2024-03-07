"""class InferenceMethod: template for inference methods"""
import numpy as np


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

    """

    def __init__(self, adata_obj) -> None:
        pass

    def convert_data(self):
        """convert adata entries into respective format for algorithm"""

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
