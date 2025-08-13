from typing import List
import torchio as tio
from .Experiment import Experiment

class Study():
    r"""This class handles:
            - Running multiple experiments at once
    """

    def __init__(self, study_config : dict = {}) -> None:
        self.experiments: List[Experiment] = []

    def add_experiment(self, experiment: Experiment) -> None:
        r"""Add an experiment to the study
            #Args
                experiment (Experiment): Experiment to add
        """
        self.experiments.append(experiment)

    def run_experiments(self) -> None:
        r"""Run all experiments
        """
        for experiment in self.experiments:
            experiment.agent.train(experiment.data_loaders['train'], experiment.loss_function)
            

    def eval_experiments(self, ood_augmentation: tio.Transform=None, output_name: str=None, export_prediction: bool=False, pseudo_ensemble=True) -> None:
        r"""Eval all experiments
        """
        for experiment in self.experiments:
            experiment.agent.getAverageDiceScore(pseudo_ensemble=pseudo_ensemble, ood_augmentation=ood_augmentation, output_name=output_name,
                                                 export_prediction=export_prediction)
            
    
    def eval_experiments_ood(self) -> None:
        for severity in range(1, 6):
            augmentations = []
            augmentations.append(tio.RandomGhosting(num_ghosts=severity, intensity=0.5 * severity))
            augmentations.append(tio.RandomAnisotropy(downsampling=severity))
            augmentations.append(tio.RandomBiasField(coefficients=0.1*severity))
            augmentations.append(tio.RandomNoise(std=0.1*severity))
            augmentations.append(tio.RandomBlur(std=0.5*severity))

            for ood_augmentation in augmentations:
                output_name = f"{ood_augmentation.__class__.__name__}_{severity}.csv"
                self.eval_experiments(ood_augmentation=ood_augmentation, output_name=output_name)