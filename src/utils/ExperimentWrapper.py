import torch.utils.data
from src.agents.Agent import BaseAgent
from src.utils.Experiment import Experiment
from torch.utils.data import RandomSampler
import copy


class ExperimentWrapper():
    def createExperiment(self, config : dict, model, agent: BaseAgent, dataset_class, dataset_args: dict, loss_function):
        model.to(config['experiment.device'])
        exp = Experiment(config, dataset_class, dataset_args, model, agent)
        exp.set_loss_function(loss_function)
        return exp