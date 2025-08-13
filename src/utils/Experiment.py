#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from src.agents.Agent import BaseAgent

import json
import os
import random
from filelock import Timeout
import torch
from src.datasets.BatchgeneratorDatagenerator import DatasetPerEpochGenerator, StepsPerEpochGenerator
from src.utils.MyMultiThreadedAugmenter import MyMultiThreadedAugmenter
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.nn as nn

from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from aim import Run, Image, Figure, Distribution
import numpy as np
from PIL import Image as PILImage
import git
from matplotlib import figure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
from src.utils.DataAugmentations import get_transform_arr
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
import pickle as pkl

class Experiment():
    r"""This class handles:
            - Interactions with the experiment folder
            - Loading / Saving experiments
            - Datasets
    """
    def __init__(self, config: dict, dataset_class, dataset_args: dict, model: nn.Module, agent) -> None:
        # Backward compatibility
        if isinstance(config, list):
            config = config[0]

        self.config = config
        self.add_required_to_config()
        self.dataset_class = dataset_class
        self.dataset_args = dataset_args
        self.model = model
        self.agent = agent
        self.storage = {}
        self.model_state = "train"
        self.general_preload()
        if(os.path.isdir(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'models'))):
            self.reload()
        else:
            self.setup()
            # Load pretrained model
            if 'pretrained' in self.config and self.currentStep == 0:
                self.load_model()

        self.general_postload()

        print("\n-------- Experiment Setup --------")
        print(json.dumps(self.config, indent=4))
        print("-------- Experiment Setup --------\n")


        #self.initializeFID()
        self.currentStep = self.currentStep+1
        #self.set_current_config()

    def add_required_to_config(self) -> None:
        r"""Fills config with basic setup if not defined otherwise
        """
        # Basic Configs
        if 'experiment.model_path' not in self.config:
            self.config['experiment.model_path'] = os.path.join(pc.STUDY_PATH, 
                                                                       'Experiments', self.config['experiment.name'] + "_" + 
                                                                       self.config['experiment.description'])
        # Git hash
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.config['experiment.git_hash'] = sha
        
    def set_loss_function(self, loss_function) -> None:
        self.loss_function = loss_function

    def setup(self) -> None:
        r"""Initial experiment setup when first started
        """
        # Create dirs
        os.makedirs(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'models'), exist_ok=True)
        # Create Run
        self.run = Run(experiment=self.config['experiment.name'], repo=os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'Aim'))
        self.run.description = self.config['experiment.description']
        self.config['experiment.run_hash'] = self.run.hash
        self.run['hparams'] = self.config

        # Create basic configuration
        self.data_split = self.new_datasplit()
        self.set_model_state("train")
        self.data_split.save_to_file(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'data_split.pkl'))
        dump_json_file(self.config, os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'config.json'))

    def new_datasplit(self) -> 'DataSplit':
        split_file = self.config.get('experiment.dataset.split_file', None)
        if split_file is not None:
            split = DataSplit(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.img_path']), 
                            os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.label_path']), 
                            dataset = self.dataset_class(**self.dataset_args))
            split.load_from_file(os.path.join(pc.FILER_BASE_PATH, split_file))
            return split
        else:
            split = DataSplit(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.img_path']), 
                            os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.label_path']), 
                            dataset = self.dataset_class(**self.dataset_args))
            split.initialize( 
                            data_split = self.config['experiment.data_split'],
                            seed=self.config['experiment.dataset.seed'])
            return split

    def temporarly_overwrite_config(self, config: dict):
        r"""This function is useful for evaluation purposes where you want to change the config, e.g. data paths or similar.
            It does not save the config and should NEVER be used during training.
        """
        print("WARNING: NEVER USE \'temporarly_overwrite_config\' FUNCTION DURING TRAINING.")
        self.projectConfig = config
        #self.set_current_config()
        self.data_split = self.new_datasplit()
        self.set_size()

    def get_max_steps(self) -> int:
        r"""Get max defined training steps of experiment
        """
        return self.config['trainer.n_epochs']

    def reload(self) -> None:
        r"""Reload old experiment to continue training
            TODO: Add functionality to load any previous saved step
        """
        self.data_split = DataSplit(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.img_path']), 
                        os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.label_path']), 
                        dataset = self.dataset_class(**self.dataset_args))
        self.data_split.load_from_file(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'data_split.pkl'))
        loaded_config = load_json_file(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'config.json'))

        config_keys = list(self.config.keys())

        if 'experiment.run_hash' in config_keys:
            config_keys.remove('experiment.run_hash')

        for k, v in loaded_config.items():
            if k == "experiment.run_hash":
                pass
            elif k == "experiment.git_hash":
                config_keys.remove(k)
            else:
                config_keys.remove(k)
                valid = True
                valid = valid and k in self.config
                valid = valid and self.config[k] == v
                if not valid:
                    print(f"Configurations do not match on key '{k}'. Check if you are loading the correct experiment.")
                    print(f"Loaded: {v} | Current: {self.config.get(k, None)}")
                    in_key = input("Do you want to continue with the loaded configuration? [y, N] ")
                    if in_key.lower() != 'y':
                        raise Exception(f"Configurations do not match on key '{k}'. Check if you are loading the correct experiment.")

        if len(config_keys) > 0:
            print(f"Loaded configuration is missing keys: '{config_keys}'. Check if you are loading the correct experiment.")
            in_key = input("Do you want to continue with the loaded configuration? [y, N] ")
            if in_key.lower() != 'y':
                raise Exception(f"Loaded configuration is missing keys: {config_keys}. Check if you are loading the correct experiment.")

        self.config = loaded_config

        self.set_model_state("train")

        try:
            self.run = Run(run_hash=self.config['experiment.run_hash'], 
                           experiment=self.config['experiment.name'], repo=os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'Aim'))
        except Timeout as e:
            print("Timeout Error: ", e)
            in_key = input("Do you want to unlock manually? [y, N] ")
            if in_key.lower() == 'y':
                self.run = Run(run_hash=self.config['experiment.run_hash'], 
                               experiment=self.config['experiment.name'], repo=os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'Aim'),
                               force_resume=True)
            else:
                raise e
            

        self.load_model()
    
    def load_model(self) -> None:
        pretrained = False
        if 'pretrained' in self.config and self.currentStep == 0:
            print('>>>>>> Load Pretrained Model <<<<<<')
            pretrained = True
            #opt_loc = self.agent.optimizer
            #sch_loc = self.agent.scheduler
            pretrained_path = os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'Experiments', self.config['pretrained'] + "_" + self.projectConfig['description'])
            pretrained_step = self.current_step(os.path.join(pretrained_path, 'models')) #os.path.join(self.config['model_path'], 'models')
            model_path = os.path.join(pretrained_path, 'models', 'epoch_' + str(pretrained_step))
            #self.agent.optimizer = opt_loc
            #self.agent.scheduler = sch_loc
        else:
            model_path = os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'models', 'epoch_' + str(self.currentStep))

        if os.path.exists(model_path):
            print("Reload State " + str(self.currentStep))
            self.agent.load_state(model_path, pretrained=pretrained)# is not True:
            #    raise Exception("Model could not be loaded. Check if folder contains weights and architecture is identical.")


    def set_size(self) -> None:
        for dataset in self.datasets.values():
            if isinstance(self.config['experiment.dataset.input_size'][0], (tuple, list)):
                dataset.set_size(self.config['experiment.dataset.input_size'][-1])
            else:
                dataset.set_size(self.config['experiment.dataset.input_size'])

    def general_preload(self) -> None:
        r"""General experiment configurations needed after setup or loading
        """
        self.currentStep = self.current_step()
        self.agent.set_exp(self)
        self.fid = None
        self.kid = None

        if self.get_from_config('performance.unlock_CPU') is None or self.get_from_config('performance.unlock_CPU') is False:
            print('In basic configuration threads are limited to 1 to limit CPU usage on shared Server. Add \'unlock_CPU:True\' to config to disable that.')
            torch.set_num_threads(4)



    def general_postload(self) -> None:
        #create datasets and dataloaders
        self.datasets = {}
        for split in ['train', 'val', 'test']:
            if len(self.data_split.get_images(split)) > 0:
                self.datasets[split] = self.dataset_class(**self.dataset_args)
                self.datasets[split].setState(split)
                self.datasets[split].setPaths(os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.img_path']), 
                                              self.data_split.get_images(split), 
                                              os.path.join(pc.FILER_BASE_PATH, self.config['experiment.dataset.label_path']), 
                                              self.data_split.get_labels(split))

                self.datasets[split].set_experiment(self)


        precomputed_difficulties = None
        if self.get_from_config('trainer.datagen.difficulty_weighted_sampling') and self.config.get('trainer.datagen.difficulty_dict', False):
            precomputed_difficulties = pkl.load(open(self.config['trainer.datagen.difficulty_dict'], 'rb'))


        self.data_loaders = {}
        assert len(self.data_split.get_images('train')) > 0, "No training data available"
        if self.config['trainer.datagen.batchgenerators']:
            if self.get_from_config('trainer.num_steps_per_epoch') is not None:
                data_generator = StepsPerEpochGenerator(self.datasets["train"], self.config['trainer.num_steps_per_epoch'], 
                                                        num_threads_in_mt=self.config['performance.num_workers'], 
                                                        batch_size=self.config['trainer.batch_size'],
                                                        difficulty_weighted_sampling=self.get_from_config('trainer.datagen.difficulty_weighted_sampling'),
                                                        precomputed_difficulties=precomputed_difficulties)
            else:
                assert self.get_from_config('trainer.datagen.difficulty_weighted_sampling') is False, "not implemented for StepsPerEpochGenerator"
                data_generator = DatasetPerEpochGenerator(self.datasets["train"], 
                                                          num_threads_in_mt=self.config['performance.num_workers'], 
                                                          batch_size=self.config['trainer.batch_size']) 

            if self.get_from_config('trainer.datagen.augmentations'):
                transforms = get_transform_arr()
            else:
                transforms = []

            transforms.append(NumpyToTensor(keys=['image', 'label']))

            assert self.config['performance.num_workers'] > 0, "Batchgenerators need more than 0 workers"
            self.data_loaders["train"] = MyMultiThreadedAugmenter(data_generator, 
                                                                  Compose(transforms), 
                                                                  num_processes=self.config['performance.num_workers'])


        else:
            assert self.get_from_config('trainer.datagen.augmentations') is False, "not implemented yet"
            assert self.get_from_config('trainer.num_steps_per_epoch') is None, "not implemented yet"
            self.data_loaders["train"] = torch.utils.data.DataLoader(self.datasets["train"], shuffle=True, 
                                                                     batch_size=self.config['trainer.batch_size'],
                                                        num_workers=self.config['performance.num_workers'])
        
        self.set_size()


    def getFID(self) -> FrechetInceptionDistance:
        if self.fid is None:
            self.initializeFID()
        return self.fid
    
    def getKID(self) -> KernelInceptionDistance:
        if self.kid is None:
            self.initializeKID()
        return self.kid

    def bufferData(self) -> None:
        self.set_model_state("train")
        dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in tqdm(enumerate(dataloader_fid)):
            continue
        self.set_model_state("val")
        dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in tqdm(enumerate(dataloader_fid)):
            continue
        self.set_model_state("test")
        dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in tqdm(enumerate(dataloader_fid)):
            continue

    def initializeKID(self) -> None:
        # Reload or generate FID Model
        fid_path = os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'DatasetsFID', os.path.basename(self.config['img_path']), 'fid.dt')

        self.set_model_state("train")
        self.kid = KernelInceptionDistance(feature=2048, reset_real_features=False, subset_size=10)
        self.dataset.set_normalize(False)
        dataloader_kid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in enumerate(dataloader_kid):
            #print(data['image'].shape)
            sample = data['image'].to(torch.uint8)
            sample = sample.transpose(1,3)
            self.kid.update(sample, real=True)
            #self.fid.compute()
            break
        print("KID CREATED")
        self.dataset.set_normalize(True)


    def initializeFID(self) -> None:
        # Reload or generate FID Model
        fid_path = os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'DatasetsFID', os.path.basename(self.config['img_path']), 'fid.dt')

        if os.path.exists(fid_path):
            # RELOAD
            self.fid = load_pickle_file(fid_path)
        else:
            self.set_model_state("train")
            self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False)
            self.dataset.set_normalize(False)
            dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
            for i, data in enumerate(dataloader_fid):
                #print(data['image'].shape)
                sample = data['image'].to(torch.uint8)
                sample = sample.transpose(1,3)
                self.fid.update(sample, real=True)
                #self.fid.compute()
                break
            print("FID CREATED")
            self.dataset.set_normalize(True)

    def reload_model(self) -> None:
        r"""Reload model
            TODO: Move to a more logical position. Probably to the model and then call directly from the agent
        """
        if 'pretrained' in self.config and self.current_step == 0:
            print('Load Pretrained Model')
            pretrained_path = os.path.join(pc.FILER_BASE_PATH, pc.STUDY_PATH, 'Experiments', self.config['pretrained'] + "_" + self.projectConfig['description'])
            pretrained_step = self.current_step(model_path = os.path.join(self.config['pretrained_path'], 'models')) #os.path.join(self.config['model_path'], 'models')
            model_path = os.path.join(pretrained_path, 'models', 'epoch_' + str(pretrained_step), 'model.pth')
        else:
            model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep), 'model.pth')
        if os.path.exists(model_path):
            self.agent.load_model(model_path)

    def save_model(self) -> None:
        r"""TODO: Same as for reload -> move to better location
        """
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep+1))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))

    def current_step(self, model_path: str = None) -> int:
        r"""Find out the initial epoch by checking the saved models"""
        if model_path is None:
            model_path = os.path.join(pc.FILER_BASE_PATH, self.config['experiment.model_path'], 'models')
        if os.path.exists(model_path):
            dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            if dirs:
                maxDir = max([int(d.split('_')[1]) for d in dirs])
                return maxDir
        return 0

    def set_model_state(self, state: str) -> None:
        r"""TODO: remove? """
        self.model_state = state

        models = [self.model] if not isinstance(self.model, list) else self.model
        for m in models:
            if self.model_state == "train":
                m.train()
            else:
                m.eval()
        
    def get_from_config(self, tag: str) -> any:
        r"""Get from config
            #Args
                tag (String): Key of requested value
        """
        return self.config[tag]
        if tag in self.config.keys():
            return self.config[tag]
        else:
            input("Key not found in config: " + tag)
            return None

    @DeprecationWarning
    def set_current_config(self) -> None:
        r"""Set current config. This can change during training and will always 
            overwrite previous settings, but keep everything else
        """
        self.config = {}
        for i in range(0, len(self.projectConfig)):
            for k in self.projectConfig[i].keys():
                self.config[k] = self.projectConfig[i][k]
            if self.projectConfig[i]['n_epoch'] > self.currentStep:
                return

    def increase_epoch(self) -> None:
        r"""Increase current epoch
        """
        self.currentStep = self.currentStep +1
        #self.set_current_config()

    def get_current_config(self) -> dict:
        r"""TODO: remove?"""
        return self.config

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        r"""Write scalars to tensorboard
        """
        #self.writer.add_scalar(tag, value, step)
        
        self.run.track(step=step, value=value, name=tag)

    def write_img(self, tag: str, image: np.ndarray, step: int, context: dict = {}, normalize: bool = False) -> None:
        r"""Write an image to tensorboard
        """
        
        #self.writer.add_image(tag, image, step, dataformats='HWC')

        #image = np.uint8(image*256)
        image= np.squeeze(image)
        if normalize:
            img_min, img_max = np.min(image), np.max(image)
            image = (image - img_min) / (img_max - img_min) 
        else:
            image = np.clip(image, 0, 1)
        image = PILImage.fromarray(np.uint8(image*255)).convert('RGB')
        #image = PILImage(image)
        aim_image = Image(image=image, optimize=True, quality=50)
        self.run.track(step=step, value=aim_image, name=tag, context=context)

    def write_text(self, tag: str, text: str, step: int) -> None:
        r"""Write text to tensorboard
        """
        self.writer.add_text(tag, text, step)

    def write_histogram(self, tag: str, data: dict, step: int) -> None:
        r"""Write data as histogram to tensorboard
        """
        data = Distribution(data)
        self.run.track(step=step, value=data, name=tag)
        #self.writer.add_histogram(tag, data, step)

    def write_figure(self, tag: str, figure: figure, step: int) -> None:
        r"""Write a figure to tensorboard images
        """
        figure = Figure(figure)
        self.run.track(step=step, value=figure, name=tag)
        #self.writer.add_figure(tag, figure, step)


class DataSplit():
    r"""Handles the splitting of data
    """
    def __init__(self, path_image: str, path_label: str, dataset: Dataset):
        self.path_image = path_image
        self.path_label = path_label
        self.dataset = dataset

    def initialize(self, data_split: dict, seed: int):
        self.images = self.split_files(self.getFilesInFolder(self.path_image, self.dataset), data_split, seed)
        self.labels = self.split_files(self.getFilesInFolder(self.path_label, self.dataset), data_split, seed)

    def load_from_file(self, path: str):
        d = load_pickle_file(path)
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['train', 'val', 'test'])
        images = self.getFilesInFolder(self.path_image, self.dataset)
        labels = self.getFilesInFolder(self.path_label, self.dataset)
        self.images = {}
        self.labels = {}
        for s in ['train', 'val', 'test']:
            self.images[s] = {}
            self.labels[s] = {}
            for file in d[s]:
                self.images[s][file] = images[file]
                self.labels[s][file] = labels[file]

    def save_to_file(self, path: str):
        d = {}
        for s in ['train', 'val', 'test']:
            d[s] = []
            for case in self.images[s]:
                if case not in d[s]:
                    d[s].append(case)
        dump_pickle_file(d, path)

    def get_images(self, state: str) ->  dict:
        r"""#Returns the images of selected state
            #Args
                state (String): Can be 'train', 'val', 'test'
        """
        return self.get_data(self.images[state])

    def get_labels(self, state: str) -> dict:
        r"""#Returns the labels of selected state
            #Args
                state (String): Can be 'train', 'val', 'test'
        """
        return self.get_data(self.labels[state])

    def get_data(self, data: dict) -> list:
        r"""#Returns the data in a list rather than the stored folder strucure
            #Args
                data ({}): Dictionary ordered by {id, {slice, img_name}}
        """
        lst = data.values()
        lst_out = []
        for d in lst:
            lst_out.extend([*d.values()])
        return lst_out

    def split_files(self, files: dict, data_split: list, seed: int) -> dict:
        r"""Split files into train, val, test according to definition
            while keeping patients slics together.
            #Args
                files ({int, {int, string}}): {id, {slice, img_name}}
                data_split ([float, float, float]): Sum of 1
        """
        temp = sorted(files.keys())
        random.Random(seed).shuffle(temp)
        dic = {'train':{}, 'val':{}, 'test':{}}
        for index, key in enumerate(temp):
            if index / len(files) < data_split[0]:
                dic['train'][key] = files[key]
            elif index / len(files) < data_split[0] + data_split[1]: 
                dic['val'][key] = files[key]
            else:
                dic['test'][key] = files[key]
        print(f"Datasplit-> train entries: {len(dic['train'])}, val entries: {len(dic['val'])}, test entries: {len(dic['test'])}")
        return dic

    def getFilesInFolder(self, path: str, dataset: Dataset) -> list:
        r"""Get files in folder
            #Args
                path (String): Path to folder
                dataset (Dataset)
        """
        return  dataset.getFilesInPath(path) 
    
def merge_config(config_parent: dict, config_child: dict) -> None:
    r"""Merge config with current config
    """
    #try:
    #    config_child['name'] = config_parent['name'] + "_" + config_child['name']
    #except:
    #    print('MISSING NAME IN CONFIG')
    #return {**config_parent, **config_child}

    #this way the parent config has priority
    return config_child | config_parent