""" The main function of rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.
  Typical usage example:


  python main_neural_method.py --config_file configs/COHFACE_TSCAN_BASIC.yaml --data_path "G:\\COHFACE"
"""

import argparse
import glob
import os
import time
import logging
import re
import sys
import tqdm
from config import get_config
from torch.utils.data import DataLoader
from dataset import data_loader
from neural_methods import trainer


def get_UBFC_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/UBFC_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "subject*")
    dirs = [{"index": re.search(
        'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
    return dirs


def get_COHFACE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/COHFACE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*")
    dirs = list()
    for data_dir in data_dirs:
        for i in range(4):
            subject = os.path.split(data_dir)[-1]
            dirs.append({"index": '{0}0{1}'.format(subject, i),
                        "path": os.path.join(data_dir, str(i))})
    return dirs


def get_PURE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/PURE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*-*")
    dirs = list()
    for data_dir in data_dirs:
        subject = os.path.split(data_dir)[-1].replace('-', '')
        dirs.append({"index": subject, "path": data_dir})
    return dirs


def get_Synthetics_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/SyntheticProcessed_dataloader.py """
    train_data_dirs = glob.glob(config.DATA.TRAIN_DATA_PATH + os.sep + "*.mat")
    train_dirs = list()
    for data_dir in train_data_dirs:
        subject = os.path.split(data_dir)[-1]
        train_dirs.append({"index": subject, "path": data_dir})

    val_data_dirs = glob.glob(config.DATA.VAL_DATA_PATH + os.sep + "*.mat")
    val_dirs = list()
    for data_dir in val_data_dirs:
        subject = os.path.split(data_dir)[-1]
        val_dirs.append({"index": subject, "path": data_dir})
    data_dict = {"train": train_dirs, "val": val_dirs}
    return data_dict

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/PURE_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument(
        '--device',
        default=None,
        type=int,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--data_path', default="G:\\COHFACE\\RawData", required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    return parser


def train(config, data_loader):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(
            config)

    model_trainer.train(data_loader)


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)

    # logging.
    # utils.check_dir(args.log_file)
    # logging.basicConfig(filename=args.log_file, level=args.verbose)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(config)

    # loads data
    if config.DATA.DATASET == "COHFACE":
        data_files = get_COHFACE_data(config)
        loader = data_loader.COHFACELoader.COHFACELoader
    elif config.DATA.DATASET == "UBFC":
        data_files = get_UBFC_data(config)
        loader = data_loader.UBFCLoader.UBFCLoader
    elif config.DATA.DATASET == "PURE":
        data_files = get_PURE_data(config)
        loader = data_loader.PURELoader.PURELoader
    elif config.DATA.DATASET == "SYNTHETICS":
        data_dict = get_Synthetics_data(config)
        loader = data_loader.SyntheticsLoader.SyntheticsLoader
    else:
        raise ValueError(
            "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")
    train_data = loader(
        name="train",
        data_dirs=data_dict["train"],
        config_data=config.DATA)
    dataloader = {
        "train": DataLoader(
            dataset=train_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True),
    }

    if True: # TODO: This requires supporting if users don't want to have validation at all. 
        valid_data = loader(
            name="valid",
            data_dirs=data_dict["val"],
            config_data=config.DATA)
        dataloader["valid"] = DataLoader(
            dataset=valid_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True)
    else:
        dataloader['valid'] = None
    train(config, dataloader)

