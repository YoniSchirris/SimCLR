from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import torch
import argparse



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="config.yaml", help='Name of config file to use')
    config = parser.parse_args()
    print(f"Loading {config.config_path}")    
    config = yaml.load(open(config.config_path, "r"), Loader=yaml.FullLoader)
    print(f"Model in this file is {config['model']['base_model']}")
    if torch.cuda.is_available() and config['allow_multiple_gpu']:
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f'There are {gpu_count} GPUs with the current setup, so we will increase batch size and later run the model on all GPUs')
            config['batch_size'] *= gpu_count
        else:
            print("There is only 1 GPU available")
    else:
        print("There are no GPUs available")

    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
