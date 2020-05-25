from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f'There are {gpu_count} GPUs with the current setup, so we will increase batch size and later run the model on all GPUs')
            batch_size = config['batch_size']*gpu_count
        else:
            print("There is only 1 GPU available")
    else:
        print("There are no GPUs available")
        batch_size = config['batch_size']


    dataset = DataSetWrapper(batch_size, **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
