import click

from src.dataops import PACSDataset, DigitsDGDataset, OfficeHomeDataset


@click.command()
@click.option('--root_data_dir', 'root_uri', type=click.Path(dir_okay=True, path_type=str), required=True, help='Specifies a directory path to store the datasets.')
def download_datasets(root_uri: str):
    
    print('Donwload begins')

    PACSDataset.download(root_uri)
    DigitsDGDataset.download(root_uri)
    OfficeHomeDataset.download(root_uri)


if __name__ == '__main__':
    download_datasets()