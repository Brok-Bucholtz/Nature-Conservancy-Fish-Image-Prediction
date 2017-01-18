from glob import glob
from shutil import copytree, rmtree, copyfile
from os.path import exists, basename
from tqdm import tqdm


def run():
    combined_dir = './data/combine/'
    rotated_filepaths = list(glob('./data/augmentation/rotation/*/*.jpg'))

    # Clear previous combine datasets
    if exists(combined_dir):
        rmtree(combined_dir)

    copytree('./data/train/', combined_dir)

    for rotated_file in tqdm(rotated_filepaths):
        label = rotated_file[29:-14]
        copyfile(rotated_file, combined_dir + label + '/' + 'rotated_' + basename(rotated_file))

if __name__ == '__main__':
    run()
