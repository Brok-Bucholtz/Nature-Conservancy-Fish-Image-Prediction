from glob import glob
from shutil import copytree, rmtree, copyfile
from os.path import exists, basename
from tqdm import tqdm


def run():
    combined_dir = './data/combine/'

    aug_file_paths = {
        'rotated': list(glob('./data/augmentation/rotation/*/*.jpg')),
        'scaled': list(glob('./data/augmentation/scale/*/*.jpg'))
    }

    # Clear previous combine datasets
    if exists(combined_dir):
        rmtree(combined_dir)

    copytree('./data/train/', combined_dir)

    for name, file_paths in aug_file_paths.items():
        print('Combining {} images'.format(name))
        for rotated_file in tqdm(file_paths):
            label = rotated_file[29:-14]
            copyfile(rotated_file, combined_dir + label + '/' + name + '_' + basename(rotated_file))

if __name__ == '__main__':
    run()
