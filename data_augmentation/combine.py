from glob import glob
from shutil import copytree, rmtree, copyfile
from os.path import exists, basename, dirname
from tqdm import tqdm


def run():
    combined_dir = './data/combine/'
    augment_data = {
        'rotated': glob('./data/augmentation/rotation/*/*.jpg'),
        'scaled': glob('./data/augmentation/scale/*/*.jpg')
    }

    # Clear previous combine datasets
    if exists(combined_dir):
        rmtree(combined_dir)

    copytree('./data/train/', combined_dir)

    for name, file_paths in augment_data.items():
        print('Combining {} images'.format(name))
        for augment_file in tqdm(list(file_paths)):
            label = dirname(augment_file)[len(dirname(dirname(augment_file)))+1:]
            copyfile(augment_file, combined_dir + label + '/' + name + '_' + basename(augment_file))

if __name__ == '__main__':
    run()
