from os.path import basename


def save_predictions(prediction_file, image_filenames, labels, predictions):
    # Save file
    header = 'image,' + ','.join(labels)
    with open(prediction_file, 'w') as out:
        out.write(header + '\n')
        for image_filenames, prediction in zip(image_filenames, predictions):
            out.write(basename(image_filenames) + ',' + ','.join([str(x) for x in prediction]) + '\n')
