import os
import glob


def main():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    imagenet_train_images = os.path.join(cur_path, "..", "ILSVRC", "Data", "CLS-LOC", "train")
    imagenet_val_images = os.path.join(cur_path, "..", "ILSVRC", "Data", "CLS-LOC", "val")
    target_path = os.path.join(cur_path, "imagenet-repmet")
    target_path_train = os.path.join(target_path, "train")
    target_path_val = os.path.join(target_path, "val")
    validation_annotation_file = os.path.join(cur_path, "imagenet-repmet", "val_classes.txt")
    os.makedirs(target_path_train)
    os.makedirs(target_path_val)

    # get test classes to exclude
    repmet_test_classes_path = os.path.join(cur_path, "..", "repmet_test_classes.txt")
    with open(repmet_test_classes_path, "r") as fid:
        repmet_test_classes = fid.readlines()
    classes_to_exclude = {}
    for cl in repmet_test_classes:
        classes_to_exclude[cl[:-1]] = 1 # cut off the EOL symbol
    
    # loop over all train classes
    all_class_folders = glob.glob(os.path.join(imagenet_train_images, "n*"))
    for class_folder in all_class_folders:
        class_name = os.path.basename(class_folder)
        if class_name not in classes_to_exclude:
            os.symlink(class_folder, os.path.join(target_path_train, class_name))

    # move validation into labeled subfolders
    for class_folder in all_class_folders:                                                                                                                                                                   
        class_name = os.path.basename(class_folder)
        if class_name not in classes_to_exclude:
            os.makedirs(os.path.join(target_path_val, class_name))
    
    with open(validation_annotation_file, "r") as fid:
        validation_annotation_lines = fid.readlines()
    for line in validation_annotation_lines:
        file_name, class_name = line.split(" ")
        class_name = class_name[:-1] # chop off the EOL symbol
        if class_name not in classes_to_exclude:
            os.symlink(os.path.join(imagenet_val_images, file_name),
                         os.path.join(target_path_val, class_name, file_name))

if __name__ == "__main__":
    main()
