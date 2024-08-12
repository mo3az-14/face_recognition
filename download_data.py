import numpy as np
import config
import os
import shutil
import kaggle


def split_data(src_data_directory: str, ratio: float = config.TRAIN_PRECENT):
    """Splits the Data into train/test split with train ratio of TRAIN_PRECENT"""

    names = [dirs for root, dirs, files in os.walk(src_data_directory)][0]

    rng = np.random.default_rng()
    random_idxs = rng.choice(len(names), len(names), replace=False)
    # make train and test directories
    try:
        os.mkdir(config.TRAIN_DATASET)
        os.mkdir(config.TEST_DATASET)
    except Exception:
        print("couldn't create train test directories or they already exists")

    try:
        for i in random_idxs[: int(len(random_idxs) * ratio)]:
            shutil.move(src_data_directory + "/" + names[i], config.TRAIN_DATASET)
    except Exception:
        print("train dataset is already split")

    try:
        for i in random_idxs[int(len(random_idxs) * ratio) :]:
            shutil.move(src_data_directory + "/" + names[i], config.TEST_DATASET)
    except Exception:
        print("test dataset is already split")

    print(
        int(np.floor(len(random_idxs) * ratio)) - int(np.ceil(len(random_idxs) * ratio))
    )
    # remove the rest of the files
    for root, dirs, files in os.walk(src_data_directory):
        for f in files:
            os.remove(src_data_directory + "/" + f)

    os.rmdir(src_data_directory)


def uncompress_data(tgz_file_path: str, dst: str):
    """uncompress the given file tgz file and removes the .tgz file"""
    import tarfile

    file = tarfile.open(tgz_file_path)
    file.extractall(dst)
    file.close()
    os.remove(tgz_file_path)


if __name__ == "__main__":
    if not os.path.exists(config.DATASET):
        print(
            'Didn\'t find a "dataset" folder... \ndownloading the dataset from kaggle. Please provide your kaggle username and\
    api key to download in config folder or in .kaggle folder'
        )

        # setup enviroment and download the dataset
        os.environ["KAGGLE_USERNAME"] = config.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = config.KAGGLE_KEY
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "atulanandjha/lfwpeople", path=config.DATASET, unzip=True
        )

        # files we won't need
        os.remove("dataset/pairsDevTrain.txt")
        os.remove("dataset/pairsDevTest.txt")
        os.remove("dataset/pairs.txt")

        uncompress_data(config.DATASET + "/lfw-funneled.tgz", config.DATASET)
        split_data(src_data_directory=config.DATASET + "/lfw_funneled")

    else:
        print(
            'found the "dataset" folder. If the folder is empty please delete it and re-run the script'
        )
