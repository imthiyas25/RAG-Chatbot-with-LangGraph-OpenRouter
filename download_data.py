import os
import kaggle


def download_kaggle_dataset():
    """Download Consumer Complaints dataset from Kaggle."""
    os.makedirs("data", exist_ok=True)
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        "shashwatwork/consume-complaints-dataset-fo-nlp",
        path="./data",
        unzip=True,
    )
    print("✓ Dataset downloaded to ./data/")


if __name__ == "__main__":
    download_kaggle_dataset()

