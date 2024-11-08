import gdown


def download_model(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)
    return output


first_model = r"1jTPE_9-AzYMww1BTSTdI7Z64qpSeysoV"
second_model = r"1tSyPKY77sZX5HBkj8-LGdTYbZ_Fi3N11"
if __name__ == "__main__":
    download_model(first_model, "model.pt")
    download_model(second_model, "model2.pt")
