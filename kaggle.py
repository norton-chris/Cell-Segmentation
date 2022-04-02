# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()

# from kaggle.api.kaggle_api_extended import KaggleApi

# api = KaggleApi()
# files = api.competition_download_files("twosigmanews")

# import requests

# # The direct link to the Kaggle data set
# data_url = 'https://www.kaggle.com/dataset/6537c391ff7aadaf1dc8aefe514d4067f7fe63e26b9de9ebe7cce2f9352b803e'

# # The local path where the data set is saved.
# local_filename = "actsual.csv"

# # Kaggle Username and Password
# kaggle_info = {'UserName': "myUsername", 'Password': "myPassword"}

# # Attempts to download the CSV file. Gets rejected because we are not logged in.
# r = requests.get(data_url)

# # Login to Kaggle and retrieve the data.
# r = requests.post(r.url, data = kaggle_info)

# # Writes the data to a local file one chunk at a time.
# f = open(local_filename, 'wb')
# for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory

    # if chunk: # filter out keep-alive new chunks
        # f.write(chunk)
# f.close()

# import kaggle

# kaggle.api.authenticate()

# kaggle.api.dataset_download_files('Cell Adhesion', path='.', unzip=False)

import requests


url = 'https://www.kaggle.com/dataset/6537c391ff7aadaf1dc8aefe514d4067f7fe63e26b9de9ebe7cce2f9352b803e/download'
r = requests.get(url, allow_redirects=True)

open('download.zip', 'wb').write(r.content)