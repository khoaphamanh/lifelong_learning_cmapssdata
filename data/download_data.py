import requests
import zipfile
import os


# function to download the zipfile data
def download_zip(url, path_output):

    # download zip directory develop
    try:
        # Send a request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the ZIP file temporarily
        zip_file_path = "{}.zip".format(path_output.split("/")[-1])
        with open(zip_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Extract the contents of the ZIP file
        unzip(path_zip_file=zip_file_path, path_output=path_output)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download ZIP file: {e}")
    except zipfile.BadZipFile:
        print(f"Failed to extract: Not a valid ZIP file")


# unzip data
def unzip(path_zip_file, path_output):
    try:
        # Open the ZIP file
        with zipfile.ZipFile(path_zip_file, "r") as zip_ref:
            # Extract all files to the specified directory
            zip_ref.extractall(path_output)
            print(
                f"ZIP file '{path_zip_file}' extracted successfully to '{path_output}'"
            )

        # Remove the temporary ZIP file
        os.remove(path_zip_file)

    except zipfile.BadZipFile:
        print(f"Error: '{path_zip_file}' is not a valid ZIP file")


# download data function
def download_data_unzip(url=None):

    # name of the data
    name_data = "CMAPSSData"

    # path data directory
    path_data_directory = os.path.dirname(os.path.abspath(__file__))

    # path cmapssdata directory
    path_cmapss_directory = os.path.join(path_data_directory, name_data)

    download_zip(url=url, path_output=path_cmapss_directory)


# url for direct download data cmapss
url_cmapss = "https://seafile.cloud.uni-hannover.de/f/da539b30c1ad4e758ee7/?dl=1"

download_data_unzip(url=url_cmapss)
