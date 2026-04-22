import shutil
import os

# Define source and destination
source_dir = r"C:\Users\Prassana Pawar\.cache\kagglehub\datasets\feyzazkefe\trashnet\versions\1"
destination_dir = "./trashnet_data" # Creates a new folder in current dir

# Copy the dataset
shutil.copytree(source_dir, destination_dir)
print(f"Dataset moved to: {os.path.abspath(destination_dir)}")
