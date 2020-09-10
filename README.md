# Content Based Image Retrieval (CBIR) with Wavelet features
For more information on the project, check out the [project wiki](https://github.com/TimothySimons/ICV_project/wiki).

## Getting Started

### Prerequisites
Simply `pip install -r requirements.txt` in your Python virtual environment and you're ready to go!

### Dataset
The data used in this project can be found [here](https://appen.com/datasets/open-images-annotated-with-bounding-boxes/).

## Running the Application
As an aside, for best performance, crop and resize your dataset prior to running the application. When doing this be sure the following key word arguments reflect this change `preprocessing.process_image(file_path, cropped=True, scaled=True, mapped=False)`.  Ensure that these key word arguments are set to `False` when processing the query image.

To run the application:
```
python gui <db-folder>
``` 
Here, `<db-folder>` is the relative path to the database containing the application _display_ images. This folder is different from the folder selected in the GUI if the user has preprocessed the images prior to application execution. The folder selected in the GUI contains the preprocessed images and the one passed in as a command line argument contains higher resolution versions of those same images (with the same file names). 
