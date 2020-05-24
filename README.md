# Artifical-pancreas

## Prerequisites
* Numpy
* Pandas
* Matplotlib
* Scipy
* Sklearn
* Pickle
* Seaborn
* Tensorflow

# Folder strcuture descriptions
* **Meal_detection_training.ipynb** - Script for training the data
* **Meal_detection_testing.py** - Script for testing the data
* **Models**: Contains trained ML models which can be used while testing

# Testing instructions
The test function takes an input of a test data csv file and predicts the
respective outputs of the data in a new csv file.
To test a particular model, call the function defined for that model with the 
absolute file path as the parameter to the function. 

For example:
If testing needs to be done for a particular model like KNN. Search for the
predefined function like Meal_Detection_KNN().
The parameter for the function should be the absolute location.
So it should be run as
Meal_Detection_KNN(“filelocation/testdata.csv”)
Similarly for other models location should be passed and the function
should be run. The names of the functions are provided in comments in the
python code file.
