from Raw_Data import extract_raw_data
from Feature_Extraction import extraction_main
from Model_Logistic_Regression import models


def start():
    extract_raw_data()
    extraction_main()
    models()


if __name__ == "__main__":
    print('main')

