# Diabetes_lifestyle_prediction

This repo was made as part of the dissertation of the first internship of the bio-informatics curriculum on the Hanze Hogeschool.
This repo holds analyses and code as used and refered to in the report.

The diabetes folder has several notebooks containing analysis:
* age_and_lada : modeling on different age categories and the exclusion of LADA patients
* modeling_crosstest : crosstesting models on different clusters
* seperate_file_modeling : Looking into the predictive power of seperate files / categories

The pipeline folder holds the experimental_modeling_pipeline file which has two classes:
* Experimental_modeling_pipeline : Used for model creation
* DiabetesPreprocessing : Preprocessing of datafiles