# QuantileFrailtyIndex
Data analysis and visualization code for publication, written in python.

# Contents
Example code (example_analysis.py) going over basic functions and simple visualizations.

Then the meat and potatoes: qfi_analysis.py has the code to generate all of the plots from the quantile frailty index paper (citation missing). It uses functions from qfi_functions.py, some of which are ancient, and may be lacking in documentation.

# Making it Go
Need the following things to make MOST of the analysis go:
    Biomarker data (table of biomarker values for a set of individuals).
        Missing data encoded with NaN, typical FI approach of ignore missing data used.
    Age, Sex, Mortality data (e.g. Time to death/death age/binary mortality at followup).
    
# Data Used: Can any be posted? Unclear.
ELSA data set is available for research purposes (https://www.elsa-project.ac.uk/accessing-elsa-data).
For the ELSA data you will have to grind through and collect the data into a useable form.
NHANES and CSHA data more widely available and easier to deal with (less reformatting to do etc.).


# Questions?
Send me an email at Garrett.Stubbings@Dal.ca if you have questions/can't get it going.
