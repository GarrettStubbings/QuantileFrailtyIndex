# QuantileFrailtyIndex
Data analysis and visualization code for publication, written in python, including some jupyter notebook stuff.

# Contents
Example code going over basic functions and simple visualizations.
Big mess of code snippets written to produce specific plots for publication.

# Making it Go
Need the following things to make MOST of the analysis go:
    Biomarker data (table of biomarker values for a set of individuals).
        Missing data encoded with NaN, typical FI approach of ignore missing data used.
    Age, Sex, Mortality data (e.g. Time to death/death age/binary mortality at followup).
    
# Data Used
ELSA data set is available for research purposes (https://www.elsa-project.ac.uk/accessing-elsa-data).
Some preprocessing code included in this repository (very messy jupyter notebook, credit: @SpencerFarrell).
NHANES and CSHA data more widely available and easier to deal with (less reformatting to do etc.).
