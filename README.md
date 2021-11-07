

Analysis code for AJE paper "Identifying Predictors of Opioid Overdose Death at a Neighborhood Level with Machine Learning"
------------

__Authors:__ Robert C. Schell,
Bennett Allen,
William C. Goedel,
Benajmin D. Hallowell,
Rachel Scagos,
Yu Li,
Maxwell S. Krieger,
Daniel B. Neill,
Brandon D.L. Marshall,
Magdalena Cerda, and
Jennifer Ahern







---

Repository Contents
------------

This repository contains the code to reproduce the results in the _American Journal of Epidemiology_ article "Identifying Predictors of Opioid Overdose at a Neighborhood Level with Machine Learning."  

Dataset Availability
-----------

Because of the sensitive nature of the overdose death data, it is only available with IRB approval both from the home institution and the Rhode Island Department of Health. The predictors are available for download online at the American Community Survey website https://www.census.gov/programs-surveys/acs/data.html.

---

R Code & Simulating
------------

1. Run the package installation R code
2. Run the function list in Github. This creates functions that allow for the specification of a hyperparameter grid based on performance in the outer loop for the Random Forest in the inner loop of the double cross-validation model. 
3. Run the double cross-validation code with "Double Cross Validation" R code. This file also contains code for the calibration plot.
