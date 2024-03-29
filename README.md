# Effect of biomass water dynamics in cosmic ray neutron sensor observations: A long-term analysis of maize-soybean rotation in Nebraska
Supporting materials for the study: 
Effect of biomass water dynamics in cosmic-ray neutron sensor observations: A long-term analysis of maize-soybean rotation in Nebraska.  
This study was performed to determine a biomass correction factor for the cosmic ray neutron sensor (CRNS) in row crops.  
# Description
This repository contains all data and code used for analysis and figure creation in the paper titled "Effect of biomass water dynamics in cosmic-ray neutron sensor observations: A long-term analysis of maize-soybean rotation in Nebraska" by Morris et al. (in submission in 2024).  This code and data can be used to reproduce the analysis performed in this study.  
# Study Abstract
Precise measurement of soil water content (SWC) is crucial for effective water resource management. This study utilizes the Cosmic Ray Neutron Sensor (CRNS) for area-averaged SWC measurements, emphasizing the need to consider all hydrogen sources, including the time-variable plant biomass and water content.  Near Mead, Nebraska, three field sites (CSP1, CSP2, and CSP3) growing a maize-soybean rotation have been monitored for 5 (CSP1 and CSP2) and 13 years (CSP3). Data collection includes destructive biomass water equivalent (_BWE_) biweekly sampling, epithermal neutron counts, atmospheric meteorological variables, and point-scale SWC from a sparse Time Domain Reflectometry (TDR) network (4 locations and five depths).  In 2023, dense gravimetric SWC surveys were collected eight (CSP1 and CSP2) and nine times (CSP3) over the growing season (April to October).  The _N0_ parameter, from Desilets et al. (2010), exhibits a linear relationship with _BWE_, suggesting a straightforward vegetation correction factor may be suitable (_fb_).  Results from the 2023 gravimetric surveys and long-term TDR data indicate a neutron count rate reduction of about 1% for every 1 kg/m^2 (or mm of water) increase in _BWE_.  This reduction factor aligns with existing shorter-term row-crop studies but nearly doubles the value previously reported for forests.  This long-term study contributes insights into the vegetation correction factor for CRNS, helping resolve a long-standing issue within the CRNS community.
# Contents
This repository contains several folders named for what each contains. 
## Data ##
The data folder contains two subfolders: Data_Excel and Data_in_Code_Format. 
### Data_Excel ###
#### Ameriflux_Summary.xlsx ####
This contains all TDR data used in this study.  This was provided by Dr. Andy Suyker. 
#### BWE_Summary.xlsx ####
This contains all BWE data from biweekely sampling performed by David Scoby. 
#### Field_Logs.xlsx ####
This contains all field logs for sites CSP1, CSP2, and CSP3.  This was provided by the researchers at ENREEC.
#### Gravimetric_Sampling_Summary.xlsx ####
This contains all gravimetric sampling data from the 2023 sampling season.  This was performed by myself and my labmates.
#### Neutron_Counts_Summary.xlsx ####
This contains the Npvi neutron counts from all 3 sites as well as any associated evironmental data.  
#### Precipitation_Irrigation_Summary.xlsx ####
This file contains the precipitation and irrigation data.  
### Data_in_Code_Format ###
These files contain all the same information as the Data_Excel file, but they are organized into yearly files, as this is how it is organized and written in my python script.  
## Python_Code ##
This folder contains all python code that was used for analysis and figure creation in this study.  The python version used was python 3.8.10.  All code in this study will need to modified to have your path name input for where files were drawn from.  All areas that need modification are written in the script as "YOUR PATH", "YOUR PATH TO MAKE", and "YOUR PATH TO SAVE".  Please note that there a couple scripts dependent on a file created in a prior script, these are listed below:
SWC_95Confidence_Interval.py must by run prior to N0_Time_Series_Individual_Years and N0_Continuous_for_All_Time_Series
### pyenvironment.txt ###
This file contains the python environment and all packages to be able to run the code.  To install these packages run the following script (this was done in Visual Studio Code): 
pip install -r pyenvironment.txt
#### BWE_Time_Series_and_Error_Analysis.py ####
This creates BWE time series figures as well as plotting their 95% confidence intervals.
#### fb.py ####
This analyzes data, performs calculations, and plots the correction factor created in this study.
#### Gravimetric_Sampling.py ####
This creates N0 vs BWE for gravimetric sampling as well as performs all weighting on gravimetric data.
#### Gravimetric_TDR_Comparison.py ####
This creates the plots comparing gravimetric data to TDR data. 
#### Multiyear_N0vsBWE.py ####
This plots N0 vs BWE for all years in each field as one plot per field.  It has gravimetric and TDR data.
#### N_N0_BWE_Plots.py ####
This plots Npvi, N0, and BWE on the same plot.  
#### N0_Continuous_for_All_Time_Series.py ####
This plots continuous N0 data for all times, including those not used in this study, with one plot per field with 95% confidence intervals.
#### N0_Time_Series_Individual_Years.py ####
This creates a plot for every individual year in every individual field with 95% confidence intervals.
#### N0_vs_BWE.py ####
This creates N0 vs BWE plots for every field in every individual year.  
#### NPVI_Time_Series.py ####
This creates Npvi date time series for every individual year in every individual field. 
#### Precipitation_Irrigation_TDR_Time_Series.py ####
This creates time series plots of TDR data as a scatterplot and precipitation and irrigation data as bar charts.
#### SWC_95Confidence_Interval.py ####
This creates the 95% confidence intervals plots and related files for the soil water content values from TDR.  
#### TDR_All_Depths_Shown_Individually_Time_Series.py ####
This creates a plot of all TDR data with each depth shown individually.
#### TDR_Continuous_Time_Series.py ####
This plots the TDR data used in the study as a time series for all years and time continuously.
#### TDR_Time_Series_Individual_Fields.py ####
This plots the TDR time series for all individual fields and years.  
#### Vapour_Pressure_Deficit.py ####
This plots the vapour pressure deficit information for the future studies portion of this study.  It is not directly used in this study, but is a future direction.
## Supplementary_Materials.docx ##
This is a Microsoft Word document that contains all supplementary figures, equations, and tables for this study.  
