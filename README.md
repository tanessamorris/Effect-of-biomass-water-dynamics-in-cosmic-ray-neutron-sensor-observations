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
This folder contains all python code that was used for analysis and figure creation in this study.  All code in this study will need to modified to have your path name input for where files were drawn from.  All areas that need modification are written in the script as "YOUR PATH", "YOUR PATH TO MAKE", and "YOUR PATH TO SAVE".  Please note that there a couple scripts dependent on a file created in a prior script, these are listed below:

### pyenvironment.txt ###
This file contains the python environment and all packages to be able to run the code.  
