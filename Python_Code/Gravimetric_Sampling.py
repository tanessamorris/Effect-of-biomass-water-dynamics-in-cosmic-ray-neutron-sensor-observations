#import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
import six


#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given


##############################################
##############################################
###################2023#######################
##############################################
##############################################

##############################################
###################CSP1#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP1 = pd.read_csv('YOUR PATH/2023_CSP1_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP1 = GWC_CSP1['SWC_1_1_1']
a_CSP1 = SWC_CSP1

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1 = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1 = NPIV_CSP1['NPIV']

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1 = a_CSP1
lwc = 0.058
soc = 0.0092
N_CSP1 = NPIVCount_CSP1

#use parameters to calculate N0
N0_CSP1 = N_CSP1 / ((0.0808 / ((pwc_CSP1 + lwc + soc) + 0.115)) + 0.372)
N0_CSP1.to_csv(r'YOUR PATH TO MAKE/2023_CSP1N0.csv')

### GET BWE DATA FOR PLOTTING ###

#Import BWE data, determined by destructive sampling
BWEcsv_CSP1 = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1 = BWEcsv_CSP1['BWE']

#define plot variables
x_CSP1 = BWE_CSP1
y_CSP1 = N0_CSP1

#calculate r-squared
corr_matrix_CSP1 = np.corrcoef(x_CSP1,y_CSP1)
corr_CSP1 = corr_matrix_CSP1[0,1]
R_sq_CSP1 = corr_CSP1**2

#Least square regression
A_CSP1_23 = np.vstack([x_CSP1, np.ones(len(x_CSP1))]).T
y_CSP1_23 = np.array(y_CSP1)  
y_CSP1_23 = y_CSP1.values[:, np.newaxis]
alpha_CSP1 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_23.T,A_CSP1_23)),A_CSP1_23.T)),y_CSP1)
m_CSP1 = alpha_CSP1[0]
n_CSP1 = alpha_CSP1[1]
mm_CSP1 = float(m_CSP1)
mmm_CSP1 = [float(m_CSP1)]
nn_CSP1 = float(n_CSP1)
r_mm_CSP1 = round(mm_CSP1,2)
r_nn_CSP1 = round(nn_CSP1,2)

#95% Confidence interval on slope and intercept
z_CSP1 = sm.add_constant(x_CSP1)
model_CSP1 = sm.OLS(y_CSP1, z_CSP1).fit()
confidence_intervals_CSP1 = model_CSP1.conf_int(alpha=0.05)
lower_bound_intercept_CSP1, upper_bound_intercept_CSP1 = confidence_intervals_CSP1.loc['const']
lower_bound_slope_CSP1, upper_bound_slope_CSP1 = confidence_intervals_CSP1.loc['BWE']
Slope_CI_CSP1 = mm_CSP1 - (lower_bound_slope_CSP1)
Intercept_CI_CSP1 = nn_CSP1 - (lower_bound_intercept_CSP1)
R_Slope_CI_CSP1 = round(Slope_CI_CSP1, 2)
R_Intercept_CI_CSP1 = round(Intercept_CI_CSP1, 2)

#P_value for the least squares regression
p_values_CSP1 = model_CSP1.pvalues
p_value_slope_CSP1 = p_values_CSP1.loc['BWE']
p_value_int_CSP1 = p_values_CSP1.loc['const']
r_p_value_slope_CSP1 = "{:.1e}".format(p_value_slope_CSP1)
r_p_value_int_CSP1 = "{:.1e}".format(p_value_int_CSP1)
p_slope_CSP1 = float(r_p_value_slope_CSP1)
p_int_CSP1 = float(r_p_value_int_CSP1)

#OLS model
model_CSP1 = sm.OLS(y_CSP1, z_CSP1).fit()

#BWE standard error
BWESE_CSP1 = pd.read_csv('YOUR PATH/2023_CSP1_BWE_SE.csv')
SE_CSP1 = BWESE_CSP1['SE']
p_CSP1_SE = SE_CSP1

#N0 standard error
N0_SE_CSP1 = pd.read_csv('YOUR PATH/2023_CSP1_N0_SE.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP1 = (2* N0_SE_CSP1['SE']) + pwc_CSP1

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
lwc = 0.058
soc = 0.0092

#use parameters to calculate N0
N0_SE_Calc_CSP1 = N_CSP1 / ((0.0808 / ((Water_Content_SE_CSP1 + lwc + soc) + 0.115)) + 0.372)
N0_error_CSP1 = N0_SE_Calc_CSP1 - N0_CSP1

#actually plot data with best fit line, slope, and r-squared
a_CSP1_1, b_CSP1_1 = np.polyfit(x_CSP1,y_CSP1,1)
plt.errorbar(x_CSP1, y_CSP1, xerr=p_CSP1_SE, yerr=N0_error_CSP1, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP1, a_CSP1_1*x_CSP1+b_CSP1_1, linestyle = '-', color = 'blue')
plt.text(-0.1, 2323, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP1, r_p_value_slope_CSP1), size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP1 Gravimetric Data 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.ylim(2300, 3020)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1,r_nn_CSP1], [R_Slope_CI_CSP1,R_Intercept_CI_CSP1]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
###################CSP2#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP2 = pd.read_csv('YOUR PATH/2023_CSP2_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP2 = GWC_CSP2['SWC_1_1_1']
a_CSP2 = SWC_CSP2

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2 = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2 = NPIV_CSP2['NPIV']

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2 = a_CSP2
lwc = 0.058
soc = 0.0092
N_CSP2 = NPIVCount_CSP2

#use parameters to calculate N0
N0_CSP2 = N_CSP2 / ((0.0808 / ((pwc_CSP2 + lwc + soc) + 0.115)) + 0.372)
N0_CSP2.to_csv(r'YOUR PATH TO MAKE/2023_CSP2N0.csv')

### GET BWE DATA FOR PLOTTING ###

#Import BWE data, determined by destructive sampling
BWEcsv_CSP2 = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2 = BWEcsv_CSP2['BWE']

#define plot variables
x_CSP2 = BWE_CSP2
y_CSP2 = N0_CSP2

#calculate r-squared
corr_matrix_CSP2 = np.corrcoef(x_CSP2,y_CSP2)
corr_CSP2 = corr_matrix_CSP2[0,1]
R_sq_CSP2 = corr_CSP2**2

#Least square regression
A_CSP2_23 = np.vstack([x_CSP2, np.ones(len(x_CSP2))]).T
y_CSP2_23 = np.array(y_CSP2)  
y_CSP2_23 = y_CSP2.values[:, np.newaxis]
alpha_CSP2 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_23.T,A_CSP2_23)),A_CSP2_23.T)),y_CSP2)
m_CSP2 = alpha_CSP2[0]
n_CSP2 = alpha_CSP2[1]
mm_CSP2 = float(m_CSP2)
mmm_CSP2 = [float(m_CSP2)]
nn_CSP2 = float(n_CSP2)
r_mm_CSP2 = round(mm_CSP2,2)
r_nn_CSP2 = round(nn_CSP2,2)

#95% Confidence interval on slope and intercept
z_CSP2 = sm.add_constant(x_CSP2)
model_CSP2 = sm.OLS(y_CSP2, z_CSP2).fit()
confidence_intervals_CSP2 = model_CSP2.conf_int(alpha=0.05)
lower_bound_intercept_CSP2, upper_bound_intercept_CSP2 = confidence_intervals_CSP2.loc['const']
lower_bound_slope_CSP2, upper_bound_slope_CSP2 = confidence_intervals_CSP2.loc['BWE']
Slope_CI_CSP2 = mm_CSP2 - (lower_bound_slope_CSP2)
Intercept_CI_CSP2 = nn_CSP2 - (lower_bound_intercept_CSP2)
R_Slope_CI_CSP2 = round(Slope_CI_CSP2, 2)
R_Intercept_CI_CSP2 = round(Intercept_CI_CSP2, 2)

#P_value for the least squares regression
p_values_CSP2 = model_CSP2.pvalues
p_value_slope_CSP2 = p_values_CSP2.loc['BWE']
p_value_int_CSP2 = p_values_CSP2.loc['const']
r_p_value_slope_CSP2 = "{:.1e}".format(p_value_slope_CSP2)
r_p_value_int_CSP2 = "{:.1e}".format(p_value_int_CSP2)
p_slope_CSP2 = float(r_p_value_slope_CSP2)
p_int_CSP2 = float(r_p_value_int_CSP2)

#OLS model
model_CSP2 = sm.OLS(y_CSP2, z_CSP2).fit()

#BWE standard error
BWESE_CSP2 = pd.read_csv('YOUR PATH/2023_CSP2_BWE_SE.csv')
SE_CSP2 = BWESE_CSP2['SE']
p_CSP2_SE = SE_CSP2

#N0 standard error
N0_SE_CSP2 = pd.read_csv('YOUR PATH/2023_CSP2_N0_SE.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP2 = (2* N0_SE_CSP2['SE']) + pwc_CSP2

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
lwc = 0.058
soc = 0.0092

#use parameters to calculate N0
N0_SE_Calc_CSP2 = N_CSP2 / ((0.0808 / ((Water_Content_SE_CSP2 + lwc + soc) + 0.115)) + 0.372)
N0_SE_Calc_CSP2.to_csv(r'YOUR PATH TO MAKE/2023_CSP2_N0_SE_frompy.csv')
N0_error_CSP2 = N0_SE_Calc_CSP2 - N0_CSP2

#actually plot data with best fit line, slope, and r-squared
a_CSP2_1, b_CSP2_1 = np.polyfit(x_CSP2,y_CSP2,1)
plt.errorbar(x_CSP2, y_CSP2, xerr=p_CSP2_SE, yerr=N0_error_CSP2, fmt='o', color='red', ecolor='gray', capsize=3)
plt.text(-0.1, 2323, u"r\u00b2 = {:0.2f}\nslope p < 0.001\nintercept p < 0.001".format(R_sq_CSP2), size=9)
plt.plot(x_CSP2, a_CSP2_1*x_CSP2+b_CSP2_1, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP2 Gravimetric Data 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.ylim(2300, 3020)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2,r_nn_CSP2], [R_Slope_CI_CSP2,R_Intercept_CI_CSP2]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2023.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
###################CSP3#######################
##############################################
# CRNS Distance Only #
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP3_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP3_CRNSDistance = GWC_CSP3_CRNSDistance['CRNS_SampleDepth']
a_CSP3_CRNSDistance = SWC_CSP3_CRNSDistance

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_CRNSDistance = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_CRNSDistance = NPIV_CSP3_CRNSDistance['NPIV']

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_CRNSDistance = a_CSP3_CRNSDistance
lwc = 0.058
soc = 0.0092
N_CSP3_CRNSDistance = NPIVCount_CSP3_CRNSDistance

#use parameters to calculate N0
N0_CSP3_CRNSDistance = N_CSP3_CRNSDistance / ((0.0808 / ((pwc_CSP3_CRNSDistance + lwc + soc) + 0.115)) + 0.372)
N0_CSP3_CRNSDistance.to_csv(r'YOUR PATH TO MAKE/2023_CSP3N0.csv')

### GET BWE DATA FOR PLOTTING ###

#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_CRNSDistance = BWEcsv_CSP3_CRNSDistance['BWE']

#define plot variables
x_CSP3_CRNSDistance = BWE_CSP3_CRNSDistance
y_CSP3_CRNSDistance = N0_CSP3_CRNSDistance

#calculate r-squared
corr_matrix_CSP3_CRNSDistance = np.corrcoef(x_CSP3_CRNSDistance,y_CSP3_CRNSDistance)
corr_CSP3_CRNSDistance = corr_matrix_CSP3_CRNSDistance[0,1]
R_sq_CSP3_CRNSDistance = corr_CSP3_CRNSDistance**2

#Least square regression
A_CSP3_23_CRNSDistance = np.vstack([x_CSP3_CRNSDistance, np.ones(len(x_CSP3_CRNSDistance))]).T
y_CSP3_23_CRNSDistance = np.array(y_CSP3_CRNSDistance) 
y_CSP3_23_CRNSDistance = y_CSP3_CRNSDistance.values[:, np.newaxis]
alpha_CSP3_CRNSDistance = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_23_CRNSDistance.T,A_CSP3_23_CRNSDistance)),A_CSP3_23_CRNSDistance.T)),y_CSP3_CRNSDistance)
m_CSP3_CRNSDistance = alpha_CSP3_CRNSDistance[0]
n_CSP3_CRNSDistance = alpha_CSP3_CRNSDistance[1]
mm_CSP3_CRNSDistance = float(m_CSP3_CRNSDistance)
mmm_CSP3_CRNSDistance = [float(m_CSP3_CRNSDistance)]
nn_CSP3_CRNSDistance = float(n_CSP3_CRNSDistance)
r_mm_CSP3_CRNSDistance = round(mm_CSP3_CRNSDistance,2)
r_nn_CSP3_CRNSDistance = round(nn_CSP3_CRNSDistance,2)

#95% Confidence interval on slope and intercept
z_CSP3_CRNSDistance = sm.add_constant(x_CSP3_CRNSDistance)
model_CSP3_CRNSDistance = sm.OLS(y_CSP3_CRNSDistance, z_CSP3_CRNSDistance).fit()
confidence_intervals_CSP3_CRNSDistance = model_CSP3_CRNSDistance.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_CRNSDistance, upper_bound_intercept_CSP3_CRNSDistance = confidence_intervals_CSP3_CRNSDistance.loc['const']
lower_bound_slope_CSP3_CRNSDistance, upper_bound_slope_CSP3_CRNSDistance = confidence_intervals_CSP3_CRNSDistance.loc['BWE']
Slope_CI_CSP3_CRNSDistance = mm_CSP3_CRNSDistance - (lower_bound_slope_CSP3_CRNSDistance)
Intercept_CI_CSP3_CRNSDistance = nn_CSP3_CRNSDistance - (lower_bound_intercept_CSP3_CRNSDistance)
R_Slope_CI_CSP3_CRNSDistance = round(Slope_CI_CSP3_CRNSDistance, 2)
R_Intercept_CI_CSP3_CRNSDistance = round(Intercept_CI_CSP3_CRNSDistance, 2)

#P_value for the least squares regression
p_values_CSP3_CRNSDistance = model_CSP3_CRNSDistance.pvalues
p_value_slope_CSP3_CRNSDistance = p_values_CSP3_CRNSDistance.loc['BWE']
p_value_int_CSP3_CRNSDistance = p_values_CSP3_CRNSDistance.loc['const']
r_p_value_slope_CSP3_CRNSDistance = "{:.1e}".format(p_value_slope_CSP3_CRNSDistance)
r_p_value_int_CSP3_CRNSDistance = "{:.1e}".format(p_value_int_CSP3_CRNSDistance)
p_slope_CSP3_CRNSDistance = float(r_p_value_slope_CSP3_CRNSDistance)
p_int_CSP3_CRNSDistance = float(r_p_value_int_CSP3_CRNSDistance)

#OLS model
model_CSP3_CRNSDistance = sm.OLS(y_CSP3_CRNSDistance, z_CSP3_CRNSDistance).fit()

#BWE standard error
BWESE_CSP3_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_BWE_SE.csv')
SE_CSP3_CRNSDistance = BWESE_CSP3_CRNSDistance['SE']
p_CSP3_SE_CRNSDistance = 2* SE_CSP3_CRNSDistance

#N0 standard error
N0_SE_CSP3_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_N0_SE.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP3_CRNSDistance = (2* N0_SE_CSP3_CRNSDistance['SE_CrnsDistance']) + pwc_CSP3_CRNSDistance

#use parameters to calculate N0
N0_SE_Calc_CSP3_CRNSDistance = N_CSP3_CRNSDistance / ((0.0808 / ((Water_Content_SE_CSP3_CRNSDistance + lwc + soc) + 0.115)) + 0.372)
N0_error_CSP3_CRNSDistance = N0_SE_Calc_CSP3_CRNSDistance - N0_CSP3_CRNSDistance

#actually plot data with best fit line, slope, and r-squared
a_CSP3_1_CRNSDistance, b_CSP3_1_CRNSDistance = np.polyfit(x_CSP3_CRNSDistance,y_CSP3_CRNSDistance,1)
plt.errorbar(x_CSP3_CRNSDistance, y_CSP3_CRNSDistance, xerr=p_CSP3_SE_CRNSDistance, yerr=N0_error_CSP3_CRNSDistance, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP3_CRNSDistance, a_CSP3_1_CRNSDistance*x_CSP3_CRNSDistance+b_CSP3_1_CRNSDistance, linestyle = '-', color = 'blue')
plt.text(-0.22, 1445, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP3_CRNSDistance, r_p_value_slope_CSP3_CRNSDistance), size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP3 Gravimetric Data 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(50))
plt.ylim(1440, 1675)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_CRNSDistance,r_nn_CSP3_CRNSDistance], [R_Slope_CI_CSP3_CRNSDistance,R_Intercept_CI_CSP3_CRNSDistance]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2023.png', dpi=300, bbox_inches='tight')
plt.show()


#############################################
#############################################
######## Schrön Depth Weighted ##############
#############################################
#############################################

##############################################
###################CSP1#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP1_weighted_schron = pd.read_csv('YOUR PATH/CSP1_Schron_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP1_schron = GWC_CSP1_weighted_schron['SWC']
a_CSP1_weighted_schron = SWC_CSP1_schron

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_weighted_schron = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_weighted_schron = NPIV_CSP1_weighted_schron['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_weighted_schron = a_CSP1_weighted_schron
lwc = 0.058
soc = 0.0092
N_CSP1_weighted_schron = NPIVCount_CSP1_weighted_schron

#use parameters to calculate N0
N0_CSP1_weighted_schron = N_CSP1_weighted_schron / ((0.0808 / ((pwc_CSP1_weighted_schron + lwc + soc) + 0.115)) + 0.372)
N0_CSP1_weighted_schron.to_csv(r'YOUR PATH TO MAKE/2023_CSP1_SchrönN0.csv')

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_weighted_schron = BWEcsv_CSP1_weighted_schron['BWE']

#define plot variables
x_CSP1_weighted_schron = BWE_CSP1_weighted_schron
y_CSP1_weighted_schron = N0_CSP1_weighted_schron

#calculate r-squared
corr_matrix_CSP1_weighted_schron = np.corrcoef(x_CSP1_weighted_schron,y_CSP1_weighted_schron)
corr_CSP1_weighted_schron = corr_matrix_CSP1_weighted_schron[0,1]
R_sq_CSP1_weighted_schron = corr_CSP1_weighted_schron**2

#Least square regression
A_CSP1_23_weighted_schron = np.vstack([x_CSP1_weighted_schron, np.ones(len(x_CSP1_weighted_schron))]).T
y_CSP1_23_weighted_schron = np.array(y_CSP1_weighted_schron)  
y_CSP1_23_weighted_schron = y_CSP1_weighted_schron.values[:, np.newaxis]
alpha_CSP1_weighted_schron = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_23_weighted_schron.T,A_CSP1_23_weighted_schron)),A_CSP1_23_weighted_schron.T)),y_CSP1_weighted_schron)
m_CSP1_weighted_schron = alpha_CSP1_weighted_schron[0]
n_CSP1_weighted_schron = alpha_CSP1_weighted_schron[1]
mm_CSP1_weighted_schron = float(m_CSP1_weighted_schron)
mmm_CSP1_weighted_schron = [float(m_CSP1_weighted_schron)]
nn_CSP1_weighted_schron = float(n_CSP1_weighted_schron)
r_mm_CSP1_weighted_schron = round(mm_CSP1_weighted_schron,2)
r_nn_CSP1_weighted_schron = round(nn_CSP1_weighted_schron,2)

#95% Confidence interval on slope and intercept
z_CSP1_weighted_schron = sm.add_constant(x_CSP1_weighted_schron)
model_CSP1_weighted_schron = sm.OLS(y_CSP1_weighted_schron, z_CSP1_weighted_schron).fit()
confidence_intervals_CSP1_weighted_schron = model_CSP1_weighted_schron.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_weighted_schron, upper_bound_intercept_CSP1_weighted_schron = confidence_intervals_CSP1_weighted_schron.loc['const']
lower_bound_slope_CSP1_weighted_schron, upper_bound_slope_CSP1_weighted_schron = confidence_intervals_CSP1_weighted_schron.loc['BWE']
Slope_CI_CSP1_weighted_schron = mm_CSP1_weighted_schron - (lower_bound_slope_CSP1_weighted_schron)
Intercept_CI_CSP1_weighted_schron = nn_CSP1_weighted_schron - (lower_bound_intercept_CSP1_weighted_schron)
R_Slope_CI_CSP1_weighted_schron = round(Slope_CI_CSP1_weighted_schron, 2)
R_Intercept_CI_CSP1_weighted_schron = round(Intercept_CI_CSP1_weighted_schron, 2)

#P_value for the least squares regression
p_values_CSP1_weighted_schron = model_CSP1_weighted_schron.pvalues
p_value_slope_CSP1_weighted_schron = p_values_CSP1_weighted_schron.loc['BWE']
p_value_int_CSP1_weighted_schron = p_values_CSP1_weighted_schron.loc['const']
r_p_value_slope_CSP1_weighted_schron = "{:.1e}".format(p_value_slope_CSP1_weighted_schron)
r_p_value_int_CSP1_weighted_schron = "{:.1e}".format(p_value_int_CSP1_weighted_schron)
p_slope_CSP1_weighted_schron = float(r_p_value_slope_CSP1_weighted_schron)
p_int_CSP1_weighted_schron = float(r_p_value_int_CSP1_weighted_schron)

#OLS model
model_CSP1_weighted_schron = sm.OLS(y_CSP1_weighted_schron, z_CSP1_weighted_schron).fit()

#BWE standard error
BWESE_CSP1_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP1_BWE_SE.csv')
SE_CSP1_weighted_schron = BWESE_CSP1_weighted_schron['SE']
p_CSP1_SE_weighted_schron = SE_CSP1_weighted_schron

#N0 standard error
N0_SE_CSP1_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP1_N0_SE_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP1_weighted_schron = (2 * N0_SE_CSP1_weighted_schron['SE']) + pwc_CSP1_weighted_schron

#use parameters to calculate N0
N0_SE_Calc_CSP1_weighted_schron = N_CSP1_weighted_schron / ((0.0808 / ((Water_Content_SE_CSP1_weighted_schron + lwc + soc) + 0.115)) + 0.372)
N0_error_CSP1_weighted_schron = N0_SE_Calc_CSP1_weighted_schron - N0_CSP1_weighted_schron

#actually plot data with best fit line, slope, and r-squared
a_CSP1_1_weighted_schron, b_CSP1_1_weighted_schron = np.polyfit(x_CSP1_weighted_schron,y_CSP1_weighted_schron,1)
plt.errorbar(x_CSP1_weighted_schron, y_CSP1_weighted_schron, xerr=p_CSP1_SE_weighted_schron, yerr=N0_error_CSP1_weighted_schron, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP1_weighted_schron, a_CSP1_1_weighted_schron*x_CSP1_weighted_schron+b_CSP1_1_weighted_schron, linestyle = '-', color = 'blue')
plt.text(-0.1, 2323, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP1_weighted_schron, r_p_value_slope_CSP1_weighted_schron), size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP1 Gravimetric Data 2023\nDepth Weighted by Schrön et al. 2017')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.ylim(2300, 3020)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_weighted_schron,r_nn_CSP1_weighted_schron], [R_Slope_CI_CSP1_weighted_schron,R_Intercept_CI_CSP1_weighted_schron]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_Schrön_Weighted.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
###################CSP2#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP2_weighted_schron = pd.read_csv('YOUR PATH/CSP2_Schron_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP2_schron = GWC_CSP2_weighted_schron['SWC']
a_CSP2_weighted_schron = SWC_CSP2_schron

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_weighted_schron = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_weighted_schron = NPIV_CSP2_weighted_schron['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_weighted_schron = a_CSP2_weighted_schron
lwc = 0.058
soc = 0.0092
N_CSP2_weighted_schron = NPIVCount_CSP2_weighted_schron

#use parameters to calculate N0
N0_CSP2_weighted_schron = N_CSP2_weighted_schron / ((0.0808 / ((pwc_CSP2_weighted_schron + lwc + soc) + 0.115)) + 0.372)
N0_CSP2_weighted_schron.to_csv(r'YOUR PATH TO MAKE/2023_CSP2_SchrönN0.csv')

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_weighted_schron = BWEcsv_CSP2_weighted_schron['BWE']

#define plot variables
x_CSP2_weighted_schron = BWE_CSP2_weighted_schron
y_CSP2_weighted_schron = N0_CSP2_weighted_schron

#calculate r-squared
corr_matrix_CSP2_weighted_schron = np.corrcoef(x_CSP2_weighted_schron,y_CSP2_weighted_schron)
corr_CSP2_weighted_schron = corr_matrix_CSP2_weighted_schron[0,1]
R_sq_CSP2_weighted_schron = corr_CSP2_weighted_schron**2

#Least square regression
A_CSP2_23_weighted_schron = np.vstack([x_CSP2_weighted_schron, np.ones(len(x_CSP2_weighted_schron))]).T
y_CSP2_23_weighted_schron = np.array(y_CSP2_weighted_schron)  
y_CSP2_23_weighted_schron = y_CSP2_weighted_schron.values[:, np.newaxis]
alpha_CSP2_weighted_schron = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_23_weighted_schron.T,A_CSP2_23_weighted_schron)),A_CSP2_23_weighted_schron.T)),y_CSP2_weighted_schron)
m_CSP2_weighted_schron = alpha_CSP2_weighted_schron[0]
n_CSP2_weighted_schron = alpha_CSP2_weighted_schron[1]
mm_CSP2_weighted_schron = float(m_CSP2_weighted_schron)
mmm_CSP2_weighted_schron = [float(m_CSP2_weighted_schron)]
nn_CSP2_weighted_schron = float(n_CSP2_weighted_schron)
r_mm_CSP2_weighted_schron = round(mm_CSP2_weighted_schron,2)
r_nn_CSP2_weighted_schron = round(nn_CSP2_weighted_schron,2)

#95% Confidence interval on slope and intercept
z_CSP2_weighted_schron = sm.add_constant(x_CSP2_weighted_schron)
model_CSP2_weighted_schron = sm.OLS(y_CSP2_weighted_schron, z_CSP2_weighted_schron).fit()
confidence_intervals_CSP2_weighted_schron = model_CSP2_weighted_schron.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_weighted_schron, upper_bound_intercept_CSP2_weighted_schron = confidence_intervals_CSP2_weighted_schron.loc['const']
lower_bound_slope_CSP2_weighted_schron, upper_bound_slope_CSP2_weighted_schron = confidence_intervals_CSP2_weighted_schron.loc['BWE']
Slope_CI_CSP2_weighted_schron = mm_CSP2_weighted_schron - (lower_bound_slope_CSP2_weighted_schron)
Intercept_CI_CSP2_weighted_schron = nn_CSP2_weighted_schron - (lower_bound_intercept_CSP2_weighted_schron)
R_Slope_CI_CSP2_weighted_schron = round(Slope_CI_CSP2_weighted_schron, 2)
R_Intercept_CI_CSP2_weighted_schron = round(Intercept_CI_CSP2_weighted_schron, 2)

#P_value for the least squares regression
p_values_CSP2_weighted_schron = model_CSP2_weighted_schron.pvalues
p_value_slope_CSP2_weighted_schron = p_values_CSP2_weighted_schron.loc['BWE']
p_value_int_CSP2_weighted_schron = p_values_CSP2_weighted_schron.loc['const']
r_p_value_slope_CSP2_weighted_schron = "{:.1e}".format(p_value_slope_CSP2_weighted_schron)
r_p_value_int_CSP2_weighted_schron = "{:.1e}".format(p_value_int_CSP2_weighted_schron)
p_slope_CSP2_weighted_schron = float(r_p_value_slope_CSP2_weighted_schron)
p_int_CSP2_weighted_schron = float(r_p_value_int_CSP2_weighted_schron)

#OLS model
model_CSP2_weighted_schron = sm.OLS(y_CSP2_weighted_schron, z_CSP2_weighted_schron).fit()

#BWE standard error
BWESE_CSP2_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP2_BWE_SE.csv')
SE_CSP2_weighted_schron = BWESE_CSP2_weighted_schron['SE']
p_CSP2_SE_weighted_schron = SE_CSP2_weighted_schron

#N0 standard error
N0_SE_CSP2_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP2_N0_SE_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP2_weighted_schron = (2 * N0_SE_CSP2_weighted_schron['SE']) + pwc_CSP2_weighted_schron

#use parameters to calculate N0
N0_SE_Calc_CSP2_weighted_schron = N_CSP2_weighted_schron / ((0.0808 / ((Water_Content_SE_CSP2_weighted_schron + lwc + soc) + 0.115)) + 0.372)
N0_error_CSP2_weighted_schron = N0_SE_Calc_CSP2_weighted_schron - N0_CSP2_weighted_schron

#actually plot data with best fit line, slope, and r-squared
a_CSP2_1_weighted_schron, b_CSP2_1_weighted_schron = np.polyfit(x_CSP2_weighted_schron,y_CSP2_weighted_schron,1)
plt.errorbar(x_CSP2_weighted_schron, y_CSP2_weighted_schron, xerr=p_CSP2_SE_weighted_schron, yerr=N0_error_CSP2_weighted_schron, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP2_weighted_schron, a_CSP2_1_weighted_schron*x_CSP2_weighted_schron+b_CSP2_1_weighted_schron, linestyle = '-', color = 'blue')
plt.text(-0.1, 2323, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP2_weighted_schron, r_p_value_slope_CSP2_weighted_schron), size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP2 Gravimetric Data 2023\nDepth Weighted by Schrön et al. 2017')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.ylim(2300, 3020)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_weighted_schron,r_nn_CSP2_weighted_schron], [R_Slope_CI_CSP2_weighted_schron,R_Intercept_CI_CSP2_weighted_schron]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_Schrön_Weighted.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
###################CSP3#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP3_weighted_schron = pd.read_csv('YOUR PATH/CSP3_Schron_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP3_schron = GWC_CSP3_weighted_schron['SWC']
a_CSP3_weighted_schron = SWC_CSP3_schron

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_weighted_schron = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_weighted_schron = NPIV_CSP3_weighted_schron['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_weighted_schron = a_CSP3_weighted_schron
lwc = 0.058
soc = 0.0092
N_CSP3_weighted_schron = NPIVCount_CSP3_weighted_schron

#use parameters to calculate N0
N0_CSP3_weighted_schron = N_CSP3_weighted_schron / ((0.0808 / ((pwc_CSP3_weighted_schron + lwc + soc) + 0.115)) + 0.372)
N0_CSP3_weighted_schron.to_csv(r'YOUR PATH TO MAKE/2023_CSP3_SchrönN0.csv')

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_weighted_schron = BWEcsv_CSP3_weighted_schron['BWE']

#define plot variables
x_CSP3_weighted_schron = BWE_CSP3_weighted_schron
y_CSP3_weighted_schron = N0_CSP3_weighted_schron

#calculate r-squared
corr_matrix_CSP3_weighted_schron = np.corrcoef(x_CSP3_weighted_schron,y_CSP3_weighted_schron)
corr_CSP3_weighted_schron = corr_matrix_CSP3_weighted_schron[0,1]
R_sq_CSP3_weighted_schron = corr_CSP3_weighted_schron**2

#Least square regression
A_CSP3_23_weighted_schron = np.vstack([x_CSP3_weighted_schron, np.ones(len(x_CSP3_weighted_schron))]).T
y_CSP3_23_weighted_schron = np.array(y_CSP3_weighted_schron)  
y_CSP3_23_weighted_schron = y_CSP3_weighted_schron.values[:, np.newaxis]
alpha_CSP3_weighted_schron = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_23_weighted_schron.T,A_CSP3_23_weighted_schron)),A_CSP3_23_weighted_schron.T)),y_CSP3_weighted_schron)
m_CSP3_weighted_schron = alpha_CSP3_weighted_schron[0]
n_CSP3_weighted_schron = alpha_CSP3_weighted_schron[1]
mm_CSP3_weighted_schron = float(m_CSP3_weighted_schron)
mmm_CSP3_weighted_schron = [float(m_CSP3_weighted_schron)]
nn_CSP3_weighted_schron = float(n_CSP3_weighted_schron)
r_mm_CSP3_weighted_schron = round(mm_CSP3_weighted_schron,2)
r_nn_CSP3_weighted_schron = round(nn_CSP3_weighted_schron,2)

#95% Confidence interval on slope and intercept
z_CSP3_weighted_schron = sm.add_constant(x_CSP3_weighted_schron)
model_CSP3_weighted_schron = sm.OLS(y_CSP3_weighted_schron, z_CSP3_weighted_schron).fit()
confidence_intervals_CSP3_weighted_schron = model_CSP3_weighted_schron.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_weighted_schron, upper_bound_intercept_CSP3_weighted_schron = confidence_intervals_CSP3_weighted_schron.loc['const']
lower_bound_slope_CSP3_weighted_schron, upper_bound_slope_CSP3_weighted_schron = confidence_intervals_CSP3_weighted_schron.loc['BWE']
Slope_CI_CSP3_weighted_schron = mm_CSP3_weighted_schron - (lower_bound_slope_CSP3_weighted_schron)
Intercept_CI_CSP3_weighted_schron = nn_CSP3_weighted_schron - (lower_bound_intercept_CSP3_weighted_schron)
R_Slope_CI_CSP3_weighted_schron = round(Slope_CI_CSP3_weighted_schron, 2)
R_Intercept_CI_CSP3_weighted_schron = round(Intercept_CI_CSP3_weighted_schron, 2)

#P_value for the least squares regression
p_values_CSP3_weighted_schron = model_CSP3_weighted_schron.pvalues
p_value_slope_CSP3_weighted_schron = p_values_CSP3_weighted_schron.loc['BWE']
p_value_int_CSP3_weighted_schron = p_values_CSP3_weighted_schron.loc['const']
r_p_value_slope_CSP3_weighted_schron = "{:.1e}".format(p_value_slope_CSP3_weighted_schron)
r_p_value_int_CSP3_weighted_schron = "{:.1e}".format(p_value_int_CSP3_weighted_schron)
p_slope_CSP3_weighted_schron = float(r_p_value_slope_CSP3_weighted_schron)
p_int_CSP3_weighted_schron = float(r_p_value_int_CSP3_weighted_schron)

#OLS model
model_CSP3_weighted_schron = sm.OLS(y_CSP3_weighted_schron, z_CSP3_weighted_schron).fit()

#BWE standard error
BWESE_CSP3_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP3_BWE_SE.csv')
SE_CSP3_weighted_schron = BWESE_CSP3_weighted_schron['SE']
p_CSP3_SE_weighted_schron = SE_CSP3_weighted_schron

#N0 standard error
N0_SE_CSP3_weighted_schron = pd.read_csv('YOUR PATH/2023_CSP3_N0_SE_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP3_weighted_schron = (2 * N0_SE_CSP3_weighted_schron['SE']) + pwc_CSP3_weighted_schron

#use parameters to calculate N0
N0_SE_Calc_CSP3_weighted_schron = N_CSP3_weighted_schron / ((0.0808 / ((Water_Content_SE_CSP3_weighted_schron + lwc + soc) + 0.115)) + 0.372)
N0_error_CSP3_weighted_schron = N0_SE_Calc_CSP3_weighted_schron - N0_CSP3_weighted_schron

#actually plot data with best fit line, slope, and r-squared
a_CSP3_1_weighted_schron, b_CSP3_1_weighted_schron = np.polyfit(x_CSP3_weighted_schron,y_CSP3_weighted_schron,1)
plt.errorbar(x_CSP3_weighted_schron, y_CSP3_weighted_schron, xerr=p_CSP3_SE_weighted_schron, yerr=N0_error_CSP3_weighted_schron, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP3_weighted_schron, a_CSP3_1_weighted_schron*x_CSP3_weighted_schron+b_CSP3_1_weighted_schron, linestyle = '-', color = 'blue')
plt.text(-0.1, 1445, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP3_weighted_schron, r_p_value_slope_CSP3_weighted_schron), size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP3 Gravimetric Data 2023\nDepth Weighted by Schrön et al. 2017')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.ylim(1440, 1675)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_weighted_schron,r_nn_CSP3_weighted_schron], [R_Slope_CI_CSP3_weighted_schron,R_Intercept_CI_CSP3_weighted_schron]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_Schrön_Weighted.png', dpi=300, bbox_inches='tight')
plt.show()


###############################################
#This script makes summary tables of gravimetric and TDR data
#The TDR data is imported below
###############Summary Tables##################
#Import TDR data to add to tables
#CSP1 2023#
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1_23 = pd.read_csv('YOUR PATH/AmerifluxCSP12023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP1_23 = GWC_CSP1_23['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP1_23 = GWC_CSP1_23['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP1_23 = GWC_CSP1_23['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP1_23 = GWC_CSP1_23['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP1_23 = GWC_CSP1_23['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP1_23 = GWC_CSP1_23['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP1_23 = GWC_CSP1_23['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP1_23 = GWC_CSP1_23['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP1_23 = GWC_CSP1_23['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP1_23 = GWC_CSP1_23['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP1_23 = GWC_CSP1_23['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP1_23 = GWC_CSP1_23['SWC_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1_23 = ((((SWC_PI_F_1_1_1_CSP1_23 + SWC_PI_F_2_1_1_CSP1_23 + SWC_PI_F_3_1_1_CSP1_23)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP1_23 + SWC_PI_F_2_2_1_CSP1_23 + SWC_PI_F_3_2_1_CSP1_23)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP1_23 = twosenwavgall_CSP1_23.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_23 = pd.read_csv('C:/Users/tessm/OneDrive/Documents/Documents/U of N/CRNS Research/CSP1Data/CSP12023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_23 = NPIV_CSP1_23['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP1_23 = NPIVCount_CSP1_23.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_23 = dailywavgSWC_CSP1_23
N_CSP1_23 = dailyNPIV_CSP1_23

#use parameters to calculate N0
N0_CSP1_23 = N_CSP1_23 / ((0.0808 / ((pwc_CSP1_23 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1_23 = N0_CSP1_23.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_23 = pd.read_csv('YOUR PATH/CSP12023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_23 = BWEcsv_CSP1_23['BWE']

#Interpolate between dates
BWEinterp_CSP1_23 = BWE_CSP1_23.interpolate(method = 'linear', limit_direction = 'forward')

BWEFillNaN_CSP1_23 = BWEinterp_CSP1_23.fillna(int(BWEinterp_CSP1_23.mean()), inplace=True)
N0plotdata_CSP1_23 = N0_CSP1_23.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP1_23 = BWEinterp_CSP1_23
y_CSP1_23 = N0interp_CSP1_23

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP1_23 = np.corrcoef(x_CSP1_23,y_CSP1_23)
corr_CSP1_23 = corr_matrix_CSP1_23[0,1]
R_sq_CSP1_23 = corr_CSP1_23**2

#Least square regression
A_CSP1_23 = np.vstack([x_CSP1_23, np.ones(len(x_CSP1_23))]).T
y_CSP1_23 = np.array(y_CSP1_23)
y_CSP1_23 = y_CSP1_23[:, np.newaxis]
alpha_CSP1_23 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_23.T,A_CSP1_23)),A_CSP1_23.T)),y_CSP1_23)
m_CSP1_23 = alpha_CSP1_23[0]
n_CSP1_23 = alpha_CSP1_23[1]
mm_CSP1_23 = float(m_CSP1_23)
nn_CSP1_23 = float(n_CSP1_23)
r_mm_CSP1_23 = round(mm_CSP1_23,2)
r_nn_CSP1_23 = round(nn_CSP1_23,2)

#95% Confidence interval on slope and intercept
z_CSP1_23 = sm.add_constant(x_CSP1_23)
model_CSP1_23 = sm.OLS(y_CSP1_23, z_CSP1_23).fit()
confidence_intervals_CSP1_23 = model_CSP1_23.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_23, upper_bound_intercept_CSP1_23 = confidence_intervals_CSP1_23.loc['const']
lower_bound_slope_CSP1_23, upper_bound_slope_CSP1_23 = confidence_intervals_CSP1_23.loc['BWE']
Slope_CI_CSP1_23 = mm_CSP1_23 - (lower_bound_slope_CSP1_23)
Intercept_CI_CSP1_23 = nn_CSP1_23 - (lower_bound_intercept_CSP1_23)
R_Slope_CI_CSP1_23 = round(Slope_CI_CSP1_23, 2)
R_Intercept_CI_CSP1_23 = round(Intercept_CI_CSP1_23, 2)

#CSP2 2023#
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP2_23 = pd.read_csv('YOUR PATH/AmerifluxCSP22023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP2_23 = GWC_CSP2_23['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP2_23 = GWC_CSP2_23['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP2_23 = GWC_CSP2_23['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP2_23 = GWC_CSP2_23['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP2_23 = GWC_CSP2_23['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP2_23 = GWC_CSP2_23['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP2_23 = GWC_CSP2_23['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP2_23 = GWC_CSP2_23['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP2_23 = GWC_CSP2_23['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP2_23 = GWC_CSP2_23['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP2_23 = GWC_CSP2_23['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP2_23 = GWC_CSP2_23['SWC_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2_23 = ((((SWC_PI_F_1_1_1_CSP2_23 + SWC_PI_F_2_1_1_CSP2_23 + SWC_PI_F_3_1_1_CSP2_23)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP2_23 + SWC_PI_F_2_2_1_CSP2_23 + SWC_PI_F_3_2_1_CSP2_23)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP2_23 = twosenwavgall_CSP2_23.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_23 = pd.read_csv('YOUR PATH/CSP22023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_23 = NPIV_CSP2_23['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP2_23 = NPIVCount_CSP2_23.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_23 = dailywavgSWC_CSP2_23
N_CSP2_23 = dailyNPIV_CSP2_23

#use parameters to calculate N0
N0_CSP2_23 = N_CSP2_23 / ((0.0808 / ((pwc_CSP2_23 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2_23 = N0_CSP2_23.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_23 = pd.read_csv('YOUR PATH/CSP22023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_23 = BWEcsv_CSP2_23['BWE']

#Interpolate between dates
BWEinterp_CSP2_23 = BWE_CSP2_23.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP2_23 = BWEinterp_CSP2_23.fillna(int(BWEinterp_CSP2_23.mean()), inplace=True)
N0plotdata_CSP2_23 = N0_CSP2_23.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP2_23 = BWEinterp_CSP2_23
y_CSP2_23 = N0interp_CSP2_23

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP2_23 = np.corrcoef(x_CSP2_23,y_CSP2_23)
corr_CSP2_23 = corr_matrix_CSP2_23[0,1]
R_sq_CSP2_23 = corr_CSP2_23**2

#Least square regression
A_CSP2_23 = np.vstack([x_CSP2_23, np.ones(len(x_CSP2_23))]).T
y_CSP2_23 = np.array(y_CSP2_23)
y_CSP2_23 = y_CSP2_23[:, np.newaxis]
alpha_CSP2_23 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_23.T,A_CSP2_23)),A_CSP2_23.T)),y_CSP2_23)
m_CSP2_23 = alpha_CSP2_23[0]
n_CSP2_23 = alpha_CSP2_23[1]
mm_CSP2_23 = float(m_CSP2_23)
nn_CSP2_23 = float(n_CSP2_23)
r_mm_CSP2_23 = round(mm_CSP2_23,2)
r_nn_CSP2_23 = round(nn_CSP2_23,2)

#95% Confidence interval on slope and intercept
z_CSP2_23 = sm.add_constant(x_CSP2_23)
model_CSP2_23 = sm.OLS(y_CSP2_23, z_CSP2_23).fit()
confidence_intervals_CSP2_23 = model_CSP2_23.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_23, upper_bound_intercept_CSP2_23 = confidence_intervals_CSP2_23.loc['const']
lower_bound_slope_CSP2_23, upper_bound_slope_CSP2_23 = confidence_intervals_CSP2_23.loc['BWE']
Slope_CI_CSP2_23 = mm_CSP2_23 - (lower_bound_slope_CSP2_23)
Intercept_CI_CSP2_23 = nn_CSP2_23 - (lower_bound_intercept_CSP2_23)
R_Slope_CI_CSP2_23 = round(Slope_CI_CSP2_23, 2)
R_Intercept_CI_CSP2_23 = round(Intercept_CI_CSP2_23, 2)


#CSP3 2023#
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_23 = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_23 = GWC_CSP3_23['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_23 = GWC_CSP3_23['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_23 = GWC_CSP3_23['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_23 = GWC_CSP3_23['SWC_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_23 = GWC_CSP3_23['SWC_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_23 = GWC_CSP3_23['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_23 = GWC_CSP3_23['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_23 = GWC_CSP3_23['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_23 = GWC_CSP3_23['SWC_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_23 = GWC_CSP3_23['SWC_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_23 = GWC_CSP3_23['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_23 = GWC_CSP3_23['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_23 = GWC_CSP3_23['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_23 = GWC_CSP3_23['SWC_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_23 = GWC_CSP3_23['SWC_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_23 = GWC_CSP3_23['SWC_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_23 = GWC_CSP3_23['SWC_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_23 = GWC_CSP3_23['SWC_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_23 = GWC_CSP3_23['SWC_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_23 = GWC_CSP3_23['SWC_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_23 = ((((SWC_PI_F_1_1_1_CSP3_23 + SWC_PI_F_2_1_1_CSP3_23 + SWC_PI_F_3_1_1_CSP3_23 + SWC_PI_F_4_1_1_CSP3_23)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_23 + SWC_PI_F_2_2_1_CSP3_23 + SWC_PI_F_3_2_1_CSP3_23 + SWC_PI_F_4_2_1_CSP3_23)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_23 = twosenwavgall_CSP3_23.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_23 = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_23 = NPIV_CSP3_23['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_23 = NPIVCount_CSP3_23.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_23 = dailywavgSWC_CSP3_23
lwc = 0.058
soc = 0.0092
N_CSP3_23 = dailyNPIV_CSP3_23

#use parameters to calculate N0
N0_CSP3_23 = N_CSP3_23 / ((0.0808 / ((pwc_CSP3_23 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_23 = N0_CSP3_23.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_23 = pd.read_csv('YOUR PATH/CSP32023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_23 = BWEcsv_CSP3_23['BWE']

#Interpolate between dates
BWEinterp_CSP3_23 = BWE_CSP3_23.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_23 = BWEinterp_CSP3_23.fillna(int(BWEinterp_CSP3_23.mean()), inplace=True)
N0plotdata_CSP3_23 = N0_CSP3_23.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_23 = BWEinterp_CSP3_23
y_CSP3_23 = N0interp_CSP3_23

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_23 = np.corrcoef(x_CSP3_23,y_CSP3_23)
corr_CSP3_23 = corr_matrix_CSP3_23[0,1]
R_sq_CSP3_23 = corr_CSP3_23**2

#Least square regression
A_CSP3_23 = np.vstack([x_CSP3_23, np.ones(len(x_CSP3_23))]).T
y_CSP3_23 = np.array(y_CSP3_23)
y_CSP3_23 = y_CSP3_23[:, np.newaxis]
alpha_CSP3_23 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_23.T,A_CSP3_23)),A_CSP3_23.T)),y_CSP3_23)
m_CSP3_23 = alpha_CSP3_23[0]
n_CSP3_23 = alpha_CSP3_23[1]
mm_CSP3_23 = float(m_CSP3_23)
nn_CSP3_23 = float(n_CSP3_23)
r_mm_CSP3_23 = round(mm_CSP3_23,2)
r_nn_CSP3_23 = round(nn_CSP3_23,2)

#95% Confidence interval on slope and intercept
z_CSP3_23 = sm.add_constant(x_CSP3_23)
model_CSP3_23 = sm.OLS(y_CSP3_23, z_CSP3_23).fit()
confidence_intervals_CSP3_23 = model_CSP3_23.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_23, upper_bound_intercept_CSP3_23 = confidence_intervals_CSP3_23.loc['const']
lower_bound_slope_CSP3_23, upper_bound_slope_CSP3_23 = confidence_intervals_CSP3_23.loc['BWE']
Slope_CI_CSP3_23 = mm_CSP3_23 - (lower_bound_slope_CSP3_23)
Intercept_CI_CSP3_23 = nn_CSP3_23 - (lower_bound_intercept_CSP3_23)
R_Slope_CI_CSP3_23 = round(Slope_CI_CSP3_23, 2)
R_Intercept_CI_CSP3_23 = round(Intercept_CI_CSP3_23, 2)


####### Weighted by other method #########

################################
#####DEPTH WEIGHTED AVERAGE#####
####Using 5-15cm and 20-30cm####
##2 samples used for each depth#
##############################################
###################CSP1#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP1_weighted = pd.read_csv('YOUR PATH/2023_CSP1_SWC_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP1_Upper = GWC_CSP1_weighted['Upper']
SWC_CSP1_Lower = GWC_CSP1_weighted['Lower']
SWC_CSP1_Weighted = ((SWC_CSP1_Upper * 0.75) + (SWC_CSP1_Lower * 0.25))
a_CSP1_weighted = SWC_CSP1_Weighted

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_weighted = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_weighted = NPIV_CSP1_weighted['NPIV']

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_weighted = a_CSP1_weighted
lwc = 0.058
soc = 0.0092
N_CSP1_weighted = NPIVCount_CSP1_weighted

#use parameters to calculate N0
N0_CSP1_weighted = N_CSP1_weighted / ((0.0808 / ((pwc_CSP1_weighted + lwc + soc) + 0.115)) + 0.372)

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_weighted = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_weighted = BWEcsv_CSP1_weighted['BWE']

#define plot variables
x_CSP1_weighted = BWE_CSP1_weighted
y_CSP1_weighted = N0_CSP1_weighted

#calculate r-squared
corr_matrix_CSP1_weighted = np.corrcoef(x_CSP1_weighted,y_CSP1_weighted)
corr_CSP1_weighted = corr_matrix_CSP1_weighted[0,1]
R_sq_CSP1_weighted = corr_CSP1_weighted**2

#Least square regression
A_CSP1_23_weighted = np.vstack([x_CSP1_weighted, np.ones(len(x_CSP1_weighted))]).T
y_CSP1_23_weighted = np.array(y_CSP1_weighted)  
y_CSP1_23_weighted = y_CSP1_weighted.values[:, np.newaxis]
alpha_CSP1_weighted = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_23_weighted.T,A_CSP1_23_weighted)),A_CSP1_23_weighted.T)),y_CSP1_weighted)
m_CSP1_weighted = alpha_CSP1_weighted[0]
n_CSP1_weighted = alpha_CSP1_weighted[1]
mm_CSP1_weighted = float(m_CSP1_weighted)
mmm_CSP1_weighted = [float(m_CSP1_weighted)]
nn_CSP1_weighted = float(n_CSP1_weighted)
r_mm_CSP1_weighted = round(mm_CSP1_weighted,2)
r_nn_CSP1_weighted = round(nn_CSP1_weighted,2)

#95% Confidence interval on slope and intercept
z_CSP1_weighted = sm.add_constant(x_CSP1_weighted)
model_CSP1_weighted = sm.OLS(y_CSP1_weighted, z_CSP1_weighted).fit()
confidence_intervals_CSP1_weighted = model_CSP1_weighted.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_weighted, upper_bound_intercept_CSP1_weighted = confidence_intervals_CSP1_weighted.loc['const']
lower_bound_slope_CSP1_weighted, upper_bound_slope_CSP1_weighted = confidence_intervals_CSP1_weighted.loc['BWE']
Slope_CI_CSP1_weighted = mm_CSP1_weighted - (lower_bound_slope_CSP1_weighted)
Intercept_CI_CSP1_weighted = nn_CSP1_weighted - (lower_bound_intercept_CSP1_weighted)
R_Slope_CI_CSP1_weighted = round(Slope_CI_CSP1_weighted, 2)
R_Intercept_CI_CSP1_weighted = round(Intercept_CI_CSP1_weighted, 2)

#P_value for the least squares regression
p_values_CSP1_weighted = model_CSP1_weighted.pvalues
p_value_slope_CSP1_weighted = p_values_CSP1_weighted.loc['BWE']
p_value_int_CSP1_weighted = p_values_CSP1_weighted.loc['const']
r_p_value_slope_CSP1_weighted = "{:.1e}".format(p_value_slope_CSP1_weighted)
r_p_value_int_CSP1_weighted = "{:.1e}".format(p_value_int_CSP1_weighted)
p_slope_CSP1_weighted = float(r_p_value_slope_CSP1_weighted)
p_int_CSP1_weighted = float(r_p_value_int_CSP1_weighted)

#OLS model
model_CSP1_weighted = sm.OLS(y_CSP1_weighted, z_CSP1_weighted).fit()

#BWE standard error
BWESE_CSP1_weighted = pd.read_csv('YOUR PATH/2023_CSP1_BWE_SE.csv')
SE_CSP1_weighted = BWESE_CSP1_weighted['SE']
p_CSP1_SE_weighted = SE_CSP1_weighted

#N0 standard error
N0_SE_CSP1_weighted = pd.read_csv('YOUR PATH/2023_CSP1_N0_SE_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP1_weighted = (2 * N0_SE_CSP1_weighted['SE']) + pwc_CSP1_weighted

#use parameters to calculate N0
N0_SE_Calc_CSP1_weighted = N_CSP1_weighted / ((0.0808 / ((Water_Content_SE_CSP1_weighted + lwc + soc) + 0.115)) + 0.372)
N0_error_CSP1_weighted = N0_SE_Calc_CSP1_weighted - N0_CSP1_weighted

#actually plot data with best fit line, slope, and r-squared
a_CSP1_1_weighted, b_CSP1_1_weighted = np.polyfit(x_CSP1_weighted,y_CSP1_weighted,1)
plt.errorbar(x_CSP1_weighted, y_CSP1_weighted, xerr=p_CSP1_SE_weighted, yerr=N0_error_CSP1_weighted, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP1_weighted, a_CSP1_1_weighted*x_CSP1_weighted+b_CSP1_1_weighted, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP1 Gravimetric 2023\n10cm, 25cm Depth Weighted')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.text(-0.1, 2323, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP1_weighted, r_p_value_slope_CSP1_weighted), size=9)
plt.ylim(2300, 3020)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_weighted,r_nn_CSP1_weighted], [R_Slope_CI_CSP1_weighted,R_Intercept_CI_CSP1_weighted]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_Weighted.png', dpi=300, bbox_inches='tight')
plt.show()

##############################################
###################CSP2#######################
##############################################
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP2_weighted = pd.read_csv('YOUR PATH/2023_CSP2_SWC_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP2_Upper = GWC_CSP2_weighted['Upper']
SWC_CSP2_Lower = GWC_CSP2_weighted['Lower']
SWC_CSP2_Weighted = ((SWC_CSP2_Upper * 0.75) + (SWC_CSP2_Lower * 0.25))
a_CSP2_weighted = SWC_CSP2_Weighted

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_weighted = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_weighted = NPIV_CSP2_weighted['NPIV']

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_weighted = a_CSP2_weighted
lwc = 0.058
soc = 0.0092
N_CSP2_weighted = NPIVCount_CSP2_weighted

#use parameters to calculate N0
N0_CSP2_weighted = N_CSP2_weighted / ((0.0808 / ((pwc_CSP2_weighted + lwc + soc) + 0.115)) + 0.372)

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_weighted = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_weighted = BWEcsv_CSP2_weighted['BWE']

#define plot variables
x_CSP2_weighted = BWE_CSP2_weighted
y_CSP2_weighted = N0_CSP2_weighted

#calculate r-squared
corr_matrix_CSP2_weighted = np.corrcoef(x_CSP2_weighted,y_CSP2_weighted)
corr_CSP2_weighted = corr_matrix_CSP2_weighted[0,1]
R_sq_CSP2_weighted = corr_CSP2_weighted**2

#Least square regression
A_CSP2_23_weighted = np.vstack([x_CSP2_weighted, np.ones(len(x_CSP2_weighted))]).T
y_CSP2_23_weighted = np.array(y_CSP2_weighted)  
y_CSP2_23_weighted = y_CSP2_weighted.values[:, np.newaxis]
alpha_CSP2_weighted = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_23_weighted.T,A_CSP2_23_weighted)),A_CSP2_23_weighted.T)),y_CSP2_weighted)
m_CSP2_weighted = alpha_CSP2_weighted[0]
n_CSP2_weighted = alpha_CSP2_weighted[1]
mm_CSP2_weighted = float(m_CSP2_weighted)
mmm_CSP2_weighted = [float(m_CSP2_weighted)]
nn_CSP2_weighted = float(n_CSP2_weighted)
r_mm_CSP2_weighted = round(mm_CSP2_weighted,2)
r_nn_CSP2_weighted = round(nn_CSP2_weighted,2)

#95% Confidence interval on slope and intercept
z_CSP2_weighted = sm.add_constant(x_CSP2_weighted)
model_CSP2_weighted = sm.OLS(y_CSP2_weighted, z_CSP2_weighted).fit()
confidence_intervals_CSP2_weighted = model_CSP2_weighted.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_weighted, upper_bound_intercept_CSP2_weighted = confidence_intervals_CSP2_weighted.loc['const']
lower_bound_slope_CSP2_weighted, upper_bound_slope_CSP2_weighted = confidence_intervals_CSP2_weighted.loc['BWE']
Slope_CI_CSP2_weighted = mm_CSP2_weighted - (lower_bound_slope_CSP2_weighted)
Intercept_CI_CSP2_weighted = nn_CSP2_weighted - (lower_bound_intercept_CSP2_weighted)
R_Slope_CI_CSP2_weighted = round(Slope_CI_CSP2_weighted, 2)
R_Intercept_CI_CSP2_weighted = round(Intercept_CI_CSP2_weighted, 2)

#P_value for the least squares regression
p_values_CSP2_weighted = model_CSP2_weighted.pvalues
p_value_slope_CSP2_weighted = p_values_CSP2_weighted.loc['BWE']
p_value_int_CSP2_weighted = p_values_CSP2_weighted.loc['const']
r_p_value_slope_CSP2_weighted = "{:.1e}".format(p_value_slope_CSP2_weighted)
r_p_value_int_CSP2_weighted = "{:.1e}".format(p_value_int_CSP2_weighted)
p_slope_CSP2_weighted = float(r_p_value_slope_CSP2_weighted)
p_int_CSP2_weighted = float(r_p_value_int_CSP2_weighted)

#OLS model
model_CSP2_weighted = sm.OLS(y_CSP2_weighted, z_CSP2_weighted).fit()

#BWE standard error
BWESE_CSP2_weighted = pd.read_csv('YOUR PATH/2023_CSP2_BWE_SE.csv')
SE_CSP2_weighted = BWESE_CSP2_weighted['SE']
p_CSP2_SE_weighted = SE_CSP2_weighted

#N0 standard error
N0_SE_CSP2_weighted = pd.read_csv('YOUR PATH/2023_CSP2_N0_SE_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP2_weighted = (2* N0_SE_CSP2_weighted['SE']) + pwc_CSP2_weighted

#use parameters to calculate N0
N0_SE_Calc_CSP2_weighted = N_CSP2_weighted / ((0.0808 / ((Water_Content_SE_CSP2_weighted + lwc + soc) + 0.115)) + 0.372)
N0_SE_Calc_CSP2_weighted.to_csv(r'YOUR PATH TO MAKE/2023_CSP2_N0_SE_frompy.csv')
N0_error_CSP2_weighted = N0_SE_Calc_CSP2_weighted - N0_CSP2_weighted

#actually plot data with best fit line, slope, and r-squared
a_CSP2_1_weighted, b_CSP2_1_weighted = np.polyfit(x_CSP2_weighted,y_CSP2_weighted,1)
plt.errorbar(x_CSP2_weighted, y_CSP2_weighted, xerr=p_CSP2_SE_weighted, yerr=N0_error_CSP2_weighted, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP2_weighted, a_CSP2_1_weighted*x_CSP2_weighted+b_CSP2_1_weighted, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP2 Gravimetric 2023\n10cm, 25cm Depth Weighted')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.text(-0.1, 2323, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP2_weighted, r_p_value_slope_CSP2_weighted), size=9)
plt.ylim(2300, 3020)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_weighted,r_nn_CSP2_weighted], [R_Slope_CI_CSP2_weighted,R_Intercept_CI_CSP2_weighted]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_Weighted.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
###################CSP3#######################
##############################################
# CRNS Distance Only #
### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP3_weighted_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_SWC_depth_weighted_CRNS_only.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP3_Upper_CRNSDistance = GWC_CSP3_weighted_CRNSDistance['Upper']
SWC_CSP3_Lower_CRNSDistance = GWC_CSP3_weighted_CRNSDistance['Lower']
SWC_CSP3_Weighted_CRNSDistance = ((SWC_CSP3_Upper_CRNSDistance * 0.75) + (SWC_CSP3_Lower_CRNSDistance * 0.25))
a_CSP3_weighted_CRNSDistance = SWC_CSP3_Weighted_CRNSDistance

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_weighted_CRNSDistance = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_weighted_CRNSDistance = NPIV_CSP3_weighted_CRNSDistance['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_weighted_CRNSDistance = a_CSP3_weighted_CRNSDistance
N_CSP3_weighted_CRNSDistance = NPIVCount_CSP3_weighted_CRNSDistance

#use parameters to calculate N0
N0_CSP3_weighted_CRNSDistance = N_CSP3_weighted_CRNSDistance / ((0.0808 / ((pwc_CSP3_weighted_CRNSDistance + lwc + soc) + 0.115)) + 0.372)

### GET BWE DATA FOR PLOTTING ###

#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_weighted_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_weighted_CRNSDistance = BWEcsv_CSP3_weighted_CRNSDistance['BWE']

#define plot variables
x_CSP3_weighted_CRNSDistance = BWE_CSP3_weighted_CRNSDistance
y_CSP3_weighted_CRNSDistance = N0_CSP3_weighted_CRNSDistance

#calculate r-squared
corr_matrix_CSP3_weighted_CRNSDistance = np.corrcoef(x_CSP3_weighted_CRNSDistance,y_CSP3_weighted_CRNSDistance)
corr_CSP3_weighted_CRNSDistance = corr_matrix_CSP3_weighted_CRNSDistance[0,1]
R_sq_CSP3_weighted_CRNSDistance = corr_CSP3_weighted_CRNSDistance**2

#Least square regression
A_CSP3_23_weighted_CRNSDistance = np.vstack([x_CSP3_weighted_CRNSDistance, np.ones(len(x_CSP3_weighted_CRNSDistance))]).T
y_CSP3_23_weighted_CRNSDistance = np.array(y_CSP3_weighted_CRNSDistance)  
y_CSP3_23_weighted_CRNSDistance = y_CSP3_weighted_CRNSDistance.values[:, np.newaxis]
alpha_CSP3_weighted_CRNSDistance = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_23_weighted_CRNSDistance.T,A_CSP3_23_weighted_CRNSDistance)),A_CSP3_23_weighted_CRNSDistance.T)),y_CSP3_weighted_CRNSDistance)
m_CSP3_weighted_CRNSDistance = alpha_CSP3_weighted_CRNSDistance[0]
n_CSP3_weighted_CRNSDistance = alpha_CSP3_weighted_CRNSDistance[1]
mm_CSP3_weighted_CRNSDistance = float(m_CSP3_weighted_CRNSDistance)
mmm_CSP3_weighted_CRNSDistance = [float(m_CSP3_weighted_CRNSDistance)]
nn_CSP3_weighted_CRNSDistance = float(n_CSP3_weighted_CRNSDistance)
r_mm_CSP3_weighted_CRNSDistance = round(mm_CSP3_weighted_CRNSDistance,2)
r_nn_CSP3_weighted_CRNSDistance = round(nn_CSP3_weighted_CRNSDistance,2)

#95% Confidence interval on slope and intercept
z_CSP3_weighted_CRNSDistance = sm.add_constant(x_CSP3_weighted_CRNSDistance)
model_CSP3_weighted_CRNSDistance = sm.OLS(y_CSP3_weighted_CRNSDistance, z_CSP3_weighted_CRNSDistance).fit()
confidence_intervals_CSP3_weighted_CRNSDistance = model_CSP3_weighted_CRNSDistance.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_weighted_CRNSDistance, upper_bound_intercept_CSP3_weighted_CRNSDistance = confidence_intervals_CSP3_weighted_CRNSDistance.loc['const']
lower_bound_slope_CSP3_weighted_CRNSDistance, upper_bound_slope_CSP3_weighted_CRNSDistance = confidence_intervals_CSP3_weighted_CRNSDistance.loc['BWE']
Slope_CI_CSP3_weighted_CRNSDistance = mm_CSP3_weighted_CRNSDistance - (lower_bound_slope_CSP3_weighted_CRNSDistance)
Intercept_CI_CSP3_weighted_CRNSDistance = nn_CSP3_weighted_CRNSDistance - (lower_bound_intercept_CSP3_weighted_CRNSDistance)
R_Slope_CI_CSP3_weighted_CRNSDistance = round(Slope_CI_CSP3_weighted_CRNSDistance, 2)
R_Intercept_CI_CSP3_weighted_CRNSDistance = round(Intercept_CI_CSP3_weighted_CRNSDistance, 2)

#P_value for the least squares regression
p_values_CSP3_weighted_CRNSDistance = model_CSP3_weighted_CRNSDistance.pvalues
p_value_slope_CSP3_weighted_CRNSDistance = p_values_CSP3_weighted_CRNSDistance.loc['BWE']
p_value_int_CSP3_weighted_CRNSDistance = p_values_CSP3_weighted_CRNSDistance.loc['const']
r_p_value_slope_CSP3_weighted_CRNSDistance = "{:.1e}".format(p_value_slope_CSP3_weighted_CRNSDistance)
r_p_value_int_CSP3_weighted_CRNSDistance = "{:.1e}".format(p_value_int_CSP3_weighted_CRNSDistance)
p_slope_CSP3_weighted_CRNSDistance = float(r_p_value_slope_CSP3_weighted_CRNSDistance)
p_int_CSP3_weighted_CRNSDistance = float(r_p_value_int_CSP3_weighted_CRNSDistance)

#OLS model
model_CSP3_weighted_CRNSDistance = sm.OLS(y_CSP3_weighted_CRNSDistance, z_CSP3_weighted_CRNSDistance).fit()

#BWE standard error
BWESE_CSP3_weighted_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_BWE_SE.csv')
SE_CSP3_weighted_CRNSDistance = BWESE_CSP3_weighted_CRNSDistance['SE']
p_CSP3_SE_weighted_CRNSDistance = 2* SE_CSP3_weighted_CRNSDistance

#N0 standard error
N0_SE_CSP3_weighted_CRNSDistance = pd.read_csv('YOUR PATH/2023_CSP3_N0_SE_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Water_Content_SE_CSP3_weighted_CRNSDistance = (2* N0_SE_CSP3_weighted_CRNSDistance['SE_Crns_Distance']) + pwc_CSP3_weighted_CRNSDistance

#use parameters to calculate N0
N0_SE_Calc_CSP3_weighted_CRNSDistance = N_CSP3_weighted_CRNSDistance / ((0.0808 / ((Water_Content_SE_CSP3_weighted_CRNSDistance + lwc + soc) + 0.115)) + 0.372)
N0_SE_Calc_CSP3_weighted_CRNSDistance.to_csv(r'YOUR PATH TO MAKE/2023_CSP3_N0_SE_frompy.csv')
N0_error_CSP3_weighted_CRNSDistance = N0_SE_Calc_CSP3_weighted_CRNSDistance - N0_CSP3_weighted_CRNSDistance

#actually plot data with best fit line, slope, and r-squared
a_CSP3_1_weighted_CRNSDistance, b_CSP3_1_weighted_CRNSDistance = np.polyfit(x_CSP3_weighted_CRNSDistance,y_CSP3_weighted_CRNSDistance,1)
plt.errorbar(x_CSP3_weighted_CRNSDistance, y_CSP3_weighted_CRNSDistance, xerr=p_CSP3_SE_weighted_CRNSDistance, yerr=N0_error_CSP3_weighted_CRNSDistance, fmt='o', color='red', ecolor='gray', capsize=3)
plt.plot(x_CSP3_weighted_CRNSDistance, a_CSP3_1_weighted_CRNSDistance*x_CSP3_weighted_CRNSDistance+b_CSP3_1_weighted_CRNSDistance, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title('CSP3 Gravimetric 2023\n10cm, 25cm Depth Weighted')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.text(-0.1, 1445, u"r\u00b2 = {:0.2f}\nslope p = {}\nintercept p < 0.001".format(R_sq_CSP3_weighted_CRNSDistance, r_p_value_slope_CSP3_weighted_CRNSDistance), size=9)
plt.gca().yaxis.set_major_locator(MultipleLocator(50))
plt.ylim(1440, 1675)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_weighted_CRNSDistance,r_nn_CSP3_weighted_CRNSDistance], [R_Slope_CI_CSP3_weighted_CRNSDistance,R_Intercept_CI_CSP3_weighted_CRNSDistance]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_Weighted.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
############ Table of Values #################
################# Slope ######################
##############################################

data = {
    '': ['Arithmetic\nSlope', 'Schrön\nSlope', 'TDR\nSlope'],
    'CSP1': [f"{r_mm_CSP1} +/- {R_Slope_CI_CSP1}", f"{r_mm_CSP1_weighted_schron} +/- {R_Slope_CI_CSP1_weighted_schron}", f"{r_mm_CSP1_23} +/- {R_Slope_CI_CSP1_23}"],
    'CSP2': [f"{r_mm_CSP2} +/- {R_Slope_CI_CSP2}", f"{r_mm_CSP2_weighted_schron} +/- {R_Slope_CI_CSP2_weighted_schron}", f"{r_mm_CSP2_23} +/- {R_Slope_CI_CSP2_23}"],
    'CSP3': [f"{r_mm_CSP3_CRNSDistance} +/- {R_Slope_CI_CSP3_CRNSDistance}", f"{r_mm_CSP3_weighted_schron} +/- {R_Slope_CI_CSP3_weighted_schron}", f"{r_mm_CSP3_23} +/- {R_Slope_CI_CSP3_23}"]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)
# Function to render DataFrame as a table
def render_mpl_table(data, col_width=3.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Slope_Summary_Schrön.png', dpi=300, bbox_inches='tight')
plt.show()

##############################################
############## Intercept #####################
##############################################

data = {
    '': ['Arithmetic\nIntercept', 'Schrön\nIntercept', 'TDR\nIntercept'],
    'CSP1': [f"{r_nn_CSP1} +/- {R_Intercept_CI_CSP1}", f"{r_nn_CSP1_weighted_schron} +/- {R_Intercept_CI_CSP1_weighted_schron}", f"{r_nn_CSP1_23} +/- {R_Intercept_CI_CSP1_23}"],
    'CSP2': [f"{r_nn_CSP2} +/- {R_Intercept_CI_CSP2}", f"{r_nn_CSP2_weighted_schron} +/- {R_Intercept_CI_CSP2_weighted_schron}", f"{r_nn_CSP2_23} +/- {R_Intercept_CI_CSP2_23}"],
    'CSP3': [f"{r_nn_CSP3_CRNSDistance} +/- {R_Intercept_CI_CSP3_CRNSDistance}", f"{r_nn_CSP3_weighted_schron} +/- {R_Intercept_CI_CSP3_weighted_schron}", f"{r_nn_CSP3_23} +/- {R_Intercept_CI_CSP3_23}"]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Function to render DataFrame as a table
def render_mpl_table(data, col_width=3.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(14,4))
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Intercept_Summary_Schrön.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
############ Table of Values #################
################# Slope ######################
# All weighting methods #
##############################################

data = {
    '': ['Arithmetic\nSlope', 'Schrön\nSlope', '10cm, 25cm\nDepth Weighted', 'TDR\nSlope'],
    'CSP1': [f"{r_mm_CSP1} +/- {R_Slope_CI_CSP1}", f"{r_mm_CSP1_weighted_schron} +/- {R_Slope_CI_CSP1_weighted_schron}", f"{r_mm_CSP1_weighted} +/- {R_Slope_CI_CSP1_weighted}", f"{r_mm_CSP1_23} +/- {R_Slope_CI_CSP1_23}"],
    'CSP2': [f"{r_mm_CSP2} +/- {R_Slope_CI_CSP2}", f"{r_mm_CSP2_weighted_schron} +/- {R_Slope_CI_CSP2_weighted_schron}", f"{r_mm_CSP2_weighted} +/- {R_Slope_CI_CSP2_weighted}", f"{r_mm_CSP2_23} +/- {R_Slope_CI_CSP2_23}"],
    'CSP3': [f"{r_mm_CSP3_CRNSDistance} +/- {R_Slope_CI_CSP3_CRNSDistance}", f"{r_mm_CSP3_weighted_schron} +/- {R_Slope_CI_CSP3_weighted_schron}", f"{r_mm_CSP3_weighted_CRNSDistance} +/- {R_Slope_CI_CSP3_weighted_CRNSDistance}", f"{r_mm_CSP3_23} +/- {R_Slope_CI_CSP3_23}"]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)
# Function to render DataFrame as a table
def render_mpl_table(data, col_width=3.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Slope_Summary_Schrön.png', dpi=300, bbox_inches='tight')
plt.show()

##############################################
############## Intercept #####################
##############################################

data = {
    '': ['Arithmetic\nIntercept', 'Schrön\nIntercept', '10cm, 25cm\nDepth Weighted', 'TDR\nIntercept'],
    'CSP1': [f"{r_nn_CSP1} +/- {R_Intercept_CI_CSP1}", f"{r_nn_CSP1_weighted_schron} +/- {R_Intercept_CI_CSP1_weighted_schron}", f"{r_nn_CSP1_weighted} +/- {R_Intercept_CI_CSP1_weighted}", f"{r_nn_CSP1_23} +/- {R_Intercept_CI_CSP1_23}"],
    'CSP2': [f"{r_nn_CSP2} +/- {R_Intercept_CI_CSP2}", f"{r_nn_CSP2_weighted_schron} +/- {R_Intercept_CI_CSP2_weighted_schron}", f"{r_nn_CSP2_weighted} +/- {R_Intercept_CI_CSP2_weighted}", f"{r_nn_CSP2_23} +/- {R_Intercept_CI_CSP2_23}"],
    'CSP3': [f"{r_nn_CSP3_CRNSDistance} +/- {R_Intercept_CI_CSP3_CRNSDistance}", f"{r_nn_CSP3_weighted_schron} +/- {R_Intercept_CI_CSP3_weighted_schron}", f"{r_nn_CSP3_weighted_CRNSDistance} +/- {R_Intercept_CI_CSP3_weighted_CRNSDistance}", f"{r_nn_CSP3_23} +/- {R_Intercept_CI_CSP3_23}"]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Function to render DataFrame as a table
def render_mpl_table(data, col_width=3.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(14,4))
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Intercept_Summary_Schrön.png', dpi=300, bbox_inches='tight')
plt.show()
