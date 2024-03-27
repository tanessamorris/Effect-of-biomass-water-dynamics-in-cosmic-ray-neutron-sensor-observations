#import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given


#This script plots N0 vs BWE for each year of data for each CRNS site
#Additional bar charts for comparison are found below this, as well as confidence interval testing and a table of values

##################################################################
##################################################################
##################################################################
############################ CSP1 ################################
##################################################################
##################################################################
##################################################################

##################################################################
#############################2019#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1_19 = pd.read_csv('YOUR PATH/AmerifluxCSP12019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP1_19 = GWC_CSP1_19['SWC_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP1_19 = GWC_CSP1_19['SWC_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP1_19 = GWC_CSP1_19['SWC_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP1_19 = GWC_CSP1_19['SWC_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP1_19 = GWC_CSP1_19['SWC_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP1_19 = GWC_CSP1_19['SWC_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP1_19 = GWC_CSP1_19['SWC_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP1_19 = GWC_CSP1_19['SWC_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP1_19 = GWC_CSP1_19['SWC_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP1_19 = GWC_CSP1_19['SWC_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP1_19 = GWC_CSP1_19['SWC_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP1_19 = GWC_CSP1_19['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1_19 = ((((SWC_PI_F_1_1_1_CSP1_19 + SWC_PI_F_2_1_1_CSP1_19 + SWC_PI_F_3_1_1_CSP1_19)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP1_19 + SWC_PI_F_2_2_1_CSP1_19 + SWC_PI_F_3_2_1_CSP1_19)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP1_19 = twosenwavgall_CSP1_19.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_19 = pd.read_csv('YOUR PATH/CSP12019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_19 = NPIV_CSP1_19['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP1_19 = NPIVCount_CSP1_19.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_19 = dailywavgSWC_CSP1_19
lwc = 0.058
soc = 0.0092
N_CSP1_19 = dailyNPIV_CSP1_19

#use parameters to calculate N0
N0_CSP1_19 = N_CSP1_19 / ((0.0808 / ((pwc_CSP1_19 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1_19 = N0_CSP1_19.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_19 = pd.read_csv('YOUR PATH/CSP12019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_19 = BWEcsv_CSP1_19['BWE']

#Interpolate between dates
BWEinterp_CSP1_19 = BWE_CSP1_19.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP1_19 = BWEinterp_CSP1_19.fillna(int(BWEinterp_CSP1_19.mean()), inplace=True)
N0plotdata_CSP1_19 = N0_CSP1_19.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP1_19 = BWEinterp_CSP1_19
y_CSP1_19 = N0interp_CSP1_19

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP1_19 = np.corrcoef(x_CSP1_19,y_CSP1_19)
corr_CSP1_19 = corr_matrix_CSP1_19[0,1]
R_sq = corr_CSP1_19**2

#Least square regression
A_CSP1_19 = np.vstack([x_CSP1_19, np.ones(len(x_CSP1_19))]).T
y_CSP1_19 = np.array(y_CSP1_19)  # convert y_CSP1_19 to a numpy array
y_CSP1_19 = y_CSP1_19[:, np.newaxis]
alpha_CSP1_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_19.T,A_CSP1_19)),A_CSP1_19.T)),y_CSP1_19)
m_CSP1_19 = alpha_CSP1_19[0]
n_CSP1_19 = alpha_CSP1_19[1]
mm_CSP1_19 = float(m_CSP1_19)
mmm_CSP1_19 = [float(m_CSP1_19)]
nn_CSP1_19 = float(n_CSP1_19)
r_mm_CSP1_19 = round(mm_CSP1_19,2)
r_nn_CSP1_19 = round(nn_CSP1_19,2)

#95% Confidence interval on slope and intercept
z_CSP1_19 = sm.add_constant(x_CSP1_19)
model_CSP1_19 = sm.OLS(y_CSP1_19, z_CSP1_19).fit()
confidence_intervals_CSP1_19 = model_CSP1_19.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_19, upper_bound_intercept_CSP1_19 = confidence_intervals_CSP1_19.loc['const']
lower_bound_slope_CSP1_19, upper_bound_slope_CSP1_19 = confidence_intervals_CSP1_19.loc['BWE']
Slope_CI_CSP1_19 = mm_CSP1_19 - (lower_bound_slope_CSP1_19)
Intercept_CI_CSP1_19 = nn_CSP1_19 - (lower_bound_intercept_CSP1_19)
R_Slope_CI_CSP1_19 = round(Slope_CI_CSP1_19, 2)
R_Intercept_CI_CSP1_19 = round(Intercept_CI_CSP1_19, 2)

#P_value for the least squares regression
p_values_CSP1_19 = model_CSP1_19.pvalues
p_value_slope_CSP1_19 = p_values_CSP1_19.loc['BWE']
p_value_int_CSP1_19 = p_values_CSP1_19.loc['const']
r_p_value_slope_CSP1_19 = "{:.1e}".format(p_value_slope_CSP1_19)
r_p_value_int_CSP1_19 = "{:.1e}".format(p_value_int_CSP1_19)
p_slope_CSP1_19 = float(r_p_value_slope_CSP1_19)
p_int_CSP1_19 = float(r_p_value_int_CSP1_19)

#OLS model
model_CSP1_19 = sm.OLS(y_CSP1_19, z_CSP1_19).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP1_19, y_CSP1_19, color = 'red')
plt.plot(x_CSP1_19, alpha_CSP1_19[0]*x_CSP1_19 + alpha_CSP1_19[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE 2019')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_19,r_nn_CSP1_19], [R_Slope_CI_CSP1_19,R_Intercept_CI_CSP1_19]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2019.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2020#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1_20 = pd.read_csv('YOUR PATH/AmerifluxCSP12020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP1_20 = GWC_CSP1_20['SWC_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP1_20 = GWC_CSP1_20['SWC_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP1_20 = GWC_CSP1_20['SWC_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP1_20 = GWC_CSP1_20['SWC_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP1_20 = GWC_CSP1_20['SWC_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP1_20 = GWC_CSP1_20['SWC_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP1_20 = GWC_CSP1_20['SWC_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP1_20 = GWC_CSP1_20['SWC_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP1_20 = GWC_CSP1_20['SWC_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP1_20 = GWC_CSP1_20['SWC_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP1_20 = GWC_CSP1_20['SWC_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP1_20 = GWC_CSP1_20['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1_20 = ((((SWC_PI_F_1_1_1_CSP1_20 + SWC_PI_F_2_1_1_CSP1_20 + SWC_PI_F_3_1_1_CSP1_20)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP1_20 + SWC_PI_F_2_2_1_CSP1_20 + SWC_PI_F_3_2_1_CSP1_20)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP1_20 = twosenwavgall_CSP1_20.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_20 = pd.read_csv('YOUR PATH/CSP12020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_20 = NPIV_CSP1_20['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP1_20 = NPIVCount_CSP1_20.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_20 = dailywavgSWC_CSP1_20
N_CSP1_20 = dailyNPIV_CSP1_20

#use parameters to calculate N0
N0_CSP1_20 = N_CSP1_20 / ((0.0808 / ((pwc_CSP1_20 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1_20 = N0_CSP1_20.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_20 = pd.read_csv('YOUR PATH/CSP12020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_20 = BWEcsv_CSP1_20['BWE']

#Interpolate between dates
BWEinterp_CSP1_20 = BWE_CSP1_20.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP1_20 = BWEinterp_CSP1_20.fillna(int(BWEinterp_CSP1_20.mean()), inplace=True)
N0plotdata_CSP1_20 = N0_CSP1_20.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP1_20 = BWEinterp_CSP1_20
y_CSP1_20 = N0interp_CSP1_20

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP1_20 = np.corrcoef(x_CSP1_20,y_CSP1_20)
corr_CSP1_20 = corr_matrix_CSP1_20[0,1]
R_sq_CSP1_20 = corr_CSP1_20**2

#Least square regression
A_CSP1_20 = np.vstack([x_CSP1_20, np.ones(len(x_CSP1_20))]).T
y_CSP1_20 = np.array(y_CSP1_20)
y_CSP1_20 = y_CSP1_20[:, np.newaxis]
alpha_CSP1_20 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_20.T,A_CSP1_20)),A_CSP1_20.T)),y_CSP1_20)
m_CSP1_20 = alpha_CSP1_20[0]
n_CSP1_20 = alpha_CSP1_20[1]
mm_CSP1_20 = float(m_CSP1_20)
mmm_CSP1_20 = [float(m_CSP1_20)]
nn_CSP1_20 = float(n_CSP1_20)
r_mm_CSP1_20 = round(mm_CSP1_20,2)
r_nn_CSP1_20 = round(nn_CSP1_20,2)

#95% Confidence interval on slope and intercept
z_CSP1_20 = sm.add_constant(x_CSP1_20)
model_CSP1_20 = sm.OLS(y_CSP1_20, z_CSP1_20).fit()
confidence_intervals_CSP1_20 = model_CSP1_20.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_20, upper_bound_intercept_CSP1_20 = confidence_intervals_CSP1_20.loc['const']
lower_bound_slope_CSP1_20, upper_bound_slope_CSP1_20 = confidence_intervals_CSP1_20.loc['BWE']
Slope_CI_CSP1_20 = mm_CSP1_20 - (lower_bound_slope_CSP1_20)
Intercept_CI_CSP1_20 = nn_CSP1_20 - (lower_bound_intercept_CSP1_20)
R_Slope_CI_CSP1_20 = round(Slope_CI_CSP1_20, 2)
R_Intercept_CI_CSP1_20 = round(Intercept_CI_CSP1_20, 2)

#P_value for the least squares regression
p_values_CSP1_20 = model_CSP1_20.pvalues
p_value_slope_CSP1_20 = p_values_CSP1_20.loc['BWE']
p_value_int_CSP1_20 = p_values_CSP1_20.loc['const']
r_p_value_slope_CSP1_20 = "{:.1e}".format(p_value_slope_CSP1_20)
r_p_value_int_CSP1_20 = "{:.1e}".format(p_value_int_CSP1_20)
p_slope_CSP1_20 = float(r_p_value_slope_CSP1_20)
p_int_CSP1_20 = float(r_p_value_int_CSP1_20)

#OLS model
model_CSP1_20 = sm.OLS(y_CSP1_20, z_CSP1_20).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP1_20, y_CSP1_20, color = 'red')
plt.plot(x_CSP1_20, alpha_CSP1_20[0]*x_CSP1_20 + alpha_CSP1_20[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP1_20), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE 2020')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_20,r_nn_CSP1_20], [R_Slope_CI_CSP1_20,R_Intercept_CI_CSP1_20]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.savefig('YOUR PATH TO SAVE/CSP1_2020.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2021#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1_21 = pd.read_csv('YOUR PATH/AmerifluxCSP12021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP1_21 = GWC_CSP1_21['SWC_PI_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1_21 = ((((SWC_PI_F_1_1_1_CSP1_21 + SWC_PI_F_2_1_1_CSP1_21 + SWC_PI_F_3_1_1_CSP1_21)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP1_21 + SWC_PI_F_2_2_1_CSP1_21 + SWC_PI_F_3_2_1_CSP1_21)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP1_21 = twosenwavgall_CSP1_21.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_21 = pd.read_csv('YOUR PATH/CSP12021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_21 = NPIV_CSP1_21['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP1_21 = NPIVCount_CSP1_21.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_21 = dailywavgSWC_CSP1_21
N_CSP1_21 = dailyNPIV_CSP1_21

#use parameters to calculate N0
N0_CSP1_21 = N_CSP1_21 / ((0.0808 / ((pwc_CSP1_21 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1_21 = N0_CSP1_21.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_21 = pd.read_csv('YOUR PATH/CSP12021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_21 = BWEcsv_CSP1_21['BWE']

#Interpolate between dates
BWEinterp_CSP1_21 = BWE_CSP1_21.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP1_21 = BWEinterp_CSP1_21.fillna(int(BWEinterp_CSP1_21.mean()), inplace=True)
N0plotdata_CSP1_21 = N0_CSP1_21.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP1_21 = BWEinterp_CSP1_21
y_CSP1_21 = N0interp_CSP1_21

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP1_21 = np.corrcoef(x_CSP1_21,y_CSP1_21)
corr_CSP1_21 = corr_matrix_CSP1_21[0,1]
R_sq_CSP1_21 = corr_CSP1_21**2

#Least square regression
A_CSP1_21 = np.vstack([x_CSP1_21, np.ones(len(x_CSP1_21))]).T
y_CSP1_21 = np.array(y_CSP1_21)
y_CSP1_21 = y_CSP1_21[:, np.newaxis]
alpha_CSP1_21 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_21.T,A_CSP1_21)),A_CSP1_21.T)),y_CSP1_21)
m_CSP1_21 = alpha_CSP1_21[0]
n_CSP1_21 = alpha_CSP1_21[1]
mm_CSP1_21 = float(m_CSP1_21)
mmm_CSP1_21 = [float(m_CSP1_21)]
nn_CSP1_21 = float(n_CSP1_21)
r_mm_CSP1_21 = round(mm_CSP1_21,2)
r_nn_CSP1_21 = round(nn_CSP1_21,2)

#95% Confidence interval on slope and intercept
z_CSP1_21 = sm.add_constant(x_CSP1_21)
model_CSP1_21 = sm.OLS(y_CSP1_21, z_CSP1_21).fit()
confidence_intervals_CSP1_21 = model_CSP1_21.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_21, upper_bound_intercept_CSP1_21 = confidence_intervals_CSP1_21.loc['const']
lower_bound_slope_CSP1_21, upper_bound_slope_CSP1_21 = confidence_intervals_CSP1_21.loc['BWE']
Slope_CI_CSP1_21 = mm_CSP1_21 - (lower_bound_slope_CSP1_21)
Intercept_CI_CSP1_21 = nn_CSP1_21 - (lower_bound_intercept_CSP1_21)
R_Slope_CI_CSP1_21 = round(Slope_CI_CSP1_21, 2)
R_Intercept_CI_CSP1_21 = round(Intercept_CI_CSP1_21, 2)

#P_value for the least squares regression
p_values_CSP1_21 = model_CSP1_21.pvalues
p_value_slope_CSP1_21 = p_values_CSP1_21.loc['BWE']
p_value_int_CSP1_21 = p_values_CSP1_21.loc['const']
r_p_value_slope_CSP1_21 = "{:.1e}".format(p_value_slope_CSP1_21)
r_p_value_int_CSP1_21 = "{:.1e}".format(p_value_int_CSP1_21)
p_slope_CSP1_21 = float(r_p_value_slope_CSP1_21)
p_int_CSP1_21 = float(r_p_value_int_CSP1_21)

#OLS model
model_CSP1_21 = sm.OLS(y_CSP1_21, z_CSP1_21).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP1_21, y_CSP1_21, color = 'red')
plt.plot(x_CSP1_21, alpha_CSP1_21[0]*x_CSP1_21 + alpha_CSP1_21[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP1_21), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE 2021')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_21,r_nn_CSP1_21], [R_Slope_CI_CSP1_21,R_Intercept_CI_CSP1_21]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2021.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2022#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1_22 = pd.read_csv('YOUR PATH/AmerifluxCSP12022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP1_22 = GWC_CSP1_22['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP1_22 = GWC_CSP1_22['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP1_22 = GWC_CSP1_22['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP1_22 = GWC_CSP1_22['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP1_22 = GWC_CSP1_22['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP1_22 = GWC_CSP1_22['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP1_22 = GWC_CSP1_22['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP1_22 = GWC_CSP1_22['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP1_22 = GWC_CSP1_22['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP1_22 = GWC_CSP1_22['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP1_22 = GWC_CSP1_22['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP1_22 = GWC_CSP1_22['SWC_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1_22 = ((((SWC_PI_F_1_1_1_CSP1_22 + SWC_PI_F_2_1_1_CSP1_22 + SWC_PI_F_3_1_1_CSP1_22)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP1_22 + SWC_PI_F_2_2_1_CSP1_22 + SWC_PI_F_3_2_1_CSP1_22)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP1_22 = twosenwavgall_CSP1_22.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_22 = pd.read_csv('YOUR PATH/CSP12022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_22 = NPIV_CSP1_22['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP1_22 = NPIVCount_CSP1_22.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_22 = dailywavgSWC_CSP1_22
N_CSP1_22 = dailyNPIV_CSP1_22

#use parameters to calculate N0
N0_CSP1_22 = N_CSP1_22 / ((0.0808 / ((pwc_CSP1_22 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1_22 = N0_CSP1_22.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_22 = pd.read_csv('YOUR PATH/CSP12022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_22 = BWEcsv_CSP1_22['BWE']

#Interpolate between dates
BWEinterp_CSP1_22 = BWE_CSP1_22.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP1_22 = BWEinterp_CSP1_22.fillna(int(BWEinterp_CSP1_22.mean()), inplace=True)
N0plotdata_CSP1_22 = N0_CSP1_22.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP1_22 = BWEinterp_CSP1_22
y_CSP1_22 = N0interp_CSP1_22

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP1_22 = np.corrcoef(x_CSP1_22,y_CSP1_22)
corr_CSP1_22 = corr_matrix_CSP1_22[0,1]
R_sq_CSP1_22 = corr_CSP1_22**2

#Least square regression
A_CSP1_22 = np.vstack([x_CSP1_22, np.ones(len(x_CSP1_22))]).T
y_CSP1_22 = np.array(y_CSP1_22)
y_CSP1_22 = y_CSP1_22[:, np.newaxis]
alpha_CSP1_22 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_22.T,A_CSP1_22)),A_CSP1_22.T)),y_CSP1_22)
m_CSP1_22 = alpha_CSP1_22[0]
n_CSP1_22 = alpha_CSP1_22[1]
mm_CSP1_22 = float(m_CSP1_22)
mmm_CSP1_22 = [float(m_CSP1_22)]
nn_CSP1_22 = float(n_CSP1_22)
r_mm_CSP1_22 = round(mm_CSP1_22,2)
r_nn_CSP1_22 = round(nn_CSP1_22,2)

#95% Confidence interval on slope and intercept
z_CSP1_22 = sm.add_constant(x_CSP1_22)
model_CSP1_22 = sm.OLS(y_CSP1_22, z_CSP1_22).fit()
confidence_intervals_CSP1_22 = model_CSP1_22.conf_int(alpha=0.05)
lower_bound_intercept_CSP1_22, upper_bound_intercept_CSP1_22 = confidence_intervals_CSP1_22.loc['const']
lower_bound_slope_CSP1_22, upper_bound_slope_CSP1_22 = confidence_intervals_CSP1_22.loc['BWE']
Slope_CI_CSP1_22 = mm_CSP1_22 - (lower_bound_slope_CSP1_22)
Intercept_CI_CSP1_22 = nn_CSP1_22 - (lower_bound_intercept_CSP1_22)
R_Slope_CI_CSP1_22 = round(Slope_CI_CSP1_22, 2)
R_Intercept_CI_CSP1_22 = round(Intercept_CI_CSP1_22, 2)

#P_value for the least squares regression
p_values_CSP1_22 = model_CSP1_22.pvalues
p_value_slope_CSP1_22 = p_values_CSP1_22.loc['BWE']
p_value_int_CSP1_22 = p_values_CSP1_22.loc['const']
r_p_value_slope_CSP1_22 = "{:.1e}".format(p_value_slope_CSP1_22)
r_p_value_int_CSP1_22 = "{:.1e}".format(p_value_int_CSP1_22)
p_slope_CSP1_22 = float(r_p_value_slope_CSP1_22)
p_int_CSP1_22 = float(r_p_value_int_CSP1_22)

#OLS model
model_CSP1_22 = sm.OLS(y_CSP1_22, z_CSP1_22).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP1_22, y_CSP1_22, color = 'red')
plt.plot(x_CSP1_22, alpha_CSP1_22[0]*x_CSP1_22 + alpha_CSP1_22[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP1_22), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE 2022')
plt.ylim(2500,3250)
plt.xlim(-0.25,4.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_22,r_nn_CSP1_22], [R_Slope_CI_CSP1_22,R_Intercept_CI_CSP1_22]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2022.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2023#################################
##################################################################

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
NPIV_CSP1_23 = pd.read_csv('YOUR PATH/CSP12023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

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

#P_value for the least squares regression
p_values_CSP1_23 = model_CSP1_23.pvalues
p_value_slope_CSP1_23 = p_values_CSP1_23.loc['BWE']
p_value_int_CSP1_23 = p_values_CSP1_23.loc['const']
r_p_value_slope_CSP1_23 = "{:.1e}".format(p_value_slope_CSP1_23)
r_p_value_int_CSP1_23 = "{:.1e}".format(p_value_int_CSP1_23)
p_slope_CSP1_23 = float(r_p_value_slope_CSP1_23)
p_int_CSP1_23 = float(r_p_value_int_CSP1_23)

#OLS model
model_CSP1_23 = sm.OLS(y_CSP1_23, z_CSP1_23).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP1_23, y_CSP1_23, color = 'red')
plt.plot(x_CSP1_23, alpha_CSP1_23[0]*x_CSP1_23 + alpha_CSP1_23[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP1_23), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE 2023')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP1_23,r_nn_CSP1_23], [R_Slope_CI_CSP1_23,R_Intercept_CI_CSP1_23]]
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




##################################################################
##################################################################
##################################################################
############################ CSP2 ################################
##################################################################
##################################################################
##################################################################

##################################################################
#############################2019#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP2_19 = pd.read_csv('YOUR PATH/AmerifluxCSP22019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP2_19 = GWC_CSP2_19['SWC_PI_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2_19 = ((((SWC_PI_F_1_1_1_CSP2_19 + SWC_PI_F_2_1_1_CSP2_19 + SWC_PI_F_3_1_1_CSP2_19)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP2_19 + SWC_PI_F_2_2_1_CSP2_19 + SWC_PI_F_3_2_1_CSP2_19)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP2_19 = twosenwavgall_CSP1_19.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_19 = pd.read_csv('YOUR PATH/CSP22019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_19 = NPIV_CSP2_19['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP2_19 = NPIVCount_CSP2_19.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_19 = dailywavgSWC_CSP2_19
lwc = 0.058
soc = 0.0092
N_CSP2_19 = dailyNPIV_CSP2_19

#use parameters to calculate N0
N0_CSP2_19 = N_CSP2_19 / ((0.0808 / ((pwc_CSP2_19 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2_19 = N0_CSP2_19.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_19 = pd.read_csv('YOUR PATH/CSP22019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_19 = BWEcsv_CSP2_19['BWE']

#Interpolate between dates
BWEinterp_CSP2_19 = BWE_CSP2_19.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP2_19 = BWEinterp_CSP2_19.fillna(int(BWEinterp_CSP2_19.mean()), inplace=True)
N0plotdata_CSP2_19 = N0_CSP2_19.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP2_19 = BWEinterp_CSP2_19
y_CSP2_19 = N0interp_CSP2_19

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP2_19 = np.corrcoef(x_CSP2_19,y_CSP2_19)
corr_CSP2_19 = corr_matrix_CSP2_19[0,1]
R_sq_CSP2_19 = corr_CSP2_19**2

#Least square regression
A_CSP2_19 = np.vstack([x_CSP2_19, np.ones(len(x_CSP2_19))]).T
y_CSP2_19 = np.array(y_CSP2_19)
y_CSP2_19 = y_CSP2_19[:, np.newaxis]
alpha_CSP2_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_19.T,A_CSP2_19)),A_CSP2_19.T)),y_CSP2_19)
m_CSP2_19 = alpha_CSP2_19[0]
n_CSP2_19 = alpha_CSP2_19[1]
mm_CSP2_19 = float(m_CSP2_19)
nn_CSP2_19 = float(n_CSP2_19)
r_mm_CSP2_19 = round(mm_CSP2_19,2)
r_nn_CSP2_19 = round(nn_CSP2_19,2)

#95% Confidence interval on slope and intercept
z_CSP2_19 = sm.add_constant(x_CSP2_19)
model_CSP2_19 = sm.OLS(y_CSP2_19, z_CSP2_19).fit()
confidence_intervals_CSP2_19 = model_CSP2_19.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_19, upper_bound_intercept_CSP2_19 = confidence_intervals_CSP2_19.loc['const']
lower_bound_slope_CSP2_19, upper_bound_slope_CSP2_19 = confidence_intervals_CSP2_19.loc['BWE']
Slope_CI_CSP2_19 = mm_CSP2_19 - (lower_bound_slope_CSP2_19)
Intercept_CI_CSP2_19 = nn_CSP2_19 - (lower_bound_intercept_CSP2_19)
R_Slope_CI_CSP2_19 = round(Slope_CI_CSP2_19, 2)
R_Intercept_CI_CSP2_19 = round(Intercept_CI_CSP2_19, 2)

#P_value for the least squares regression
p_values_CSP2_19 = model_CSP2_19.pvalues
p_value_slope_CSP2_19 = p_values_CSP2_19.loc['BWE']
p_value_int_CSP2_19 = p_values_CSP2_19.loc['const']
r_p_value_slope_CSP2_19 = "{:.1e}".format(p_value_slope_CSP2_19)
r_p_value_int_CSP2_19 = "{:.1e}".format(p_value_int_CSP2_19)
p_slope_CSP2_19 = float(r_p_value_slope_CSP2_19)
p_int_CSP2_19 = float(r_p_value_int_CSP2_19)

#OLS model
model_CSP2_19 = sm.OLS(y_CSP2_19, z_CSP2_19).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP2_19, y_CSP2_19, color = 'red')
plt.plot(x_CSP2_19, alpha_CSP2_19[0]*x_CSP2_19 + alpha_CSP2_19[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP2_19), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE 2019')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_19,r_nn_CSP2_19], [R_Slope_CI_CSP2_19,R_Intercept_CI_CSP2_19]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2019.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2020#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP2_20 = pd.read_csv('YOUR PATH/AmerifluxCSP22020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP2_20 = GWC_CSP2_20['SWC_PI_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2_20 = ((((SWC_PI_F_1_1_1_CSP2_20 + SWC_PI_F_2_1_1_CSP2_20 + SWC_PI_F_3_1_1_CSP2_20)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP2_20 + SWC_PI_F_2_2_1_CSP2_20 + SWC_PI_F_3_2_1_CSP2_20)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP2_20 = twosenwavgall_CSP2_20.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_20 = pd.read_csv('YOUR PATH/CSP22020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_20 = NPIV_CSP2_20['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP2_20 = NPIVCount_CSP2_20.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_20 = dailywavgSWC_CSP2_20
N_CSP2_20 = dailyNPIV_CSP2_20

#use parameters to calculate N0
N0_CSP2_20 = N_CSP2_20 / ((0.0808 / ((pwc_CSP2_20 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2_20 = N0_CSP2_20.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_20 = pd.read_csv('YOUR PATH/CSP22020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_20 = BWEcsv_CSP2_20['BWE']

#Interpolate between dates
BWEinterp_CSP2_20 = BWE_CSP2_20.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP2_20 = BWEinterp_CSP2_20.fillna(int(BWEinterp_CSP2_20.mean()), inplace=True)
N0plotdata_CSP2_20 = N0_CSP2_20.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP2_20 = BWEinterp_CSP2_20
y_CSP2_20 = N0interp_CSP2_20

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP2_20 = np.corrcoef(x_CSP2_20,y_CSP2_20)
corr_CSP2_20 = corr_matrix_CSP2_20[0,1]
R_sq_CSP2_20 = corr_CSP2_20**2

#Least square regression
A_CSP2_20 = np.vstack([x_CSP2_20, np.ones(len(x_CSP2_20))]).T
y_CSP2_20 = np.array(y_CSP2_20)
y_CSP2_20 = y_CSP2_20[:, np.newaxis]
alpha_CSP2_20 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_20.T,A_CSP2_20)),A_CSP2_20.T)),y_CSP2_20)
m_CSP2_20 = alpha_CSP2_20[0]
n_CSP2_20 = alpha_CSP2_20[1]
mm_CSP2_20 = float(m_CSP2_20)
nn_CSP2_20 = float(n_CSP2_20)
r_mm_CSP2_20 = round(mm_CSP2_20,2)
r_nn_CSP2_20 = round(nn_CSP2_20,2)

#95% Confidence interval on slope and intercept
z_CSP2_20 = sm.add_constant(x_CSP2_20)
model_CSP2_20 = sm.OLS(y_CSP2_20, z_CSP2_20).fit()
confidence_intervals_CSP2_20 = model_CSP2_20.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_20, upper_bound_intercept_CSP2_20 = confidence_intervals_CSP2_20.loc['const']
lower_bound_slope_CSP2_20, upper_bound_slope_CSP2_20 = confidence_intervals_CSP2_20.loc['BWE']
Slope_CI_CSP2_20 = mm_CSP2_20 - (lower_bound_slope_CSP2_20)
Intercept_CI_CSP2_20 = nn_CSP2_20 - (lower_bound_intercept_CSP2_20)
R_Slope_CI_CSP2_20 = round(Slope_CI_CSP2_20, 2)
R_Intercept_CI_CSP2_20 = round(Intercept_CI_CSP2_20, 2)

#P_value for the least squares regression
p_values_CSP2_20 = model_CSP2_20.pvalues
p_value_slope_CSP2_20 = p_values_CSP2_20.loc['BWE']
p_value_int_CSP2_20 = p_values_CSP2_20.loc['const']
r_p_value_slope_CSP2_20 = "{:.1e}".format(p_value_slope_CSP2_20)
r_p_value_int_CSP2_20 = "{:.1e}".format(p_value_int_CSP2_20)
p_slope_CSP2_20 = float(r_p_value_slope_CSP2_20)
p_int_CSP2_20 = float(r_p_value_int_CSP2_20)

#OLS model
model_CSP2_20 = sm.OLS(y_CSP2_20, z_CSP2_20).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP2_20, y_CSP2_20, color = 'red')
plt.plot(x_CSP2_20, alpha_CSP2_20[0]*x_CSP2_20 + alpha_CSP2_20[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP2_20), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE 2020')
plt.ylim(2500,3250)
plt.xlim(-0.25,4.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_20,r_nn_CSP2_20], [R_Slope_CI_CSP2_20,R_Intercept_CI_CSP2_20]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2020.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2021#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP2_21 = pd.read_csv('YOUR PATH/AmerifluxCSP22021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP2_21 = GWC_CSP2_21['SWC_PI_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2_21 = ((((SWC_PI_F_1_1_1_CSP2_21 + SWC_PI_F_2_1_1_CSP2_21 + SWC_PI_F_3_1_1_CSP2_21)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP2_21 + SWC_PI_F_2_2_1_CSP2_21 + SWC_PI_F_3_2_1_CSP2_21)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP2_21 = twosenwavgall_CSP2_21.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_21 = pd.read_csv('YOUR PATH/CSP22021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_21 = NPIV_CSP2_21['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP2_21 = NPIVCount_CSP2_21.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_21 = dailywavgSWC_CSP2_21
N_CSP2_21 = dailyNPIV_CSP2_21

#use parameters to calculate N0
N0_CSP2_21 = N_CSP2_21 / ((0.0808 / ((pwc_CSP2_21 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2_21 = N0_CSP2_21.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_21 = pd.read_csv('YOUR PATH/CSP22021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_21 = BWEcsv_CSP2_21['BWE']

#Interpolate between dates
BWEinterp_CSP2_21 = BWE_CSP2_21.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP2_21 = BWEinterp_CSP2_21.fillna(int(BWEinterp_CSP2_21.mean()), inplace=True)
N0plotdata_CSP2_21 = N0_CSP2_21.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP2_21 = BWEinterp_CSP2_21
y_CSP2_21 = N0interp_CSP2_21

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP2_21 = np.corrcoef(x_CSP2_21,y_CSP2_21)
corr_CSP2_21 = corr_matrix_CSP2_21[0,1]
R_sq_CSP2_21 = corr_CSP2_21**2

#Least square regression
A_CSP2_21 = np.vstack([x_CSP2_21, np.ones(len(x_CSP2_21))]).T
y_CSP2_21 = np.array(y_CSP2_21)
y_CSP2_21 = y_CSP2_21[:, np.newaxis]
alpha_CSP2_21 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_21.T,A_CSP2_21)),A_CSP2_21.T)),y_CSP2_21)
m_CSP2_21 = alpha_CSP2_21[0]
n_CSP2_21 = alpha_CSP2_21[1]
mm_CSP2_21 = float(m_CSP2_21)
nn_CSP2_21 = float(n_CSP2_21)
r_mm_CSP2_21 = round(mm_CSP2_21,2)
r_nn_CSP2_21 = round(nn_CSP2_21,2)

#95% Confidence interval on slope and intercept
z_CSP2_21 = sm.add_constant(x_CSP2_21)
model_CSP2_21 = sm.OLS(y_CSP2_21, z_CSP2_21).fit()
confidence_intervals_CSP2_21 = model_CSP2_21.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_21, upper_bound_intercept_CSP2_21 = confidence_intervals_CSP2_21.loc['const']
lower_bound_slope_CSP2_21, upper_bound_slope_CSP2_21 = confidence_intervals_CSP2_21.loc['BWE']
Slope_CI_CSP2_21 = mm_CSP2_21 - (lower_bound_slope_CSP2_21)
Intercept_CI_CSP2_21 = nn_CSP2_21 - (lower_bound_intercept_CSP2_21)
R_Slope_CI_CSP2_21 = round(Slope_CI_CSP2_21, 2)
R_Intercept_CI_CSP2_21 = round(Intercept_CI_CSP2_21, 2)

#P_value for the least squares regression
p_values_CSP2_21 = model_CSP2_21.pvalues
p_value_slope_CSP2_21 = p_values_CSP2_21.loc['BWE']
p_value_int_CSP2_21 = p_values_CSP2_21.loc['const']
r_p_value_slope_CSP2_21 = "{:.1e}".format(p_value_slope_CSP2_21)
r_p_value_int_CSP2_21 = "{:.1e}".format(p_value_int_CSP2_21)
p_slope_CSP2_21 = float(r_p_value_slope_CSP2_21)
p_int_CSP2_21 = float(r_p_value_int_CSP2_21)

#OLS model
model_CSP2_21 = sm.OLS(y_CSP2_21, z_CSP2_21).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP2_21, y_CSP2_21, color = 'red')
plt.plot(x_CSP2_21, alpha_CSP2_21[0]*x_CSP2_21 + alpha_CSP2_21[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP2_21), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE 2021')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_21,r_nn_CSP2_21], [R_Slope_CI_CSP2_21,R_Intercept_CI_CSP2_21]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2021.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2022#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP2_22 = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP2_22 = GWC_CSP2_22['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP2_22 = GWC_CSP2_22['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP2_22 = GWC_CSP2_22['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP2_22 = GWC_CSP2_22['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP2_22 = GWC_CSP2_22['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP2_22 = GWC_CSP2_22['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP2_22 = GWC_CSP2_22['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP2_22 = GWC_CSP2_22['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP2_22 = GWC_CSP2_22['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP2_22 = GWC_CSP2_22['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP2_22 = GWC_CSP2_22['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP2_22 = GWC_CSP2_22['SWC_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2_22 = ((((SWC_PI_F_1_1_1_CSP2_22 + SWC_PI_F_2_1_1_CSP2_22 + SWC_PI_F_3_1_1_CSP2_22)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP2_22 + SWC_PI_F_2_2_1_CSP2_22 + SWC_PI_F_3_2_1_CSP2_22)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP2_22 = twosenwavgall_CSP2_22.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_22 = pd.read_csv('YOUR PATH/CSP22022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_22 = NPIV_CSP2_22['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP2_22 = NPIVCount_CSP2_22.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_22 = dailywavgSWC_CSP2_22
N_CSP2_22 = dailyNPIV_CSP2_22

#use parameters to calculate N0
N0_CSP2_22 = N_CSP2_22 / ((0.0808 / ((pwc_CSP2_22 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2_22 = N0_CSP2_22.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_22 = pd.read_csv('YOUR PATH.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_22 = BWEcsv_CSP2_22['BWE']

#Interpolate between dates
BWEinterp_CSP2_22 = BWE_CSP2_22.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP2_22 = BWEinterp_CSP2_22.fillna(int(BWEinterp_CSP2_22.mean()), inplace=True)
N0plotdata_CSP2_22 = N0_CSP2_22.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP2_22 = BWEinterp_CSP2_22
y_CSP2_22 = N0interp_CSP2_22

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP2_22 = np.corrcoef(x_CSP2_22,y_CSP2_22)
corr_CSP2_22 = corr_matrix_CSP2_22[0,1]
R_sq_CSP2_22 = corr_CSP2_22**2

#Least square regression
A_CSP2_22 = np.vstack([x_CSP2_22, np.ones(len(x_CSP2_22))]).T
y_CSP2_22 = np.array(y_CSP2_22)
y_CSP2_22 = y_CSP2_22[:, np.newaxis]
alpha_CSP2_22 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_22.T,A_CSP2_22)),A_CSP2_22.T)),y_CSP2_22)
m_CSP2_22 = alpha_CSP2_22[0]
n_CSP2_22 = alpha_CSP2_22[1]
mm_CSP2_22 = float(m_CSP2_22)
nn_CSP2_22 = float(n_CSP2_22)
r_mm_CSP2_22 = round(mm_CSP2_22,2)
r_nn_CSP2_22 = round(nn_CSP2_22,2)

#95% Confidence interval on slope and intercept
z_CSP2_22 = sm.add_constant(x_CSP2_22)
model_CSP2_22 = sm.OLS(y_CSP2_22, z_CSP2_22).fit()
confidence_intervals_CSP2_22 = model_CSP2_22.conf_int(alpha=0.05)
lower_bound_intercept_CSP2_22, upper_bound_intercept_CSP2_22 = confidence_intervals_CSP2_22.loc['const']
lower_bound_slope_CSP2_22, upper_bound_slope_CSP2_22 = confidence_intervals_CSP2_22.loc['BWE']
Slope_CI_CSP2_22 = mm_CSP2_22 - (lower_bound_slope_CSP2_22)
Intercept_CI_CSP2_22 = nn_CSP2_22 - (lower_bound_intercept_CSP2_22)
R_Slope_CI_CSP2_22 = round(Slope_CI_CSP2_22, 2)
R_Intercept_CI_CSP2_22 = round(Intercept_CI_CSP2_22, 2)

#P_value for the least squares regression
p_values_CSP2_22 = model_CSP2_22.pvalues
p_value_slope_CSP2_22 = p_values_CSP2_22.loc['BWE']
p_value_int_CSP2_22 = p_values_CSP2_22.loc['const']
r_p_value_slope_CSP2_22 = "{:.1e}".format(p_value_slope_CSP2_22)
r_p_value_int_CSP2_22 = "{:.1e}".format(p_value_int_CSP2_22)
p_slope_CSP2_22 = float(r_p_value_slope_CSP2_22)
p_int_CSP2_22 = float(r_p_value_int_CSP2_22)

#OLS model
model_CSP2_22 = sm.OLS(y_CSP2_22, z_CSP2_22).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP2_22, y_CSP2_22, color = 'red')
plt.plot(x_CSP2_22, alpha_CSP2_22[0]*x_CSP2_22 + alpha_CSP2_22[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP2_22), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE 2022')
plt.ylim(2500,3250)
plt.xlim(-0.25,4.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_22,r_nn_CSP2_22], [R_Slope_CI_CSP2_22,R_Intercept_CI_CSP2_22]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH/CSP2_2022.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2023#################################
##################################################################

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

#P_value for the least squares regression
p_values_CSP2_23 = model_CSP2_23.pvalues
p_value_slope_CSP2_23 = p_values_CSP2_23.loc['BWE']
p_value_int_CSP2_23 = p_values_CSP2_23.loc['const']
r_p_value_slope_CSP2_23 = "{:.1e}".format(p_value_slope_CSP2_23)
r_p_value_int_CSP2_23 = "{:.1e}".format(p_value_int_CSP2_23)
p_slope_CSP2_23 = float(r_p_value_slope_CSP2_23)
p_int_CSP2_23 = float(r_p_value_int_CSP2_23)

#OLS model
model_CSP2_23 = sm.OLS(y_CSP2_23, z_CSP2_23).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP2_23, y_CSP2_23, color = 'red')
plt.plot(x_CSP2_23, alpha_CSP2_23[0]*x_CSP2_23 + alpha_CSP2_23[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 2570, u"r\u00b2 = {:0.2f}".format(R_sq_CSP2_23), size=9)
plt.text(-0.2, 2540, "slope p < 0.001", size=9)
plt.text(-0.2, 2510, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE 2023')
plt.ylim(2500,3250)
plt.xlim(-0.25,7.3)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP2_23,r_nn_CSP2_23], [R_Slope_CI_CSP2_23,R_Intercept_CI_CSP2_23]]
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



##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
##################################################################
##################################################################
##################################################################

##################################################################
#############################2011#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_11 = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_11 = GWC_CSP3_11['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_11 = ((((SWC_PI_F_1_1_1_CSP3_11 + SWC_PI_F_2_1_1_CSP3_11 + SWC_PI_F_3_1_1_CSP3_11 + SWC_PI_F_4_1_1_CSP3_11)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_11 + SWC_PI_F_2_2_1_CSP3_11 + SWC_PI_F_3_2_1_CSP3_11 + SWC_PI_F_4_2_1_CSP3_11)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_11 = twosenwavgall_CSP3_11.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_11 = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_11 = NPIV_CSP3_11['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_11 = NPIVCount_CSP3_11.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_11 = dailywavgSWC_CSP3_11
lwc = 0.058
soc = 0.0092
N_CSP3_11 = dailyNPIV_CSP3_11

#use parameters to calculate N0
N0_CSP3_11 = N_CSP3_11 / ((0.0808 / ((pwc_CSP3_11 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_11 = N0_CSP3_11.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_11 = pd.read_csv('YOUR PATH/CSP32011BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_11 = BWEcsv_CSP3_11['BWE']

#Interpolate between dates
BWEinterp_CSP3_11 = BWE_CSP3_11.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_11 = BWEinterp_CSP3_11.fillna(int(BWEinterp_CSP3_11.mean()), inplace=True)
N0plotdata_CSP3_11 = N0_CSP3_11.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_11 = BWEinterp_CSP3_11
y_CSP3_11 = N0interp_CSP3_11

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_11 = np.corrcoef(x_CSP3_11,y_CSP3_11)
corr_CSP3_11 = corr_matrix_CSP3_11[0,1]
R_sq_CSP3_11 = corr_CSP3_11**2

#Least square regression
A_CSP3_11 = np.vstack([x_CSP3_11, np.ones(len(x_CSP3_11))]).T
y_CSP3_11 = np.array(y_CSP3_11)
y_CSP3_11 = y_CSP3_11[:, np.newaxis]
alpha_CSP3_11 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_11.T,A_CSP3_11)),A_CSP3_11.T)),y_CSP3_11)
m_CSP3_11 = alpha_CSP3_11[0]
n_CSP3_11 = alpha_CSP3_11[1]
mm_CSP3_11 = float(m_CSP3_11)
nn_CSP3_11 = float(n_CSP3_11)
r_mm_CSP3_11 = round(mm_CSP3_11,2)
r_nn_CSP3_11 = round(nn_CSP3_11,2)

#95% Confidence interval on slope and intercept
z_CSP3_11 = sm.add_constant(x_CSP3_11)
model_CSP3_11 = sm.OLS(y_CSP3_11, z_CSP3_11).fit()
confidence_intervals_CSP3_11 = model_CSP3_11.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_11, upper_bound_intercept_CSP3_11 = confidence_intervals_CSP3_11.loc['const']
lower_bound_slope_CSP3_11, upper_bound_slope_CSP3_11 = confidence_intervals_CSP3_11.loc['BWE']
Slope_CI_CSP3_11 = mm_CSP3_11 - (lower_bound_slope_CSP3_11)
Intercept_CI_CSP3_11 = nn_CSP3_11 - (lower_bound_intercept_CSP3_11)
R_Slope_CI_CSP3_11 = round(Slope_CI_CSP3_11, 2)
R_Intercept_CI_CSP3_11 = round(Intercept_CI_CSP3_11, 2)

#P_value for the least squares regression
p_values_CSP3_11 = model_CSP3_11.pvalues
p_value_slope_CSP3_11 = p_values_CSP3_11.loc['BWE']
p_value_int_CSP3_11 = p_values_CSP3_11.loc['const']
r_p_value_slope_CSP3_11 = "{:.1e}".format(p_value_slope_CSP3_11)
r_p_value_int_CSP3_11 = "{:.1e}".format(p_value_int_CSP3_11)
p_slope_CSP3_11 = float(r_p_value_slope_CSP3_11)
p_int_CSP3_11 = float(r_p_value_int_CSP3_11)

#OLS model
model_CSP3_11 = sm.OLS(y_CSP3_11, z_CSP3_11).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_11, y_CSP3_11, color = 'red')
plt.plot(x_CSP3_11, alpha_CSP3_11[0]*x_CSP3_11 + alpha_CSP3_11[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_11), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2011')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_11,r_nn_CSP3_11], [R_Slope_CI_CSP3_11,R_Intercept_CI_CSP3_11]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2011.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2012#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_12 = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_12 = GWC_CSP3_12['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_12 = ((((SWC_PI_F_1_1_1_CSP3_12 + SWC_PI_F_2_1_1_CSP3_12 + SWC_PI_F_3_1_1_CSP3_12 + SWC_PI_F_4_1_1_CSP3_12)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_12 + SWC_PI_F_2_2_1_CSP3_12 + SWC_PI_F_3_2_1_CSP3_12 + SWC_PI_F_4_2_1_CSP3_12)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_12 = twosenwavgall_CSP3_12.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_12 = pd.read_csv('YOUR PATH/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_12 = NPIV_CSP3_12['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_12 = NPIVCount_CSP3_12.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_12 = dailywavgSWC_CSP3_12
lwc = 0.058
soc = 0.0092
N_CSP3_12 = dailyNPIV_CSP3_12

#use parameters to calculate N0
N0_CSP3_12 = N_CSP3_12 / ((0.0808 / ((pwc_CSP3_12 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_12 = N0_CSP3_12.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_12 = pd.read_csv('YOUR PATH/CSP32012BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_12 = BWEcsv_CSP3_12['BWE']

#Interpolate between dates
BWEinterp_CSP3_12 = BWE_CSP3_12.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_12 = BWEinterp_CSP3_12.fillna(int(BWEinterp_CSP3_12.mean()), inplace=True)
N0plotdata_CSP3_12 = N0_CSP3_12.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_12 = BWEinterp_CSP3_12
y_CSP3_12 = N0interp_CSP3_12

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_12 = np.corrcoef(x_CSP3_12,y_CSP3_12)
corr_CSP3_12 = corr_matrix_CSP3_12[0,1]
R_sq_CSP3_12 = corr_CSP3_12**2

#Least square regression
A_CSP3_12 = np.vstack([x_CSP3_12, np.ones(len(x_CSP3_12))]).T
y_CSP3_12 = np.array(y_CSP3_12)
y_CSP3_12 = y_CSP3_12[:, np.newaxis]
alpha_CSP3_12 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_12.T,A_CSP3_12)),A_CSP3_12.T)),y_CSP3_12)
m_CSP3_12 = alpha_CSP3_12[0]
n_CSP3_12 = alpha_CSP3_12[1]
mm_CSP3_12 = float(m_CSP3_12)
nn_CSP3_12 = float(n_CSP3_12)
r_mm_CSP3_12 = round(mm_CSP3_12,2)
r_nn_CSP3_12 = round(nn_CSP3_12,2)

#95% Confidence interval on slope and intercept
z_CSP3_12 = sm.add_constant(x_CSP3_12)
model_CSP3_12 = sm.OLS(y_CSP3_12, z_CSP3_12).fit()
confidence_intervals_CSP3_12 = model_CSP3_12.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_12, upper_bound_intercept_CSP3_12 = confidence_intervals_CSP3_12.loc['const']
lower_bound_slope_CSP3_12, upper_bound_slope_CSP3_12 = confidence_intervals_CSP3_12.loc['BWE']
Slope_CI_CSP3_12 = mm_CSP3_12 - (lower_bound_slope_CSP3_12)
Intercept_CI_CSP3_12 = nn_CSP3_12 - (lower_bound_intercept_CSP3_12)
R_Slope_CI_CSP3_12 = round(Slope_CI_CSP3_12, 2)
R_Intercept_CI_CSP3_12 = round(Intercept_CI_CSP3_12, 2)

#P_value for the least squares regression
p_values_CSP3_12 = model_CSP3_12.pvalues
p_value_slope_CSP3_12 = p_values_CSP3_12.loc['BWE']
p_value_int_CSP3_12 = p_values_CSP3_12.loc['const']
r_p_value_slope_CSP3_12 = "{:.1e}".format(p_value_slope_CSP3_12)
r_p_value_int_CSP3_12 = "{:.1e}".format(p_value_int_CSP3_12)
p_slope_CSP3_12 = float(r_p_value_slope_CSP3_12)
p_int_CSP3_12 = float(r_p_value_int_CSP3_12)

#OLS model
model_CSP3_12 = sm.OLS(y_CSP3_12, z_CSP3_12).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_12, y_CSP3_12, color = 'red')
plt.plot(x_CSP3_12, alpha_CSP3_12[0]*x_CSP3_12 + alpha_CSP3_12[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_12), size=9)
plt.text(-0.2, 1430, f"slope p = {r_p_value_slope_CSP3_12}", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2012')
plt.ylim(1400,1950)
plt.xlim(-0.25,3.4)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_12,r_nn_CSP3_12], [R_Slope_CI_CSP3_12,R_Intercept_CI_CSP3_12]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2012.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2013#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_13 = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_13 = GWC_CSP3_13['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_13 = ((((SWC_PI_F_1_1_1_CSP3_13 + SWC_PI_F_2_1_1_CSP3_13 + SWC_PI_F_3_1_1_CSP3_13 + SWC_PI_F_4_1_1_CSP3_13)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_13 + SWC_PI_F_2_2_1_CSP3_13 + SWC_PI_F_3_2_1_CSP3_13 + SWC_PI_F_4_2_1_CSP3_13)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_13 = twosenwavgall_CSP3_13.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_13 = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_13 = NPIV_CSP3_13['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_13 = NPIVCount_CSP3_13.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_13 = dailywavgSWC_CSP3_13
lwc = 0.058
soc = 0.0092
N_CSP3_13 = dailyNPIV_CSP3_13

#use parameters to calculate N0
N0_CSP3_13 = N_CSP3_13 / ((0.0808 / ((pwc_CSP3_13 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_13 = N0_CSP3_13.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_13 = pd.read_csv('YOUR PATH/CSP32013BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_13 = BWEcsv_CSP3_13['BWE']

#Interpolate between dates
BWEinterp_CSP3_13 = BWE_CSP3_13.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_13 = BWEinterp_CSP3_13.fillna(int(BWEinterp_CSP3_13.mean()), inplace=True)
N0plotdata_CSP3_13 = N0_CSP3_13.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_13 = BWEinterp_CSP3_13
y_CSP3_13 = N0interp_CSP3_13

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_13 = np.corrcoef(x_CSP3_13,y_CSP3_13)
corr_CSP3_13 = corr_matrix_CSP3_13[0,1]
R_sq_CSP3_13 = corr_CSP3_13**2

#Least square regression
A_CSP3_13 = np.vstack([x_CSP3_13, np.ones(len(x_CSP3_13))]).T
y_CSP3_13 = np.array(y_CSP3_13)
y_CSP3_13 = y_CSP3_13[:, np.newaxis]
alpha_CSP3_13 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_13.T,A_CSP3_13)),A_CSP3_13.T)),y_CSP3_13)
m_CSP3_13 = alpha_CSP3_13[0]
n_CSP3_13 = alpha_CSP3_13[1]
mm_CSP3_13 = float(m_CSP3_13)
nn_CSP3_13 = float(n_CSP3_13)
r_mm_CSP3_13 = round(mm_CSP3_13,2)
r_nn_CSP3_13 = round(nn_CSP3_13,2)

#95% Confidence interval on slope and intercept
z_CSP3_13 = sm.add_constant(x_CSP3_13)
model_CSP3_13 = sm.OLS(y_CSP3_13, z_CSP3_13).fit()
confidence_intervals_CSP3_13 = model_CSP3_13.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_13, upper_bound_intercept_CSP3_13 = confidence_intervals_CSP3_13.loc['const']
lower_bound_slope_CSP3_13, upper_bound_slope_CSP3_13 = confidence_intervals_CSP3_13.loc['BWE']
Slope_CI_CSP3_13 = mm_CSP3_13 - (lower_bound_slope_CSP3_13)
Intercept_CI_CSP3_13 = nn_CSP3_13 - (lower_bound_intercept_CSP3_13)
R_Slope_CI_CSP3_13 = round(Slope_CI_CSP3_13, 2)
R_Intercept_CI_CSP3_13 = round(Intercept_CI_CSP3_13, 2)

#P_value for the least squares regression
p_values_CSP3_13 = model_CSP3_13.pvalues
p_value_slope_CSP3_13 = p_values_CSP3_13.loc['BWE']
p_value_int_CSP3_13 = p_values_CSP3_13.loc['const']
r_p_value_slope_CSP3_13 = "{:.1e}".format(p_value_slope_CSP3_13)
r_p_value_int_CSP3_13 = "{:.1e}".format(p_value_int_CSP3_13)
p_slope_CSP3_13 = float(r_p_value_slope_CSP3_13)
p_int_CSP3_13 = float(r_p_value_int_CSP3_13)

#OLS model
model_CSP3_13 = sm.OLS(y_CSP3_13, z_CSP3_13).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_13, y_CSP3_13, color = 'red')
plt.plot(x_CSP3_13, alpha_CSP3_13[0]*x_CSP3_13 + alpha_CSP3_13[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_13), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2013')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_13,r_nn_CSP3_13], [R_Slope_CI_CSP3_13,R_Intercept_CI_CSP3_13]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2013.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2014#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_14 = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_14 = GWC_CSP3_14['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_14 = ((((SWC_PI_F_1_1_1_CSP3_14 + SWC_PI_F_2_1_1_CSP3_14 + SWC_PI_F_3_1_1_CSP3_14 + SWC_PI_F_4_1_1_CSP3_14)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_14 + SWC_PI_F_2_2_1_CSP3_14 + SWC_PI_F_3_2_1_CSP3_14 + SWC_PI_F_4_2_1_CSP3_14)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_14 = twosenwavgall_CSP3_14.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_14 = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_14 = NPIV_CSP3_14['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_14 = NPIVCount_CSP3_14.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_14 = dailywavgSWC_CSP3_14
lwc = 0.058
soc = 0.0092
N_CSP3_14 = dailyNPIV_CSP3_14

#use parameters to calculate N0
N0_CSP3_14 = N_CSP3_14 / ((0.0808 / ((pwc_CSP3_14 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_14 = N0_CSP3_14.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_14 = pd.read_csv('YOUR PATH/CSP32014BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_14 = BWEcsv_CSP3_14['BWE']

#Interpolate between dates
BWEinterp_CSP3_14 = BWE_CSP3_14.interpolate(method = 'linear', limit_direction = 'forward')

BWEFillNaN_CSP3_14 = BWEinterp_CSP3_14.fillna(int(BWEinterp_CSP3_14.mean()), inplace=True)
N0plotdata_CSP3_14 = N0_CSP3_14.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_14 = BWEinterp_CSP3_14
y_CSP3_14 = N0interp_CSP3_14

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_14 = np.corrcoef(x_CSP3_14,y_CSP3_14)
corr_CSP3_14 = corr_matrix_CSP3_14[0,1]
R_sq_CSP3_14 = corr_CSP3_14**2

#Least square regression
A_CSP3_14 = np.vstack([x_CSP3_14, np.ones(len(x_CSP3_14))]).T
y_CSP3_14 = np.array(y_CSP3_14)
y_CSP3_14 = y_CSP3_14[:, np.newaxis]
alpha_CSP3_14 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_14.T,A_CSP3_14)),A_CSP3_14.T)),y_CSP3_14)
m_CSP3_14 = alpha_CSP3_14[0]
n_CSP3_14 = alpha_CSP3_14[1]
mm_CSP3_14 = float(m_CSP3_14)
nn_CSP3_14 = float(n_CSP3_14)
r_mm_CSP3_14 = round(mm_CSP3_14,2)
r_nn_CSP3_14 = round(nn_CSP3_14,2)

#95% Confidence interval on slope and intercept
z_CSP3_14 = sm.add_constant(x_CSP3_14)
model_CSP3_14 = sm.OLS(y_CSP3_14, z_CSP3_14).fit()
confidence_intervals_CSP3_14 = model_CSP3_14.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_14, upper_bound_intercept_CSP3_14 = confidence_intervals_CSP3_14.loc['const']
lower_bound_slope_CSP3_14, upper_bound_slope_CSP3_14 = confidence_intervals_CSP3_14.loc['BWE']
Slope_CI_CSP3_14 = mm_CSP3_14 - (lower_bound_slope_CSP3_14)
Intercept_CI_CSP3_14 = nn_CSP3_14 - (lower_bound_intercept_CSP3_14)
R_Slope_CI_CSP3_14 = round(Slope_CI_CSP3_14, 2)
R_Intercept_CI_CSP3_14 = round(Intercept_CI_CSP3_14, 2)

#P_value for the least squares regression
p_values_CSP3_14 = model_CSP3_14.pvalues
p_value_slope_CSP3_14 = p_values_CSP3_14.loc['BWE']
p_value_int_CSP3_14 = p_values_CSP3_14.loc['const']
r_p_value_slope_CSP3_14 = "{:.1e}".format(p_value_slope_CSP3_14)
r_p_value_int_CSP3_14 = "{:.1e}".format(p_value_int_CSP3_14)
p_slope_CSP3_14 = float(r_p_value_slope_CSP3_14)
p_int_CSP3_14 = float(r_p_value_int_CSP3_14)

#OLS model
model_CSP3_14 = sm.OLS(y_CSP3_14, z_CSP3_14).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_14, y_CSP3_14, color = 'red')
plt.plot(x_CSP3_14, alpha_CSP3_14[0]*x_CSP3_14 + alpha_CSP3_14[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_14), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2014')
plt.ylim(1400,1950)
plt.xlim(-0.25,3.4)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_14,r_nn_CSP3_14], [R_Slope_CI_CSP3_14,R_Intercept_CI_CSP3_14]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2014.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2015#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###

#import spreadsheet
GWC_CSP3_15 = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_15 = GWC_CSP3_15['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_15 = ((((SWC_PI_F_1_1_1_CSP3_15 + SWC_PI_F_2_1_1_CSP3_15 + SWC_PI_F_3_1_1_CSP3_15 + SWC_PI_F_4_1_1_CSP3_15)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_15 + SWC_PI_F_2_2_1_CSP3_15 + SWC_PI_F_3_2_1_CSP3_15 + SWC_PI_F_4_2_1_CSP3_15)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_15 = twosenwavgall_CSP3_15.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_15 = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_15 = NPIV_CSP3_15['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_15 = NPIVCount_CSP3_15.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_15 = dailywavgSWC_CSP3_15
lwc = 0.058
soc = 0.0092
N_CSP3_15 = dailyNPIV_CSP3_15

#use parameters to calculate N0
N0_CSP3_15 = N_CSP3_15 / ((0.0808 / ((pwc_CSP3_15 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_15 = N0_CSP3_15.interpolate(method = 'linear', limit_direction = 'forward')


### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_15 = pd.read_csv('YOUR PATH/CSP32015BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_15 = BWEcsv_CSP3_15['BWE']

#Interpolate between dates
BWEinterp_CSP3_15 = BWE_CSP3_15.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_15 = BWEinterp_CSP3_15.fillna(int(BWEinterp_CSP3_15.mean()), inplace=True)
N0plotdata_CSP3_15 = N0_CSP3_15.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_15 = BWEinterp_CSP3_15
y_CSP3_15 = N0interp_CSP3_15

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_15 = np.corrcoef(x_CSP3_15,y_CSP3_15)
corr_CSP3_15 = corr_matrix_CSP3_15[0,1]
R_sq_CSP3_15 = corr_CSP3_15**2

#Least square regression
A_CSP3_15 = np.vstack([x_CSP3_15, np.ones(len(x_CSP3_15))]).T
y_CSP3_15 = np.array(y_CSP3_15)
y_CSP3_15 = y_CSP3_15[:, np.newaxis]
alpha_CSP3_15 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_15.T,A_CSP3_15)),A_CSP3_15.T)),y_CSP3_15)
m_CSP3_15 = alpha_CSP3_15[0]
n_CSP3_15 = alpha_CSP3_15[1]
mm_CSP3_15 = float(m_CSP3_15)
nn_CSP3_15 = float(n_CSP3_15)
r_mm_CSP3_15 = round(mm_CSP3_15,2)
r_nn_CSP3_15 = round(nn_CSP3_15,2)

#95% Confidence interval on slope and intercept
z_CSP3_15 = sm.add_constant(x_CSP3_15)
model_CSP3_15 = sm.OLS(y_CSP3_15, z_CSP3_15).fit()
confidence_intervals_CSP3_15 = model_CSP3_15.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_15, upper_bound_intercept_CSP3_15 = confidence_intervals_CSP3_15.loc['const']
lower_bound_slope_CSP3_15, upper_bound_slope_CSP3_15 = confidence_intervals_CSP3_15.loc['BWE']
Slope_CI_CSP3_15 = mm_CSP3_15 - (lower_bound_slope_CSP3_15)
Intercept_CI_CSP3_15 = nn_CSP3_15 - (lower_bound_intercept_CSP3_15)
R_Slope_CI_CSP3_15 = round(Slope_CI_CSP3_15, 2)
R_Intercept_CI_CSP3_15 = round(Intercept_CI_CSP3_15, 2)

#P_value for the least squares regression
p_values_CSP3_15 = model_CSP3_15.pvalues
p_value_slope_CSP3_15 = p_values_CSP3_15.loc['BWE']
p_value_int_CSP3_15 = p_values_CSP3_15.loc['const']
r_p_value_slope_CSP3_15 = "{:.1e}".format(p_value_slope_CSP3_15)
r_p_value_int_CSP3_15 = "{:.1e}".format(p_value_int_CSP3_15)
p_slope_CSP3_15 = float(r_p_value_slope_CSP3_15)
p_int_CSP3_15 = float(r_p_value_int_CSP3_15)

#OLS model
model_CSP3_15 = sm.OLS(y_CSP3_15, z_CSP3_15).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_15, y_CSP3_15, color = 'red')
plt.plot(x_CSP3_15, alpha_CSP3_15[0]*x_CSP3_15 + alpha_CSP3_15[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_15), size=9)
plt.text(-0.2, 1430, f"slope p = {r_p_value_slope_CSP3_15}", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2015')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_15,r_nn_CSP3_15], [R_Slope_CI_CSP3_15,R_Intercept_CI_CSP3_15]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2015.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2016#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_16 = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_16 = GWC_CSP3_16['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_16 = ((((SWC_PI_F_1_1_1_CSP3_16 + SWC_PI_F_2_1_1_CSP3_16 + SWC_PI_F_3_1_1_CSP3_16 + SWC_PI_F_4_1_1_CSP3_16)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_16 + SWC_PI_F_2_2_1_CSP3_16 + SWC_PI_F_3_2_1_CSP3_16 + SWC_PI_F_4_2_1_CSP3_16)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_16 = twosenwavgall_CSP3_16.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_16 = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_16 = NPIV_CSP3_16['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_16 = NPIVCount_CSP3_16.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_16 = dailywavgSWC_CSP3_16
lwc = 0.058
soc = 0.0092
N_CSP3_16 = dailyNPIV_CSP3_16

#use parameters to calculate N0
N0_CSP3_16 = N_CSP3_16 / ((0.0808 / ((pwc_CSP3_16 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_16 = N0_CSP3_16.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_16 = pd.read_csv('YOUR PATH/CSP32016BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_16 = BWEcsv_CSP3_16['BWE']

#Interpolate between dates
BWEinterp_CSP3_16 = BWE_CSP3_16.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_16 = BWEinterp_CSP3_16.fillna(int(BWEinterp_CSP3_16.mean()), inplace=True)
N0plotdata_CSP3_16 = N0_CSP3_16.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_16 = BWEinterp_CSP3_16
y_CSP3_16 = N0interp_CSP3_16

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_16 = np.corrcoef(x_CSP3_16,y_CSP3_16)
corr_CSP3_16 = corr_matrix_CSP3_16[0,1]
R_sq_CSP3_16 = corr_CSP3_16**2

#Least square regression
A_CSP3_16 = np.vstack([x_CSP3_16, np.ones(len(x_CSP3_16))]).T
y_CSP3_16 = np.array(y_CSP3_16)
y_CSP3_16 = y_CSP3_16[:, np.newaxis]
alpha_CSP3_16 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_16.T,A_CSP3_16)),A_CSP3_16.T)),y_CSP3_16)
m_CSP3_16 = alpha_CSP3_16[0]
n_CSP3_16 = alpha_CSP3_16[1]
mm_CSP3_16 = float(m_CSP3_16)
nn_CSP3_16 = float(n_CSP3_16)
r_mm_CSP3_16 = round(mm_CSP3_16,2)
r_nn_CSP3_16 = round(nn_CSP3_16,2)

#95% Confidence interval on slope and intercept
z_CSP3_16 = sm.add_constant(x_CSP3_16)
model_CSP3_16 = sm.OLS(y_CSP3_16, z_CSP3_16).fit()
confidence_intervals_CSP3_16 = model_CSP3_16.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_16, upper_bound_intercept_CSP3_16 = confidence_intervals_CSP3_16.loc['const']
lower_bound_slope_CSP3_16, upper_bound_slope_CSP3_16 = confidence_intervals_CSP3_16.loc['BWE']
Slope_CI_CSP3_16 = mm_CSP3_16 - (lower_bound_slope_CSP3_16)
Intercept_CI_CSP3_16 = nn_CSP3_16 - (lower_bound_intercept_CSP3_16)
R_Slope_CI_CSP3_16 = round(Slope_CI_CSP3_16, 2)
R_Intercept_CI_CSP3_16 = round(Intercept_CI_CSP3_16, 2)

#P_value for the least squares regression
p_values_CSP3_16 = model_CSP3_16.pvalues
p_value_slope_CSP3_16 = p_values_CSP3_16.loc['BWE']
p_value_int_CSP3_16 = p_values_CSP3_16.loc['const']
r_p_value_slope_CSP3_16 = "{:.1e}".format(p_value_slope_CSP3_16)
r_p_value_int_CSP3_16 = "{:.1e}".format(p_value_int_CSP3_16)
p_slope_CSP3_16 = float(r_p_value_slope_CSP3_16)
p_int_CSP3_16 = float(r_p_value_int_CSP3_16)

#OLS model
model_CSP3_16 = sm.OLS(y_CSP3_16, z_CSP3_16).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_16, y_CSP3_16, color = 'red')
plt.plot(x_CSP3_16, alpha_CSP3_16[0]*x_CSP3_16 + alpha_CSP3_16[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_16), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2016')
plt.ylim(1400,1950)
plt.xlim(-0.25,3.4)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_16,r_nn_CSP3_16], [R_Slope_CI_CSP3_16,R_Intercept_CI_CSP3_16]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2016.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2017#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_17 = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_17 = GWC_CSP3_17['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_17 = ((((SWC_PI_F_1_1_1_CSP3_17 + SWC_PI_F_2_1_1_CSP3_17 + SWC_PI_F_3_1_1_CSP3_17 + SWC_PI_F_4_1_1_CSP3_17)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_17 + SWC_PI_F_2_2_1_CSP3_17 + SWC_PI_F_3_2_1_CSP3_17 + SWC_PI_F_4_2_1_CSP3_17)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_17 = twosenwavgall_CSP3_17.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_17 = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_17 = NPIV_CSP3_17['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_17 = NPIVCount_CSP3_17.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_17 = dailywavgSWC_CSP3_17
lwc = 0.058
soc = 0.0092
N_CSP3_17 = dailyNPIV_CSP3_17

#use parameters to calculate N0
N0_CSP3_17 = N_CSP3_17 / ((0.0808 / ((pwc_CSP3_17 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_17 = N0_CSP3_17.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_17 = pd.read_csv('YOUR PATH/CSP32017BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_17 = BWEcsv_CSP3_17['BWE']

#Interpolate between dates
BWEinterp_CSP3_17 = BWE_CSP3_17.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_17 = BWEinterp_CSP3_17.fillna(int(BWEinterp_CSP3_17.mean()), inplace=True)
N0plotdata_CSP3_17 = N0_CSP3_17.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_17 = BWEinterp_CSP3_17
y_CSP3_17 = N0interp_CSP3_17

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_17 = np.corrcoef(x_CSP3_17,y_CSP3_17)
corr_CSP3_17 = corr_matrix_CSP3_17[0,1]
R_sq_CSP3_17 = corr_CSP3_17**2

#Least square regression
A_CSP3_17 = np.vstack([x_CSP3_17, np.ones(len(x_CSP3_17))]).T
y_CSP3_17 = np.array(y_CSP3_17)
y_CSP3_17 = y_CSP3_17[:, np.newaxis]
alpha_CSP3_17 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_17.T,A_CSP3_17)),A_CSP3_17.T)),y_CSP3_17)
m_CSP3_17 = alpha_CSP3_17[0]
n_CSP3_17 = alpha_CSP3_17[1]
mm_CSP3_17 = float(m_CSP3_17)
nn_CSP3_17 = float(n_CSP3_17)
r_mm_CSP3_17 = round(mm_CSP3_17,2)
r_nn_CSP3_17 = round(nn_CSP3_17,2)

#95% Confidence interval on slope and intercept
z_CSP3_17 = sm.add_constant(x_CSP3_17)
model_CSP3_17 = sm.OLS(y_CSP3_17, z_CSP3_17).fit()
confidence_intervals_CSP3_17 = model_CSP3_17.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_17, upper_bound_intercept_CSP3_17 = confidence_intervals_CSP3_17.loc['const']
lower_bound_slope_CSP3_17, upper_bound_slope_CSP3_17 = confidence_intervals_CSP3_17.loc['BWE']
Slope_CI_CSP3_17 = mm_CSP3_17 - (lower_bound_slope_CSP3_17)
Intercept_CI_CSP3_17 = nn_CSP3_17 - (lower_bound_intercept_CSP3_17)
R_Slope_CI_CSP3_17 = round(Slope_CI_CSP3_17, 2)
R_Intercept_CI_CSP3_17 = round(Intercept_CI_CSP3_17, 2)

#P_value for the least squares regression
p_values_CSP3_17 = model_CSP3_17.pvalues
p_value_slope_CSP3_17 = p_values_CSP3_17.loc['BWE']
p_value_int_CSP3_17 = p_values_CSP3_17.loc['const']
r_p_value_slope_CSP3_17 = "{:.1e}".format(p_value_slope_CSP3_17)
r_p_value_int_CSP3_17 = "{:.1e}".format(p_value_int_CSP3_17)
p_slope_CSP3_17 = float(r_p_value_slope_CSP3_17)
p_int_CSP3_17 = float(r_p_value_int_CSP3_17)

#OLS model
model_CSP3_17 = sm.OLS(y_CSP3_17, z_CSP3_17).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_17, y_CSP3_17, color = 'red')
plt.plot(x_CSP3_17, alpha_CSP3_17[0]*x_CSP3_17 + alpha_CSP3_17[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_17), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2017')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_17,r_nn_CSP3_17], [R_Slope_CI_CSP3_17,R_Intercept_CI_CSP3_17]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2017.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
#############################2018#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_18 = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_18 = GWC_CSP3_18['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_18 = ((((SWC_PI_F_1_1_1_CSP3_18 + SWC_PI_F_2_1_1_CSP3_18 + SWC_PI_F_3_1_1_CSP3_18 + SWC_PI_F_4_1_1_CSP3_18)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_18 + SWC_PI_F_2_2_1_CSP3_18 + SWC_PI_F_3_2_1_CSP3_18 + SWC_PI_F_4_2_1_CSP3_18)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_18 = twosenwavgall_CSP3_18.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_18 = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_18 = NPIV_CSP3_18['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_18 = NPIVCount_CSP3_18.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_18 = dailywavgSWC_CSP3_18
lwc = 0.058
soc = 0.0092
N_CSP3_18 = dailyNPIV_CSP3_18

#use parameters to calculate N0
N0_CSP3_18 = N_CSP3_18 / ((0.0808 / ((pwc_CSP3_18 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_18 = N0_CSP3_18.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_18 = pd.read_csv('YOUR PATH/CSP32018BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_18 = BWEcsv_CSP3_18['BWE']

#Interpolate between dates
BWEinterp_CSP3_18 = BWE_CSP3_18.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_18 = BWEinterp_CSP3_18.fillna(int(BWEinterp_CSP3_18.mean()), inplace=True)
N0plotdata_CSP3_18 = N0_CSP3_18.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_18 = BWEinterp_CSP3_18
y_CSP3_18 = N0interp_CSP3_18

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_18 = np.corrcoef(x_CSP3_18,y_CSP3_18)
corr_CSP3_18 = corr_matrix_CSP3_18[0,1]
R_sq_CSP3_18 = corr_CSP3_18**2

#Least square regression
A_CSP3_18 = np.vstack([x_CSP3_18, np.ones(len(x_CSP3_18))]).T
y_CSP3_18 = np.array(y_CSP3_18)
y_CSP3_18 = y_CSP3_18[:, np.newaxis]
alpha_CSP3_18 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_18.T,A_CSP3_18)),A_CSP3_18.T)),y_CSP3_18)
m_CSP3_18 = alpha_CSP3_18[0]
n_CSP3_18 = alpha_CSP3_18[1]
mm_CSP3_18 = float(m_CSP3_18)
nn_CSP3_18 = float(n_CSP3_18)
r_mm_CSP3_18 = round(mm_CSP3_18,2)
r_nn_CSP3_18 = round(nn_CSP3_18,2)

#95% Confidence interval on slope and intercept
z_CSP3_18 = sm.add_constant(x_CSP3_18)
model_CSP3_18 = sm.OLS(y_CSP3_18, z_CSP3_18).fit()
confidence_intervals_CSP3_18 = model_CSP3_18.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_18, upper_bound_intercept_CSP3_18 = confidence_intervals_CSP3_18.loc['const']
lower_bound_slope_CSP3_18, upper_bound_slope_CSP3_18 = confidence_intervals_CSP3_18.loc['BWE']
Slope_CI_CSP3_18 = mm_CSP3_18 - (lower_bound_slope_CSP3_18)
Intercept_CI_CSP3_18 = nn_CSP3_18 - (lower_bound_intercept_CSP3_18)
R_Slope_CI_CSP3_18 = round(Slope_CI_CSP3_18, 2)
R_Intercept_CI_CSP3_18 = round(Intercept_CI_CSP3_18, 2)

#P_value for the least squares regression
p_values_CSP3_18 = model_CSP3_18.pvalues
p_value_slope_CSP3_18 = p_values_CSP3_18.loc['BWE']
p_value_int_CSP3_18 = p_values_CSP3_18.loc['const']
r_p_value_slope_CSP3_18 = "{:.1e}".format(p_value_slope_CSP3_18)
r_p_value_int_CSP3_18 = "{:.1e}".format(p_value_int_CSP3_18)
p_slope_CSP3_18 = float(r_p_value_slope_CSP3_18)
p_int_CSP3_18 = float(r_p_value_int_CSP3_18)

#OLS model
model_CSP3_18 = sm.OLS(y_CSP3_18, z_CSP3_18).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_18, y_CSP3_18, color = 'red')
plt.plot(x_CSP3_18, alpha_CSP3_18[0]*x_CSP3_18 + alpha_CSP3_18[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_18), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2018')
plt.ylim(1400,1950)
plt.xlim(-0.25,3.4)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_18,r_nn_CSP3_18], [R_Slope_CI_CSP3_18,R_Intercept_CI_CSP3_18]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2018.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2019#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_19 = pd.read_csv('YOUR PATH/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_19 = GWC_CSP3_19['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_19 = ((((SWC_PI_F_1_1_1_CSP3_19 + SWC_PI_F_2_1_1_CSP3_19 + SWC_PI_F_3_1_1_CSP3_19 + SWC_PI_F_4_1_1_CSP3_19)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_19 + SWC_PI_F_2_2_1_CSP3_19 + SWC_PI_F_3_2_1_CSP3_19 + SWC_PI_F_4_2_1_CSP3_19)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_19 = twosenwavgall_CSP3_19.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_19 = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_19 = NPIV_CSP3_19['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_19 = NPIVCount_CSP3_19.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_19 = dailywavgSWC_CSP3_19
lwc = 0.058
soc = 0.0092
N_CSP3_19 = dailyNPIV_CSP3_19

#use parameters to calculate N0
N0_CSP3_19 = N_CSP3_19 / ((0.0808 / ((pwc_CSP3_19 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_19 = N0_CSP3_19.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_19 = pd.read_csv('YOUR PATH/CSP32019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_19 = BWEcsv_CSP3_19['BWE']

#Interpolate between dates
BWEinterp_CSP3_19 = BWE_CSP3_19.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_19 = BWEinterp_CSP3_19.fillna(int(BWEinterp_CSP3_19.mean()), inplace=True)
N0plotdata_CSP3_19 = N0_CSP3_19.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_19 = BWEinterp_CSP3_19
y_CSP3_19 = N0interp_CSP3_19

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_19 = np.corrcoef(x_CSP3_19,y_CSP3_19)
corr_CSP3_19 = corr_matrix_CSP3_19[0,1]
R_sq_CSP3_19 = corr_CSP3_19**2

#Least square regression
A_CSP3_19 = np.vstack([x_CSP3_19, np.ones(len(x_CSP3_19))]).T
y_CSP3_19 = np.array(y_CSP3_19)
y_CSP3_19 = y_CSP3_19[:, np.newaxis]
alpha_CSP3_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_19.T,A_CSP3_19)),A_CSP3_19.T)),y_CSP3_19)
m_CSP3_19 = alpha_CSP3_19[0]
n_CSP3_19 = alpha_CSP3_19[1]
mm_CSP3_19 = float(m_CSP3_19)
nn_CSP3_19 = float(n_CSP3_19)
r_mm_CSP3_19 = round(mm_CSP3_19,2)
r_nn_CSP3_19 = round(nn_CSP3_19,2)

#95% Confidence interval on slope and intercept
z_CSP3_19 = sm.add_constant(x_CSP3_19)
model_CSP3_19 = sm.OLS(y_CSP3_19, z_CSP3_19).fit()
confidence_intervals_CSP3_19 = model_CSP3_19.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_19, upper_bound_intercept_CSP3_19 = confidence_intervals_CSP3_19.loc['const']
lower_bound_slope_CSP3_19, upper_bound_slope_CSP3_19 = confidence_intervals_CSP3_19.loc['BWE']
Slope_CI_CSP3_19 = mm_CSP3_19 - (lower_bound_slope_CSP3_19)
Intercept_CI_CSP3_19 = nn_CSP3_19 - (lower_bound_intercept_CSP3_19)
R_Slope_CI_CSP3_19 = round(Slope_CI_CSP3_19, 2)
R_Intercept_CI_CSP3_19 = round(Intercept_CI_CSP3_19, 2)

#P_value for the least squares regression
p_values_CSP3_19 = model_CSP3_19.pvalues
p_value_slope_CSP3_19 = p_values_CSP3_19.loc['BWE']
p_value_int_CSP3_19 = p_values_CSP3_19.loc['const']
r_p_value_slope_CSP3_19 = "{:.1e}".format(p_value_slope_CSP3_19)
r_p_value_int_CSP3_19 = "{:.1e}".format(p_value_int_CSP3_19)
p_slope_CSP3_19 = float(r_p_value_slope_CSP3_19)
p_int_CSP3_19 = float(r_p_value_int_CSP3_19)

#OLS model
model_CSP3_19 = sm.OLS(y_CSP3_19, z_CSP3_19).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_19, y_CSP3_19, color = 'red')
plt.plot(x_CSP3_19, alpha_CSP3_19[0]*x_CSP3_19 + alpha_CSP3_19[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_19), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2019')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_19,r_nn_CSP3_19], [R_Slope_CI_CSP3_19,R_Intercept_CI_CSP3_19]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2019.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2020#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_20 = pd.read_csv('YOUR PATH/Ameriflux2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_20 = GWC_CSP3_20['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_20 = ((((SWC_PI_F_1_1_1_CSP3_20 + SWC_PI_F_2_1_1_CSP3_20 + SWC_PI_F_3_1_1_CSP3_20 + SWC_PI_F_4_1_1_CSP3_20)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_20 + SWC_PI_F_2_2_1_CSP3_20 + SWC_PI_F_3_2_1_CSP3_20 + SWC_PI_F_4_2_1_CSP3_20)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_20 = twosenwavgall_CSP3_20.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_20 = pd.read_csv('YOUR PATH/CSP32020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_20 = NPIV_CSP3_20['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_20 = NPIVCount_CSP3_20.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_20 = dailywavgSWC_CSP3_20
lwc = 0.058
soc = 0.0092
N_CSP3_20 = dailyNPIV_CSP3_20

#use parameters to calculate N0
N0_CSP3_20 = N_CSP3_20 / ((0.0808 / ((pwc_CSP3_20 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_20 = N0_CSP3_20.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_20 = pd.read_csv('YOUR PATH/CSP32020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_20 = BWEcsv_CSP3_20['BWE']

#Interpolate between dates
BWEinterp_CSP3_20 = BWE_CSP3_20.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_20 = BWEinterp_CSP3_20.fillna(int(BWEinterp_CSP3_20.mean()), inplace=True)
N0plotdata_CSP3_20 = N0_CSP3_20.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_20 = BWEinterp_CSP3_20
y_CSP3_20 = N0interp_CSP3_20

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_20 = np.corrcoef(x_CSP3_20,y_CSP3_20)
corr_CSP3_20 = corr_matrix_CSP3_20[0,1]
R_sq_CSP3_20 = corr_CSP3_20**2

#Least square regression
A_CSP3_20 = np.vstack([x_CSP3_20, np.ones(len(x_CSP3_20))]).T
y_CSP3_20 = np.array(y_CSP3_20)
y_CSP3_20 = y_CSP3_20[:, np.newaxis]
alpha_CSP3_20 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_20.T,A_CSP3_20)),A_CSP3_20.T)),y_CSP3_20)
m_CSP3_20 = alpha_CSP3_20[0]
n_CSP3_20 = alpha_CSP3_20[1]
mm_CSP3_20 = float(m_CSP3_20)
nn_CSP3_20 = float(n_CSP3_20)
r_mm_CSP3_20 = round(mm_CSP3_20,2)
r_nn_CSP3_20 = round(nn_CSP3_20,2)

#95% Confidence interval on slope and intercept
z_CSP3_20 = sm.add_constant(x_CSP3_20)
model_CSP3_20 = sm.OLS(y_CSP3_20, z_CSP3_20).fit()
confidence_intervals_CSP3_20 = model_CSP3_20.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_20, upper_bound_intercept_CSP3_20 = confidence_intervals_CSP3_20.loc['const']
lower_bound_slope_CSP3_20, upper_bound_slope_CSP3_20 = confidence_intervals_CSP3_20.loc['BWE']
Slope_CI_CSP3_20 = mm_CSP3_20 - (lower_bound_slope_CSP3_20)
Intercept_CI_CSP3_20 = nn_CSP3_20 - (lower_bound_intercept_CSP3_20)
R_Slope_CI_CSP3_20 = round(Slope_CI_CSP3_20, 2)
R_Intercept_CI_CSP3_20 = round(Intercept_CI_CSP3_20, 2)

#P_value for the least squares regression
p_values_CSP3_20 = model_CSP3_20.pvalues
p_value_slope_CSP3_20 = p_values_CSP3_20.loc['BWE']
p_value_int_CSP3_20 = p_values_CSP3_20.loc['const']
r_p_value_slope_CSP3_20 = "{:.1e}".format(p_value_slope_CSP3_20)
r_p_value_int_CSP3_20 = "{:.1e}".format(p_value_int_CSP3_20)
p_slope_CSP3_20 = float(r_p_value_slope_CSP3_20)
p_int_CSP3_20 = float(r_p_value_int_CSP3_20)

#OLS model
model_CSP3_20 = sm.OLS(y_CSP3_20, z_CSP3_20).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_20, y_CSP3_20, color = 'red')
plt.plot(x_CSP3_20, alpha_CSP3_20[0]*x_CSP3_20 + alpha_CSP3_20[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_20), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2020')
plt.ylim(1400,1950)
plt.xlim(-0.25,3.4)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_20,r_nn_CSP3_20], [R_Slope_CI_CSP3_20,R_Intercept_CI_CSP3_20]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2020.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2021#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_21 = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_21 = GWC_CSP3_21['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_21 = ((((SWC_PI_F_1_1_1_CSP3_21 + SWC_PI_F_2_1_1_CSP3_21 + SWC_PI_F_3_1_1_CSP3_21 + SWC_PI_F_4_1_1_CSP3_21)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_21 + SWC_PI_F_2_2_1_CSP3_21 + SWC_PI_F_3_2_1_CSP3_21 + SWC_PI_F_4_2_1_CSP3_21)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_21 = twosenwavgall_CSP3_21.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_21 = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_21 = NPIV_CSP3_21['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_21 = NPIVCount_CSP3_21.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_21 = dailywavgSWC_CSP3_21
lwc = 0.058
soc = 0.0092
N_CSP3_21 = dailyNPIV_CSP3_21

#use parameters to calculate N0
N0_CSP3_21 = N_CSP3_21 / ((0.0808 / ((pwc_CSP3_21 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_21 = N0_CSP3_21.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_21 = pd.read_csv('YOUR PATH/CSP32021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_21 = BWEcsv_CSP3_21['BWE']

#Interpolate between dates
BWEinterp_CSP3_21 = BWE_CSP3_21.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_21 = BWEinterp_CSP3_21.fillna(int(BWEinterp_CSP3_21.mean()), inplace=True)
N0plotdata_CSP3_21 = N0_CSP3_21.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_21 = BWEinterp_CSP3_21
y_CSP3_21 = N0interp_CSP3_21

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_21 = np.corrcoef(x_CSP3_21,y_CSP3_21)
corr_CSP3_21 = corr_matrix_CSP3_21[0,1]
R_sq_CSP3_21 = corr_CSP3_21**2

#Least square regression
A_CSP3_21 = np.vstack([x_CSP3_21, np.ones(len(x_CSP3_21))]).T
y_CSP3_21 = np.array(y_CSP3_21)
y_CSP3_21 = y_CSP3_21[:, np.newaxis]
alpha_CSP3_21 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_21.T,A_CSP3_21)),A_CSP3_21.T)),y_CSP3_21)
m_CSP3_21 = alpha_CSP3_21[0]
n_CSP3_21 = alpha_CSP3_21[1]
mm_CSP3_21 = float(m_CSP3_21)
nn_CSP3_21 = float(n_CSP3_21)
r_mm_CSP3_21 = round(mm_CSP3_21,2)
r_nn_CSP3_21 = round(nn_CSP3_21,2)

#95% Confidence interval on slope and intercept
z_CSP3_21 = sm.add_constant(x_CSP3_21)
model_CSP3_21 = sm.OLS(y_CSP3_21, z_CSP3_21).fit()
confidence_intervals_CSP3_21 = model_CSP3_21.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_21, upper_bound_intercept_CSP3_21 = confidence_intervals_CSP3_21.loc['const']
lower_bound_slope_CSP3_21, upper_bound_slope_CSP3_21 = confidence_intervals_CSP3_21.loc['BWE']
Slope_CI_CSP3_21 = mm_CSP3_21 - (lower_bound_slope_CSP3_21)
Intercept_CI_CSP3_21 = nn_CSP3_21 - (lower_bound_intercept_CSP3_21)
R_Slope_CI_CSP3_21 = round(Slope_CI_CSP3_21, 2)
R_Intercept_CI_CSP3_21 = round(Intercept_CI_CSP3_21, 2)

#P_value for the least squares regression
p_values_CSP3_21 = model_CSP3_21.pvalues
p_value_slope_CSP3_21 = p_values_CSP3_21.loc['BWE']
p_value_int_CSP3_21 = p_values_CSP3_21.loc['const']
r_p_value_slope_CSP3_21 = "{:.1e}".format(p_value_slope_CSP3_21)
r_p_value_int_CSP3_21 = "{:.1e}".format(p_value_int_CSP3_21)
p_slope_CSP3_21 = float(r_p_value_slope_CSP3_21)
p_int_CSP3_21 = float(r_p_value_int_CSP3_21)

#OLS model
model_CSP3_21 = sm.OLS(y_CSP3_21, z_CSP3_21).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_21, y_CSP3_21, color = 'red')
plt.plot(x_CSP3_21, alpha_CSP3_21[0]*x_CSP3_21 + alpha_CSP3_21[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_21), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2021')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_21,r_nn_CSP3_21], [R_Slope_CI_CSP3_21,R_Intercept_CI_CSP3_21]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2021.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2022#################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_22 = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3_22 = GWC_CSP3_22['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3_22 = GWC_CSP3_22['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3_22 = GWC_CSP3_22['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3_22 = GWC_CSP3_22['SWC_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3_22 = GWC_CSP3_22['SWC_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3_22 = GWC_CSP3_22['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3_22 = GWC_CSP3_22['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3_22 = GWC_CSP3_22['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3_22 = GWC_CSP3_22['SWC_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3_22 = GWC_CSP3_22['SWC_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3_22 = GWC_CSP3_22['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3_22 = GWC_CSP3_22['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3_22 = GWC_CSP3_22['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3_22 = GWC_CSP3_22['SWC_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3_22 = GWC_CSP3_22['SWC_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3_22 = GWC_CSP3_22['SWC_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3_22 = GWC_CSP3_22['SWC_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3_22 = GWC_CSP3_22['SWC_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3_22 = GWC_CSP3_22['SWC_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3_22 = GWC_CSP3_22['SWC_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_22 = ((((SWC_PI_F_1_1_1_CSP3_22 + SWC_PI_F_2_1_1_CSP3_22 + SWC_PI_F_3_1_1_CSP3_22 + SWC_PI_F_4_1_1_CSP3_22)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3_22 + SWC_PI_F_2_2_1_CSP3_22 + SWC_PI_F_3_2_1_CSP3_22 + SWC_PI_F_4_2_1_CSP3_22)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3_22 = twosenwavgall_CSP3_22.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_22 = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_22 = NPIV_CSP3_22['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV_CSP3_22 = NPIVCount_CSP3_22.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_22 = dailywavgSWC_CSP3_22
lwc = 0.058
soc = 0.0092
N_CSP3_22 = dailyNPIV_CSP3_22

#use parameters to calculate N0
N0_CSP3_22 = N_CSP3_22 / ((0.0808 / ((pwc_CSP3_22 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_22 = N0_CSP3_22.interpolate(method = 'linear', limit_direction = 'forward')

### BWE DATA ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_22 = pd.read_csv('YOUR PATH/CSP32022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_22 = BWEcsv_CSP3_22['BWE']

#Interpolate between dates
BWEinterp_CSP3_22 = BWE_CSP3_22.interpolate(method = 'linear', limit_direction = 'forward')
BWEFillNaN_CSP3_22 = BWEinterp_CSP3_22.fillna(int(BWEinterp_CSP3_22.mean()), inplace=True)
N0plotdata_CSP3_22 = N0_CSP3_22.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

#define plot variables
x_CSP3_22 = BWEinterp_CSP3_22
y_CSP3_22 = N0interp_CSP3_22

### STATISTICAL ANALYSIS ###
#calculate r-squared
corr_matrix_CSP3_22 = np.corrcoef(x_CSP3_22,y_CSP3_22)
corr_CSP3_22 = corr_matrix_CSP3_22[0,1]
R_sq_CSP3_22 = corr_CSP3_22**2

#Least square regression
A_CSP3_22 = np.vstack([x_CSP3_22, np.ones(len(x_CSP3_22))]).T
y_CSP3_22 = np.array(y_CSP3_22)
y_CSP3_22 = y_CSP3_22[:, np.newaxis]
alpha_CSP3_22 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_22.T,A_CSP3_22)),A_CSP3_22.T)),y_CSP3_22)
m_CSP3_22 = alpha_CSP3_22[0]
n_CSP3_22 = alpha_CSP3_22[1]
mm_CSP3_22 = float(m_CSP3_22)
nn_CSP3_22 = float(n_CSP3_22)
r_mm_CSP3_22 = round(mm_CSP3_22,2)
r_nn_CSP3_22 = round(nn_CSP3_22,2)

#95% Confidence interval on slope and intercept
z_CSP3_22 = sm.add_constant(x_CSP3_22)
model_CSP3_22 = sm.OLS(y_CSP3_22, z_CSP3_22).fit()
confidence_intervals_CSP3_22 = model_CSP3_22.conf_int(alpha=0.05)
lower_bound_intercept_CSP3_22, upper_bound_intercept_CSP3_22 = confidence_intervals_CSP3_22.loc['const']
lower_bound_slope_CSP3_22, upper_bound_slope_CSP3_22 = confidence_intervals_CSP3_22.loc['BWE']
Slope_CI_CSP3_22 = mm_CSP3_22 - (lower_bound_slope_CSP3_22)
Intercept_CI_CSP3_22 = nn_CSP3_22 - (lower_bound_intercept_CSP3_22)
R_Slope_CI_CSP3_22 = round(Slope_CI_CSP3_22, 2)
R_Intercept_CI_CSP3_22 = round(Intercept_CI_CSP3_22, 2)

#P_value for the least squares regression
p_values_CSP3_22 = model_CSP3_22.pvalues
p_value_slope_CSP3_22 = p_values_CSP3_22.loc['BWE']
p_value_int_CSP3_22 = p_values_CSP3_22.loc['const']
r_p_value_slope_CSP3_22 = "{:.1e}".format(p_value_slope_CSP3_22)
r_p_value_int_CSP3_22 = "{:.1e}".format(p_value_int_CSP3_22)
p_slope_CSP3_22 = float(r_p_value_slope_CSP3_22)
p_int_CSP3_22 = float(r_p_value_int_CSP3_22)

#OLS model
model_CSP3_22 = sm.OLS(y_CSP3_22, z_CSP3_22).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_22, y_CSP3_22, color = 'red')
plt.plot(x_CSP3_22, alpha_CSP3_22[0]*x_CSP3_22 + alpha_CSP3_22[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_22), size=9)
plt.text(-0.2, 1430, f"slope p = {r_p_value_slope_CSP3_22}", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2022')
plt.ylim(1400,1950)
plt.xlim(-0.25,3.4)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_22,r_nn_CSP3_22], [R_Slope_CI_CSP3_22,R_Intercept_CI_CSP3_22]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2022.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
#############################2023#################################
##################################################################

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

#P_value for the least squares regression
p_values_CSP3_23 = model_CSP3_23.pvalues
p_value_slope_CSP3_23 = p_values_CSP3_23.loc['BWE']
p_value_int_CSP3_23 = p_values_CSP3_23.loc['const']
r_p_value_slope_CSP3_23 = "{:.1e}".format(p_value_slope_CSP3_23)
r_p_value_int_CSP3_23 = "{:.1e}".format(p_value_int_CSP3_23)
p_slope_CSP3_23 = float(r_p_value_slope_CSP3_23)
p_int_CSP3_23 = float(r_p_value_int_CSP3_23)

#OLS model
model_CSP3_23 = sm.OLS(y_CSP3_23, z_CSP3_23).fit()

#### PLOT ####
#Plot the lines and upper data first
plt.scatter(x_CSP3_23, y_CSP3_23, color = 'red')
plt.plot(x_CSP3_23, alpha_CSP3_23[0]*x_CSP3_23 + alpha_CSP3_23[1], color = 'blue', linestyle = 'solid')
plt.text(-0.2, 1450, u"r\u00b2 = {:0.2f}".format(R_sq_CSP3_23), size=9)
plt.text(-0.2, 1430, "slope p < 0.001", size=9)
plt.text(-0.2, 1410, "intercept p < 0.001", size=9)
plt.xlabel('BWE (mm)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE 2023')
plt.ylim(1400,1950)
plt.xlim(-0.25,6.2)
plt.gca().set_axisbelow(True)
plt.grid(which='major', zorder=-1.0, alpha=0.5)
#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_mm_CSP3_23,r_nn_CSP3_23], [R_Slope_CI_CSP3_23,R_Intercept_CI_CSP3_23]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH/CSP3_2023.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
##################################################################
#Below imports gravimetric data form the 2023 gravimetric sampling 
#so it can be added to the table of values for the data above
##################################################################

## Arithmetic Average ##
### DATA AND CALCULATIONS FOR SWC ###

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

#use parameters to calculate N0 and export as its own file
N0_CSP1 = N_CSP1 / ((0.0808 / ((pwc_CSP1 + lwc + soc) + 0.115)) + 0.372)
N0_CSP1.to_csv(r'YOUR PATH TO MAKE/2023_CSP1.csv')

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
r_nn_CSP1 = round(n_CSP1,2)

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
N0_SE_Calc_CSP1.to_csv(r'YOUR PATH TO MAKE/2023_CSP1_N0_SE_frompy.csv')
N0_error_CSP1 = N0_SE_Calc_CSP1 - N0_CSP1


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
N0_CSP2.to_csv(r'YOUR PATH TO MAKE/2023_CSP2.csv')

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
N0_CSP3_CRNSDistance.to_csv(r'YOUR PATH TO MAKE/2023_CSP3.csv')

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



#The same thing as above, but weighted at 10 cm depth and 25 cm depth, like the TDR system is

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
N0_SE_Calc_CSP1_weighted.to_csv(r'YOUR PATH TO MAKE/2023_CSP1_N0_SE_frompy.csv')
N0_error_CSP1_weighted = N0_SE_Calc_CSP1_weighted - N0_CSP1_weighted


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


##############################################
###################CSP3#######################
##############################################
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



#The same thing as above, but weighted by Schron et al., 2017
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
NPIV_CSP1_weighted_schron = pd.read_csv('YOUR PATH/N0_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

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
NPIV_CSP2_weighted_schron = pd.read_csv('YOUR PATH/N0_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

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
NPIV_CSP3_weighted_schron = pd.read_csv('YOUR PATH/N0_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

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



##################################################################
##################################################################
##################################################################
#########################Bar Chart################################
##################################################################
##################################################################
##################################################################

#This script makes the bar charts for comparison between years for the intercepts and slopes of the N0 vs BWE relationships
##################################################################
#########################Intercept################################
##################################################################

#############################CSP1#################################
# Define the data
nn_CSP1 = [nn_CSP1_19, nn_CSP1_20, nn_CSP1_21, nn_CSP1_22, nn_CSP1_23, nn_CSP1_weighted_schron]
Intercept_CI_CSP1= [Intercept_CI_CSP1_19, Intercept_CI_CSP1_20, Intercept_CI_CSP1_21, Intercept_CI_CSP1_22, Intercept_CI_CSP1_23, Intercept_CI_CSP1_weighted_schron]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023\nGravimetric']

# Create two subplots with different y-axis limits and proportions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 0.3]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax2.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Create the bar chart in the top subplot
ax1.bar(labels, nn_CSP1, yerr=Intercept_CI_CSP1, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax1.set_ylim(2675, 2955)

# Create the bar chart in the bottom subplot
ax2.bar(labels, nn_CSP1, yerr=Intercept_CI_CSP1, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax2.set_ylim(0, 100)  

# hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False) 
ax2.xaxis.tick_bottom()

# Add labels and title
ax1.set_title('CSP1 Intercepts')
plt.xlabel('Year')
plt.ylabel(r'N$_{0}$ vs BWE Intercept', labelpad = 10, y=8)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP1_Intercept_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP2#################################
# Define the data
nn_CSP2 = [nn_CSP2_19, nn_CSP2_20, nn_CSP2_21, nn_CSP2_22, nn_CSP2_23, nn_CSP2_weighted_schron]
Intercept_CI_CSP2= [Intercept_CI_CSP2_19, Intercept_CI_CSP2_20, Intercept_CI_CSP2_21, Intercept_CI_CSP2_22, Intercept_CI_CSP2_23, Intercept_CI_CSP2_weighted_schron]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023\nGravimetric']

# Create two subplots with different y-axis limits and proportions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 0.3]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax2.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Create the bar chart in the top subplot
ax1.bar(labels, nn_CSP2, yerr=Intercept_CI_CSP2, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax1.set_ylim(2755, 3080) 

# Create the bar chart in the bottom subplot
ax2.bar(labels, nn_CSP2, yerr=Intercept_CI_CSP2, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax2.set_ylim(0, 100)  

# hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False) 
ax2.xaxis.tick_bottom()

# Add labels and title
ax1.set_title('CSP2 Intercepts')
plt.xlabel('Year')
plt.ylabel(r'N$_{0}$ vs BWE Intercept', labelpad = 10, y=8)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP2_Intercept_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP3#################################
# Define the data
nn_CSP3 = [nn_CSP3_11, nn_CSP3_12, nn_CSP3_13, nn_CSP3_14, nn_CSP3_15, nn_CSP3_16, nn_CSP3_17, nn_CSP3_18, nn_CSP3_19, nn_CSP3_20, nn_CSP3_21, nn_CSP3_22, nn_CSP3_23, nn_CSP3_weighted_schron]
Intercept_CI_CSP3 = [Intercept_CI_CSP3_11, Intercept_CI_CSP3_12, Intercept_CI_CSP3_13, Intercept_CI_CSP3_14, Intercept_CI_CSP3_15, Intercept_CI_CSP3_16, Intercept_CI_CSP3_17, Intercept_CI_CSP3_18, Intercept_CI_CSP3_19, Intercept_CI_CSP3_20, Intercept_CI_CSP3_21, Intercept_CI_CSP3_22, Intercept_CI_CSP3_23, Intercept_CI_CSP3_weighted_schron]

# Define the labels
labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2023 Grav']

# Create two subplots with different y-axis limits and proportions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 0.3]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax2.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Create the bar chart in the top subplot
ax1.bar(labels, nn_CSP3, yerr=Intercept_CI_CSP3, capsize=6, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 3})
ax1.set_ylim(1495, 1750)  

# Create the bar chart in the bottom subplot
ax2.bar(labels, nn_CSP3, yerr=Intercept_CI_CSP3, capsize=6, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 3})
ax2.set_ylim(0, 100)  

# hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Add labels and title
ax1.set_title('CSP3 Intercepts')
plt.xlabel('Year')
plt.ylabel(r'N$_{0}$ vs BWE Intercept', labelpad = 10, y=8)
ax2.set_xticklabels(labels, rotation=45)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP3_Intercept_Bar_no_2014_grav.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
############################Slope#################################
##################################################################

#############################CSP1#################################
# Define the data
slope_bar_CSP1 = [mm_CSP1_19, mm_CSP1_20, mm_CSP1_21, mm_CSP1_22, mm_CSP1_23, mm_CSP1_weighted_schron]
Slope_bar_CI_CSP1= [Slope_CI_CSP1_19, Slope_CI_CSP1_20, Slope_CI_CSP1_21, Slope_CI_CSP1_22, Slope_CI_CSP1_23, Slope_CI_CSP1_weighted_schron]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023 Grav']

fig, ax3 = plt.subplots()

# Create a bar chart with error bars
ax3.bar(labels, slope_bar_CSP1, yerr=Slope_bar_CI_CSP1, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})

# Add labels and title
ax3.set_xlabel('Year')
ax3.set_ylabel(r'N$_{0}$ vs BWE Slope', labelpad = 9)
ax3.set_title('CSP1 Slopes')
ax3.set_ylim(-58, 0)
ax3.set_axisbelow(True)
ax3.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP1_Slope_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP2#################################
# Define the data
slope_bar_CSP2 = [mm_CSP2_19, mm_CSP2_20, mm_CSP2_21, mm_CSP2_22, mm_CSP2_23, mm_CSP2_weighted_schron]
Slope_bar_CI_CSP2= [Slope_CI_CSP2_19, Slope_CI_CSP2_20, Slope_CI_CSP2_21, Slope_CI_CSP2_22, Slope_CI_CSP2_23, Slope_CI_CSP2_weighted_schron]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023 Grav']

fig, ax4 = plt.subplots()

# Create a bar chart with error bars
ax4.bar(labels, slope_bar_CSP2, yerr=Slope_bar_CI_CSP2, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})

# Add labels and title
ax4.set_xlabel('Year')
ax4.set_ylabel(r'N$_{0}$ vs BWE Slope', labelpad = 9)
ax4.set_title('CSP2 Slopes')
ax4.set_ylim(-86, 0)
ax4.set_axisbelow(True)
ax4.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP2_Slope_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP3#################################
# Define the data
Slope_bar_CSP3 = [mm_CSP3_11, mm_CSP3_12, mm_CSP3_13, mm_CSP3_14, mm_CSP3_15, mm_CSP3_16, mm_CSP3_17, mm_CSP3_18, mm_CSP3_19, mm_CSP3_20, mm_CSP3_21, mm_CSP3_22, mm_CSP3_23, mm_CSP3_weighted_schron]
Slope_bar_CI_CSP3 = [Slope_CI_CSP3_11, Slope_CI_CSP3_12, Slope_CI_CSP3_13, Slope_CI_CSP3_14, Slope_CI_CSP3_15, Slope_CI_CSP3_16, Slope_CI_CSP3_17, Slope_CI_CSP3_18, Slope_CI_CSP3_19, Slope_CI_CSP3_20, Slope_CI_CSP3_21, Slope_CI_CSP3_22, Slope_CI_CSP3_23, Slope_CI_CSP3_weighted_schron]

# Define the labels
labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023', '2023 Grav']

fig, ax5 = plt.subplots()

# Create a bar chart with error bars
ax5.bar(labels, Slope_bar_CSP3, yerr=Slope_bar_CI_CSP3, capsize=6, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 3})

# Add labels and title
ax5.set_xlabel('Year')
ax5.set_ylabel(r'N$_{0}$ vs BWE Slope', labelpad = 9)
ax5.set_title('CSP3 Slopes')
ax5.set_ylim(-32, 45)
ax5.set_axisbelow(True)
ax5.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax5.set_xticklabels(labels, rotation=45)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP3_Slope_Bar_no_2014.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
############ Bar Charts with All Grav Variations #################
##################################################################
##################################################################

#This script makes the tables for the intercepts and slopes of the N0 vs BWE relationships with 95% confidence intervals
#This version includes all gravimetric value weightings for comparison
##################################################################
#########################Intercept################################
##################################################################

#############################CSP1#################################
# Define the data
Intercept_CI_CSP1_19 = float(Intercept_CI_CSP1_19)
Intercept_CI_CSP1_20 = float(Intercept_CI_CSP1_20)
Intercept_CI_CSP1_21 = float(Intercept_CI_CSP1_21)
Intercept_CI_CSP1_22 = float(Intercept_CI_CSP1_22)
Intercept_CI_CSP1_23 = float(Intercept_CI_CSP1_23)
R_Intercept_CI_CSP1 = float(R_Intercept_CI_CSP1)
Intercept_CI_CSP1_weighted_schron = float(Intercept_CI_CSP1_weighted_schron)
Intercept_CI_CSP1_weighted = float(Intercept_CI_CSP1_weighted)

nn_CSP1_new = [float(nn_CSP1_19), float(nn_CSP1_20), float(nn_CSP1_21), float(nn_CSP1_22), float(nn_CSP1_23), float(r_nn_CSP1), float(nn_CSP1_weighted_schron), float(nn_CSP1_weighted)]
Intercept_CSP1_new= [Intercept_CI_CSP1_19, Intercept_CI_CSP1_20, Intercept_CI_CSP1_21, Intercept_CI_CSP1_22, Intercept_CI_CSP1_23, R_Intercept_CI_CSP1, Intercept_CI_CSP1_weighted_schron, Intercept_CI_CSP1_weighted]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023\nGrav.\nArith', '2023\nGrav.\nSchrön', '2023\nGrav.\n10,25cm']

# Create two subplots with different y-axis limits and proportions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 0.3]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax2.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Create the bar chart in the top subplot
ax1.bar(labels, nn_CSP1_new, yerr=Intercept_CSP1_new, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax1.set_ylim(2675, 2955)

# Create the bar chart in the bottom subplot
ax2.bar(labels, nn_CSP1_new, yerr=Intercept_CSP1_new, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax2.set_ylim(0, 100)  

# hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False) 
ax2.xaxis.tick_bottom()

# Add labels and title
ax1.set_title('CSP1 Intercepts')
plt.xlabel('Year')
plt.ylabel(r'N$_{0}$ vs BWE Intercept', labelpad = 10, y=8)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP1_Intercept_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP2#################################
# Define the data
Intercept_CI_CSP2_19 = float(Intercept_CI_CSP2_19)
Intercept_CI_CSP2_20 = float(Intercept_CI_CSP2_20)
Intercept_CI_CSP2_21 = float(Intercept_CI_CSP2_21)
Intercept_CI_CSP2_22 = float(Intercept_CI_CSP2_22)
Intercept_CI_CSP2_23 = float(Intercept_CI_CSP2_23)
R_Intercept_CI_CSP2 = float(R_Intercept_CI_CSP2)
Intercept_CI_CSP2_weighted_schron = float(Intercept_CI_CSP2_weighted_schron)
Intercept_CI_CSP2_weighted = float(Intercept_CI_CSP2_weighted)

nn_CSP2_new = [float(nn_CSP2_19), float(nn_CSP2_20), float(nn_CSP2_21), float(nn_CSP2_22), float(nn_CSP2_23), float(r_nn_CSP2), float(nn_CSP2_weighted_schron), float(nn_CSP2_weighted)]
Intercept_CI_CSP2_new= [Intercept_CI_CSP2_19, Intercept_CI_CSP2_20, Intercept_CI_CSP2_21, Intercept_CI_CSP2_22, Intercept_CI_CSP2_23, R_Intercept_CI_CSP2, Intercept_CI_CSP2_weighted_schron, Intercept_CI_CSP2_weighted]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023\nGrav.\nArith', '2023\nGrav.\nSchrön', '2023\nGrav.\n10,25cm']

# Create two subplots with different y-axis limits and proportions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 0.3]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax2.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Create the bar chart in the top subplot
ax1.bar(labels, nn_CSP2_new, yerr=Intercept_CI_CSP2_new, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax1.set_ylim(2755, 3080) 

# Create the bar chart in the bottom subplot
ax2.bar(labels, nn_CSP2_new, yerr=Intercept_CI_CSP2_new, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})
ax2.set_ylim(0, 100)  

# hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False) 
ax2.xaxis.tick_bottom()

# Add labels and title
ax1.set_title('CSP2 Intercepts')
plt.xlabel('Year')
plt.ylabel(r'N$_{0}$ vs BWE Intercept', labelpad = 10, y=8)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP2_Intercept_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP3#################################
# Define the data
nn_CSP3_11 = float(nn_CSP3_11)
nn_CSP3_12 = float(nn_CSP3_12)
nn_CSP3_13 = float(nn_CSP3_13)
nn_CSP3_14 = float(nn_CSP3_14)
nn_CSP3_15 = float(nn_CSP3_15)
nn_CSP3_16 = float(nn_CSP3_16)
nn_CSP3_17 = float(nn_CSP3_17)
nn_CSP3_18 = float(nn_CSP3_18)
nn_CSP3_19 = float(nn_CSP3_19)
nn_CSP3_20 = float(nn_CSP3_20)
nn_CSP3_21 = float(nn_CSP3_21)
nn_CSP3_22 = float(nn_CSP3_22)
nn_CSP3_23 = float(nn_CSP3_23)
nn_CSP3_CRNSDistance = float(nn_CSP3_CRNSDistance)
nn_CSP3_weighted_schron = float(nn_CSP3_weighted_schron)
nn_CSP3_weighted_CRNSDistance = float(nn_CSP3_weighted_CRNSDistance)
Intercept_CI_CSP3_11 = float(Intercept_CI_CSP3_11)
Intercept_CI_CSP3_12 = float(Intercept_CI_CSP3_12)
Intercept_CI_CSP3_13 = float(Intercept_CI_CSP3_13)
Intercept_CI_CSP3_14 = float(Intercept_CI_CSP3_14)
Intercept_CI_CSP3_15 = float(Intercept_CI_CSP3_15)
Intercept_CI_CSP3_16 = float(Intercept_CI_CSP3_16)
Intercept_CI_CSP3_17 = float(Intercept_CI_CSP3_17)
Intercept_CI_CSP3_18 = float(Intercept_CI_CSP3_18)
Intercept_CI_CSP3_19 = float(Intercept_CI_CSP3_19)
Intercept_CI_CSP3_20 = float(Intercept_CI_CSP3_20)
Intercept_CI_CSP3_21 = float(Intercept_CI_CSP3_21)
Intercept_CI_CSP3_22 = float(Intercept_CI_CSP3_22)
Intercept_CI_CSP3_23 = float(Intercept_CI_CSP3_23)
R_Intercept_CI_CSP3_CRNSDistance = float(R_Intercept_CI_CSP3_CRNSDistance)
Intercept_CI_CSP3_weighted_schron = float(Intercept_CI_CSP3_weighted_schron)
Intercept_CI_CSP3_weighted_CRNSDistance = float(Intercept_CI_CSP3_weighted_CRNSDistance)
nn_CSP3_new = [nn_CSP3_11, nn_CSP3_12, nn_CSP3_13, nn_CSP3_14, nn_CSP3_15, nn_CSP3_16, nn_CSP3_17, nn_CSP3_18, nn_CSP3_19, nn_CSP3_20, nn_CSP3_21, nn_CSP3_22, nn_CSP3_23, nn_CSP3_CRNSDistance, nn_CSP3_weighted_schron, nn_CSP3_weighted_CRNSDistance]
Intercept_CI_CSP3_new = [Intercept_CI_CSP3_11, Intercept_CI_CSP3_12, Intercept_CI_CSP3_13, Intercept_CI_CSP3_14, Intercept_CI_CSP3_15, Intercept_CI_CSP3_16, Intercept_CI_CSP3_17, Intercept_CI_CSP3_18, Intercept_CI_CSP3_19, Intercept_CI_CSP3_20, Intercept_CI_CSP3_21, Intercept_CI_CSP3_22, Intercept_CI_CSP3_23, R_Intercept_CI_CSP3_CRNSDistance, Intercept_CI_CSP3_weighted_schron, Intercept_CI_CSP3_weighted_CRNSDistance]

# Define the labels
labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', 'Grav. Arith.', 'Grav. Schrön', 'Grav. 10,25cm']

# Create two subplots with different y-axis limits and proportions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 0.3]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax1.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax2.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Create the bar chart in the top subplot
ax1.bar(labels, nn_CSP3_new, yerr=Intercept_CI_CSP3_new, capsize=6, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 3})
ax1.set_ylim(1495, 1750)  

# Create the bar chart in the bottom subplot
ax2.bar(labels, nn_CSP3_new, yerr=Intercept_CI_CSP3_new, capsize=6, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 3})
ax2.set_ylim(0, 100)  

# hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Add labels and title
ax1.set_title('CSP3 Intercepts')
plt.xlabel('Year')
plt.ylabel(r'N$_{0}$ vs BWE Intercept', labelpad = 10, y=8)
ax2.set_xticklabels(labels, rotation=45)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP3_Intercept_Bar.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
############################Slope#################################
##################################################################

#############################CSP1#################################
# Define the data
slope_bar_CSP1 = [mm_CSP1_19, mm_CSP1_20, mm_CSP1_21, mm_CSP1_22, mm_CSP1_23, mm_CSP1, mm_CSP1_weighted_schron, mm_CSP1_weighted]
Slope_bar_CI_CSP1= [Slope_CI_CSP1_19, Slope_CI_CSP1_20, Slope_CI_CSP1_21, Slope_CI_CSP1_22, Slope_CI_CSP1_23, Slope_CI_CSP1, Slope_CI_CSP1_weighted_schron, Slope_CI_CSP1_weighted]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023\nGrav.\nArith', '2023\nGrav.\nSchrön', '2023\nGrav.\n10,25cm']

fig, ax3 = plt.subplots()

# Create a bar chart with error bars
ax3.bar(labels, slope_bar_CSP1, yerr=Slope_bar_CI_CSP1, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})

# Add labels and title
ax3.set_xlabel('Year')
ax3.set_ylabel(r'N$_{0}$ vs BWE Slope', labelpad = 9)
ax3.set_title('CSP1 Slopes')
ax3.set_ylim(-58, 0)
ax3.set_axisbelow(True)
ax3.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP1_Slope_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP2#################################
# Define the data
slope_bar_CSP2 = [mm_CSP2_19, mm_CSP2_20, mm_CSP2_21, mm_CSP2_22, mm_CSP2_23, mm_CSP2, mm_CSP2_weighted_schron, mm_CSP2_weighted]
Slope_bar_CI_CSP2= [Slope_CI_CSP2_19, Slope_CI_CSP2_20, Slope_CI_CSP2_21, Slope_CI_CSP2_22, Slope_CI_CSP2_23, Slope_CI_CSP2, Slope_CI_CSP2_weighted_schron, Slope_CI_CSP2_weighted]

# Define the labels
labels = ['2019', '2020', '2021', '2022', '2023', '2023\nGrav.\nArith', '2023\nGrav.\nSchrön', '2023\nGrav.\n10,25cm']

fig, ax4 = plt.subplots()

# Create a bar chart with error bars
ax4.bar(labels, slope_bar_CSP2, yerr=Slope_bar_CI_CSP2, capsize=15, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 4})

# Add labels and title
ax4.set_xlabel('Year')
ax4.set_ylabel(r'N$_{0}$ vs BWE Slope', labelpad = 9)
ax4.set_title('CSP2 Slopes')
ax4.set_ylim(-86, 0)
ax4.set_axisbelow(True)
ax4.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP2_Slope_Bar.png', dpi=300, bbox_inches='tight')
plt.show()

#############################CSP3#################################
# Define the data
Slope_bar_CSP3 = [mm_CSP3_11, mm_CSP3_12, mm_CSP3_13, mm_CSP3_14, mm_CSP3_15, mm_CSP3_16, mm_CSP3_17, mm_CSP3_18, mm_CSP3_19, mm_CSP3_20, mm_CSP3_21, mm_CSP3_22, mm_CSP3_23, mm_CSP3_CRNSDistance, mm_CSP3_weighted_schron, mm_CSP3_weighted_CRNSDistance]
Slope_bar_CI_CSP3 = [Slope_CI_CSP3_11, Slope_CI_CSP3_12, Slope_CI_CSP3_13, Slope_CI_CSP3_14, Slope_CI_CSP3_15, Slope_CI_CSP3_16, Slope_CI_CSP3_17, Slope_CI_CSP3_18, Slope_CI_CSP3_19, Slope_CI_CSP3_20, Slope_CI_CSP3_21, Slope_CI_CSP3_22, Slope_CI_CSP3_23, Slope_CI_CSP3_CRNSDistance, Slope_CI_CSP3_weighted_schron, Slope_CI_CSP3_weighted_CRNSDistance]

# Define the labels
labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023', 'Grav. Arith.', 'Grav. Schrön', 'Grav. 10,25cm']

fig, ax5 = plt.subplots()

# Create a bar chart with error bars
ax5.bar(labels, Slope_bar_CSP3, yerr=Slope_bar_CI_CSP3, capsize=6, color = 'royalblue', ecolor='red', error_kw={'elinewidth': 3})

# Add labels and title
ax5.set_xlabel('Year')
ax5.set_ylabel(r'N$_{0}$ vs BWE Slope', labelpad = 9)
ax5.set_title('CSP3 Slopes')
ax5.set_ylim(-32, 45)
ax5.set_axisbelow(True)
ax5.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
ax5.set_xticklabels(labels, rotation=45)

# Show the plot
plt.savefig('YOUR PATH TO SAVE/CSP3_Slope_Bar.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
##################################################################
################ Testing Confidence Intervals ####################
##################################################################
##################################################################
#This testing was done to confirm whether or not the bar chart confidence intervals overlap
#This determines if a significant relationship exists between the years
#This does not create a plot, so can be commented out if not needed

# Define the confidence intervals CSP1
ci_slopes = [
    (mm_CSP1_19 - Slope_CI_CSP1_19, mm_CSP1_19 + Slope_CI_CSP1_19),
    (mm_CSP1_20 - Slope_CI_CSP1_20, mm_CSP1_20 + Slope_CI_CSP1_20),
    (mm_CSP1_21 - Slope_CI_CSP1_21, mm_CSP1_21 + Slope_CI_CSP1_21),
    (mm_CSP1_22 - Slope_CI_CSP1_22, mm_CSP1_22 + Slope_CI_CSP1_22),
    (mm_CSP1_23 - Slope_CI_CSP1_23, mm_CSP1_23 + Slope_CI_CSP1_23)
]

# Check if the confidence intervals overlap
for i in range(len(ci_slopes)):
    for j in range(i + 1, len(ci_slopes)):
        if ci_slopes[i][1] < ci_slopes[j][0] or ci_slopes[j][1] < ci_slopes[i][0]:
            print(f"The confidence intervals for the slopes of CSP1 in years {2019 + i} and {2019 + j} do not overlap, suggesting a significant difference.")
        else:
            print(f"The confidence intervals for the slopes of CSP1 in years {2019 + i} and {2019 + j} overlap, suggesting no significant difference.")

# Define the confidence intervals CSP2
ci_slopes_CSP2 = [
    (mm_CSP2_19 - Slope_CI_CSP2_19, mm_CSP2_19 + Slope_CI_CSP2_19),
    (mm_CSP2_20 - Slope_CI_CSP2_20, mm_CSP2_20 + Slope_CI_CSP2_20),
    (mm_CSP2_21 - Slope_CI_CSP2_21, mm_CSP2_21 + Slope_CI_CSP2_21),
    (mm_CSP2_22 - Slope_CI_CSP2_22, mm_CSP2_22 + Slope_CI_CSP2_22),
    (mm_CSP2_23 - Slope_CI_CSP2_23, mm_CSP2_23 + Slope_CI_CSP2_23)
]

# Check if the confidence intervals overlap
for i in range(len(ci_slopes_CSP2)):
    for j in range(i + 1, len(ci_slopes_CSP2)):
        if ci_slopes_CSP2[i][1] < ci_slopes_CSP2[j][0] or ci_slopes_CSP2[j][1] < ci_slopes_CSP2[i][0]:
            print(f"The confidence intervals for the slopes of CSP2 in years {2019 + i} and {2019 + j} do not overlap, suggesting a significant difference.")
        else:
            print(f"The confidence intervals for the slopes of CSP2 in years {2019 + i} and {2019 + j} overlap, suggesting no significant difference.")


# Define the confidence intervals CSP1
ci_slopes_CSP3 = [
    (mm_CSP3_11 - Slope_CI_CSP3_11, mm_CSP3_11 + Slope_CI_CSP3_11),
    (mm_CSP3_12 - Slope_CI_CSP3_12, mm_CSP3_12 + Slope_CI_CSP3_12),
    (mm_CSP3_13 - Slope_CI_CSP3_13, mm_CSP3_13 + Slope_CI_CSP3_13),
    (mm_CSP3_14 - Slope_CI_CSP3_14, mm_CSP3_14 + Slope_CI_CSP3_14),
    (mm_CSP3_15 - Slope_CI_CSP3_15, mm_CSP3_15 + Slope_CI_CSP3_15),
    (mm_CSP3_16 - Slope_CI_CSP3_16, mm_CSP3_16 + Slope_CI_CSP3_16),
    (mm_CSP3_17 - Slope_CI_CSP3_17, mm_CSP3_17 + Slope_CI_CSP3_17),
    (mm_CSP3_18 - Slope_CI_CSP3_18, mm_CSP3_18 + Slope_CI_CSP3_18),
    (mm_CSP3_19 - Slope_CI_CSP3_19, mm_CSP3_19 + Slope_CI_CSP3_19),
    (mm_CSP3_20 - Slope_CI_CSP3_20, mm_CSP3_20 + Slope_CI_CSP3_20),
    (mm_CSP3_21 - Slope_CI_CSP3_21, mm_CSP3_21 + Slope_CI_CSP3_21),
    (mm_CSP3_22 - Slope_CI_CSP3_22, mm_CSP3_22 + Slope_CI_CSP3_22),
    (mm_CSP3_23 - Slope_CI_CSP3_23, mm_CSP3_23 + Slope_CI_CSP3_23)
]

# Check if the confidence intervals overlap
for i in range(len(ci_slopes_CSP3)):
    for j in range(i + 1, len(ci_slopes_CSP3)):
        if ci_slopes_CSP3[i][1] < ci_slopes_CSP3[j][0] or ci_slopes_CSP3[j][1] < ci_slopes_CSP3[i][0]:
            print(f"The confidence intervals for the slopes of CSP3 in years {2011 + i} and {2011 + j} do not overlap, suggesting a significant difference.")
        else:
            print(f"The confidence intervals for the slopes of CSP3 in years {2011 + i} and {2011 + j} overlap, suggesting no significant difference.")

# Define the confidence intervals for intercepts CSP1
ci_intercepts = [
    (nn_CSP1_19 - Intercept_CI_CSP1_19, nn_CSP1_19 + Intercept_CI_CSP1_19),
    (nn_CSP1_20 - Intercept_CI_CSP1_20, nn_CSP1_20 + Intercept_CI_CSP1_20),
    (nn_CSP1_21 - Intercept_CI_CSP1_21, nn_CSP1_21 + Intercept_CI_CSP1_21),
    (nn_CSP1_22 - Intercept_CI_CSP1_22, nn_CSP1_22 + Intercept_CI_CSP1_22),
    (nn_CSP1_23 - Intercept_CI_CSP1_23, nn_CSP1_23 + Intercept_CI_CSP1_23)
]

# Check if the confidence intervals overlap
for i in range(len(ci_intercepts)):
    for j in range(i + 1, len(ci_intercepts)):
        if ci_intercepts[i][1] < ci_intercepts[j][0] or ci_intercepts[j][1] < ci_intercepts[i][0]:
            print(f"The confidence intervals for the intercepts of CSP1 in years {2019 + i} and {2019 + j} do not overlap, suggesting a significant difference.")
        else:
            print(f"The confidence intervals for the intercepts of CSP1 in years {2019 + i} and {2019 + j} overlap, suggesting no significant difference.")

# Define the confidence intervals for intercepts CSP2
ci_intercepts_CSP2 = [
    (nn_CSP2_19 - Intercept_CI_CSP2_19, nn_CSP2_19 + Intercept_CI_CSP2_19),
    (nn_CSP2_20 - Intercept_CI_CSP2_20, nn_CSP2_20 + Intercept_CI_CSP2_20),
    (nn_CSP2_21 - Intercept_CI_CSP2_21, nn_CSP2_21 + Intercept_CI_CSP2_21),
    (nn_CSP2_22 - Intercept_CI_CSP2_22, nn_CSP2_22 + Intercept_CI_CSP2_22),
    (nn_CSP2_23 - Intercept_CI_CSP2_23, nn_CSP2_23 + Intercept_CI_CSP2_23)
]

# Check if the confidence intervals overlap
for i in range(len(ci_intercepts_CSP2)):
    for j in range(i + 1, len(ci_intercepts_CSP2)):
        if ci_intercepts_CSP2[i][1] < ci_intercepts_CSP2[j][0] or ci_intercepts_CSP2[j][1] < ci_intercepts_CSP2[i][0]:
            print(f"The confidence intervals for the intercepts of CSP2 in years {2019 + i} and {2019 + j} do not overlap, suggesting a significant difference.")
        else:
            print(f"The confidence intervals for the intercepts of CSP2 in years {2019 + i} and {2019 + j} overlap, suggesting no significant difference.")

ci_intercepts_CSP3 = [
    (nn_CSP3_11 - Intercept_CI_CSP3_11, nn_CSP3_11 + Intercept_CI_CSP3_11),
    (nn_CSP3_12 - Intercept_CI_CSP3_12, nn_CSP3_12 + Intercept_CI_CSP3_12),
    (nn_CSP3_13 - Intercept_CI_CSP3_13, nn_CSP3_13 + Intercept_CI_CSP3_13),
    (nn_CSP3_14 - Intercept_CI_CSP3_14, nn_CSP3_14 + Intercept_CI_CSP3_14),
    (nn_CSP3_15 - Intercept_CI_CSP3_15, nn_CSP3_15 + Intercept_CI_CSP3_15),
    (nn_CSP3_16 - Intercept_CI_CSP3_16, nn_CSP3_16 + Intercept_CI_CSP3_16),
    (nn_CSP3_17 - Intercept_CI_CSP3_17, nn_CSP3_17 + Intercept_CI_CSP3_17),
    (nn_CSP3_18 - Intercept_CI_CSP3_18, nn_CSP3_18 + Intercept_CI_CSP3_18),
    (nn_CSP3_19 - Intercept_CI_CSP3_19, nn_CSP3_19 + Intercept_CI_CSP3_19),
    (nn_CSP3_20 - Intercept_CI_CSP3_20, nn_CSP3_20 + Intercept_CI_CSP3_20),
    (nn_CSP3_21 - Intercept_CI_CSP3_21, nn_CSP3_21 + Intercept_CI_CSP3_21),
    (nn_CSP3_22 - Intercept_CI_CSP3_22, nn_CSP3_22 + Intercept_CI_CSP3_22),
    (nn_CSP3_23 - Intercept_CI_CSP3_23, nn_CSP3_23 + Intercept_CI_CSP3_23)
]

# Check if the confidence intervals overlap
for i in range(len(ci_intercepts_CSP3)):
    for j in range(i + 1, len(ci_intercepts_CSP3)):
        if ci_intercepts_CSP3[i][1] < ci_intercepts_CSP3[j][0] or ci_intercepts_CSP3[j][1] < ci_intercepts_CSP3[i][0]:
            print(f"The confidence intervals for the intercepts of CSP3 in years {2011 + i} and {2011 + j} do not overlap, suggesting a significant difference.")
        else:
            print(f"The confidence intervals for the intercepts of CSP3 in years {2011 + i} and {2011 + j} overlap, suggesting no significant difference.")





##################################################################
##################################################################
################ Summary Table ####################################
##################################################################
##################################################################
#This script makes the summary table of slopes, intercepts, and confidence intervals in a table format

# Calculate the mean and SEM for each group
mean_CSP1 = np.mean([r_mm_CSP1_23/r_nn_CSP1_23, r_mm_CSP1_22/r_nn_CSP1_22, r_mm_CSP1_21/r_nn_CSP1_21, r_mm_CSP1_20/r_nn_CSP1_20, r_mm_CSP1_19/r_nn_CSP1_19])
sem_CSP1 = np.std([r_mm_CSP1_23/r_nn_CSP1_23, r_mm_CSP1_22/r_nn_CSP1_22, r_mm_CSP1_21/r_nn_CSP1_21, r_mm_CSP1_20/r_nn_CSP1_20, r_mm_CSP1_19/r_nn_CSP1_19]) / np.sqrt(5)

# Calculate the 95% confidence interval for CSP1
ci_CSP1 = [mean_CSP1 - 1.96 * sem_CSP1, mean_CSP1 + 1.96 * sem_CSP1]

# Calculate the mean and SEM for CSP2
mean_CSP2 = np.mean([r_mm_CSP2_23/r_nn_CSP2_23, r_mm_CSP2_22/r_nn_CSP2_22, r_mm_CSP2_21/r_nn_CSP2_21, r_mm_CSP2_20/r_nn_CSP2_20, r_mm_CSP2_19/r_nn_CSP2_19])
sem_CSP2 = np.std([r_mm_CSP2_23/r_nn_CSP2_23, r_mm_CSP2_22/r_nn_CSP2_22, r_mm_CSP2_21/r_nn_CSP2_21, r_mm_CSP2_20/r_nn_CSP2_20, r_mm_CSP2_19/r_nn_CSP2_19]) / np.sqrt(5)

# Calculate the 95% confidence interval for CSP2
ci_CSP2 = [mean_CSP2 - 1.96 * sem_CSP2, mean_CSP2 + 1.96 * sem_CSP2]

# Calculate the mean and SEM for CSP3
mean_CSP3 = np.mean([r_mm_CSP3_23/r_nn_CSP3_23, r_mm_CSP3_22/r_nn_CSP3_22, r_mm_CSP3_21/r_nn_CSP3_21, r_mm_CSP3_20/r_nn_CSP3_20, r_mm_CSP3_19/r_nn_CSP3_19, r_mm_CSP3_18/r_nn_CSP3_18, r_mm_CSP3_17/r_nn_CSP3_17, r_mm_CSP3_16/r_nn_CSP3_16, r_mm_CSP3_15/r_nn_CSP3_15, r_mm_CSP3_14/r_nn_CSP3_14, r_mm_CSP3_13/r_nn_CSP3_13, r_mm_CSP3_12/r_nn_CSP3_12, r_mm_CSP3_11/r_nn_CSP3_11])
sem_CSP3 = np.std([r_mm_CSP3_23/r_nn_CSP3_23, r_mm_CSP3_22/r_nn_CSP3_22, r_mm_CSP3_21/r_nn_CSP3_21, r_mm_CSP3_20/r_nn_CSP3_20, r_mm_CSP3_19/r_nn_CSP3_19, r_mm_CSP3_18/r_nn_CSP3_18, r_mm_CSP3_17/r_nn_CSP3_17, r_mm_CSP3_16/r_nn_CSP3_16, r_mm_CSP3_15/r_nn_CSP3_15, r_mm_CSP3_14/r_nn_CSP3_14, r_mm_CSP3_13/r_nn_CSP3_13, r_mm_CSP3_12/r_nn_CSP3_12, r_mm_CSP3_11/r_nn_CSP3_11]) / np.sqrt(13)

# Calculate the 95% confidence interval for CSP3
ci_CSP3 = [mean_CSP3 - 1.96 * sem_CSP3, mean_CSP3 + 1.96 * sem_CSP3]

# Create a Table of summary data from above
table_data = {
    'Year': ['2023 Gravimetric\nArithmetic', '2023 Gravimetric\nSchrön Weighted', '2023 Gravimetric\n10cm, 25cm Weighted','2023', '2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', 'TDR\nAverage'],
    'CSP1\nIntercept': [f"{r_nn_CSP1} +/- {R_Intercept_CI_CSP1}", f"{r_nn_CSP1_weighted_schron} +/- {R_Intercept_CI_CSP1_weighted_schron}",f"{r_nn_CSP1_weighted} +/- {R_Intercept_CI_CSP1_weighted}", f"{r_nn_CSP1_23} +/- {R_Intercept_CI_CSP1_23}", f"{r_nn_CSP1_22} +/- {R_Intercept_CI_CSP1_22}", f"{r_nn_CSP1_21} +/- {R_Intercept_CI_CSP1_21}", f"{r_nn_CSP1_20} +/- {R_Intercept_CI_CSP1_20}", f"{r_nn_CSP1_19} +/- {R_Intercept_CI_CSP1_19}", None, None, None, None, None, None, None, None, f"{round((r_nn_CSP1_23 + r_nn_CSP1_22 + r_nn_CSP1_21 + r_nn_CSP1_20 + r_nn_CSP1_19)/5,1)} +/- {round((R_Intercept_CI_CSP1_23 + R_Intercept_CI_CSP1_22 + R_Intercept_CI_CSP1_21 + R_Intercept_CI_CSP1_20 + R_Intercept_CI_CSP1_19)/5,1)}"],
    'CSP1 Slope': [f"{r_mm_CSP1} +/- {R_Slope_CI_CSP1}", f"{r_mm_CSP1_weighted_schron} +/- {R_Slope_CI_CSP1_weighted_schron}", f"{r_mm_CSP1_weighted} +/- {R_Slope_CI_CSP1_weighted}", f"{r_mm_CSP1_23} +/- {R_Slope_CI_CSP1_23}", f"{r_mm_CSP1_22} +/- {R_Slope_CI_CSP1_22}", f"{r_mm_CSP1_21} +/- {R_Slope_CI_CSP1_21}", f"{r_mm_CSP1_20} +/- {R_Slope_CI_CSP1_20}", f"{r_mm_CSP1_19} +/- {R_Slope_CI_CSP1_19}", None, None, None, None, None, None, None, None, f"{round((r_mm_CSP1_23 + r_mm_CSP1_22 + r_mm_CSP1_21 + r_mm_CSP1_20 + r_mm_CSP1_19)/5,1)} +/- {round((R_Slope_CI_CSP1_23 + R_Slope_CI_CSP1_22 + R_Slope_CI_CSP1_21 + R_Slope_CI_CSP1_20 + R_Slope_CI_CSP1_19)/5, 1)}"],
    'CSP2\nIntercept': [f"{r_nn_CSP2} +/- {R_Intercept_CI_CSP2}", f"{r_nn_CSP2_weighted_schron} +/- {R_Intercept_CI_CSP2_weighted_schron}", f"{r_nn_CSP2_weighted} +/- {R_Intercept_CI_CSP2_weighted}", f"{r_nn_CSP2_23} +/- {R_Intercept_CI_CSP2_23}", f"{r_nn_CSP2_22} +/- {R_Intercept_CI_CSP2_22}", f"{r_nn_CSP2_21} +/- {R_Intercept_CI_CSP2_21}", f"{r_nn_CSP2_20} +/- {R_Intercept_CI_CSP2_20}", f"{r_nn_CSP2_19} +/- {R_Intercept_CI_CSP2_19}", None, None, None, None, None, None, None, None, f"{round((r_nn_CSP2_23 + r_nn_CSP2_22 + r_nn_CSP2_21 + r_nn_CSP2_20 + r_nn_CSP2_19)/5,1)} +/- {round((R_Intercept_CI_CSP2_23 + R_Intercept_CI_CSP2_22 + R_Intercept_CI_CSP2_21 + R_Intercept_CI_CSP2_20 + R_Intercept_CI_CSP2_19)/5,1)}"],
    "CSP2 Slope": [f"{r_mm_CSP2} +/- {R_Slope_CI_CSP2}", f"{r_mm_CSP2_weighted_schron} +/- {R_Slope_CI_CSP2_weighted_schron}", f"{r_mm_CSP2_weighted} +/- {R_Slope_CI_CSP2_weighted}", f"{r_mm_CSP2_23} +/- {R_Slope_CI_CSP2_23}", f"{r_mm_CSP2_22} +/- {R_Slope_CI_CSP2_22}", f"{r_mm_CSP2_21} +/- {R_Slope_CI_CSP2_21}", f"{r_mm_CSP2_20} +/- {R_Slope_CI_CSP2_20}", f"{r_mm_CSP2_19} +/- {R_Slope_CI_CSP2_19}", None, None, None, None, None, None, None, None, f"{round((r_mm_CSP2_23 + r_mm_CSP2_22 + r_mm_CSP2_21 + r_mm_CSP2_20 + r_mm_CSP2_19)/5,1)} +/- {round((R_Slope_CI_CSP2_23 + R_Slope_CI_CSP2_22 + R_Slope_CI_CSP2_21 + R_Slope_CI_CSP2_20 + R_Slope_CI_CSP2_19)/5,1)}"],
    'CSP3\nIntercept': [f"{r_nn_CSP3_CRNSDistance} +/- {R_Intercept_CI_CSP3_CRNSDistance}", f"{r_nn_CSP3_weighted_schron} +/- {R_Intercept_CI_CSP3_weighted_schron}", f"{r_nn_CSP3_weighted_CRNSDistance} +/- {R_Intercept_CI_CSP3_weighted_CRNSDistance}",f"{r_nn_CSP3_23} +/- {R_Intercept_CI_CSP3_23}", f"{r_nn_CSP3_22} +/- {R_Intercept_CI_CSP3_22}", f"{r_nn_CSP3_21} +/- {R_Intercept_CI_CSP3_21}", f"{r_nn_CSP3_20} +/- {R_Intercept_CI_CSP3_20}", f"{r_nn_CSP3_19} +/- {R_Intercept_CI_CSP3_19}", f"{r_nn_CSP3_18} +/- {R_Intercept_CI_CSP3_18}", f"{r_nn_CSP3_17} +/- {R_Intercept_CI_CSP3_17}", f"{r_nn_CSP3_16} +/- {R_Intercept_CI_CSP3_16}", f"{r_nn_CSP3_15} +/- {R_Intercept_CI_CSP3_15}", f"{r_nn_CSP3_14} +/- {R_Intercept_CI_CSP3_14}", f"{r_nn_CSP3_13} +/- {R_Intercept_CI_CSP3_13}", f"{r_nn_CSP3_12} +/- {R_Intercept_CI_CSP3_12}", f"{r_nn_CSP3_11} +/- {R_Intercept_CI_CSP3_11}", f"{round((r_nn_CSP3_23 + r_nn_CSP3_22 + r_nn_CSP3_21 + r_nn_CSP3_20 + r_nn_CSP3_19 + r_nn_CSP3_18 + r_nn_CSP3_17 + r_nn_CSP3_16 + r_nn_CSP3_15 + r_nn_CSP3_14 + r_nn_CSP3_13 + r_nn_CSP3_12 + r_nn_CSP3_11)/13,1)} +/- {round((R_Intercept_CI_CSP3_23 + R_Intercept_CI_CSP3_22 + R_Intercept_CI_CSP3_21 + R_Intercept_CI_CSP3_20 + R_Intercept_CI_CSP3_19 + R_Intercept_CI_CSP3_18 + R_Intercept_CI_CSP3_17 + R_Intercept_CI_CSP3_16 + R_Intercept_CI_CSP3_15 + R_Intercept_CI_CSP3_14 + R_Intercept_CI_CSP3_13 + R_Intercept_CI_CSP3_12 + R_Intercept_CI_CSP3_11)/13, 1)}"],
    "CSP3 Slope": [f"{r_mm_CSP3_CRNSDistance} +/- {R_Slope_CI_CSP3_CRNSDistance}", f"{r_mm_CSP3_weighted_schron} +/- {R_Intercept_CI_CSP3_weighted_schron}", f"{r_mm_CSP3_weighted_CRNSDistance} +/- {R_Slope_CI_CSP3_weighted_CRNSDistance}",f"{r_mm_CSP3_23} +/- {R_Slope_CI_CSP3_23}", f"{r_mm_CSP3_22} +/- {R_Slope_CI_CSP3_22}", f"{r_mm_CSP3_21} +/- {R_Slope_CI_CSP3_21}", f"{r_mm_CSP3_20} +/- {R_Slope_CI_CSP3_20}", f"{r_mm_CSP3_19} +/- {R_Slope_CI_CSP3_19}", f"{r_mm_CSP3_18} +/- {R_Slope_CI_CSP3_18}", f"{r_mm_CSP3_17} +/- {R_Slope_CI_CSP3_17}", f"{r_mm_CSP3_16} +/- {R_Slope_CI_CSP3_16}", f"{r_mm_CSP3_15} +/- {R_Slope_CI_CSP3_15}", f"{r_mm_CSP3_14} +/- {R_Slope_CI_CSP3_14}", f"{r_mm_CSP3_13} +/- {R_Slope_CI_CSP3_13}", f"{r_mm_CSP3_12} +/- {R_Slope_CI_CSP3_12}", f"{r_mm_CSP3_11} +/- {R_Slope_CI_CSP3_11}", f"{round((r_mm_CSP3_23 + r_mm_CSP3_22 + r_mm_CSP3_21 + r_mm_CSP3_20 + r_mm_CSP3_19 + r_mm_CSP3_18 + r_mm_CSP3_17 + r_mm_CSP3_16 + r_mm_CSP3_15 + r_mm_CSP3_14 + r_mm_CSP3_13 + r_mm_CSP3_12 + r_mm_CSP3_11)/13,1)} +/- {round((R_Slope_CI_CSP3_23 + R_Slope_CI_CSP3_22 + R_Slope_CI_CSP3_21 + R_Slope_CI_CSP3_20 + R_Slope_CI_CSP3_19 + R_Slope_CI_CSP3_18 + R_Slope_CI_CSP3_17 + R_Slope_CI_CSP3_16 + R_Slope_CI_CSP3_15 + R_Slope_CI_CSP3_14 + R_Slope_CI_CSP3_13 + R_Slope_CI_CSP3_12 + R_Slope_CI_CSP3_11)/13, 1)}"],
    r'CSP1 $\eta$': [f"{round((r_mm_CSP1/r_nn_CSP1)*100,2)}", f"{round((r_mm_CSP1_weighted_schron/r_nn_CSP1_weighted_schron)*100,2)}", f"{round((r_mm_CSP1_weighted/r_nn_CSP1_weighted)*100,2)}",f"{round((r_mm_CSP1_23/r_nn_CSP1_23)*100,2)}", f"{round((r_mm_CSP1_22/r_nn_CSP1_22)*100,2)}", f"{round((r_mm_CSP1_21/r_nn_CSP1_21)*100,2)}", f"{round((r_mm_CSP1_20/r_nn_CSP1_20)*100,2)}", f"{round((r_mm_CSP1_19/r_nn_CSP1_19)*100,2)}", None, None, None, None, None, None, None, None, f"{round(((r_mm_CSP1_23/r_nn_CSP1_23)*100 + (r_mm_CSP1_22/r_nn_CSP1_22)*100 + (r_mm_CSP1_21/r_nn_CSP1_21)*100 + (r_mm_CSP1_20/r_nn_CSP1_20)*100 + (r_mm_CSP1_19/r_nn_CSP1_19)*100)/5,1)} +/- 0.27"],
    r'CSP2 $\eta$': [f"{round((r_mm_CSP2/r_nn_CSP2)*100,2)}", f"{round((r_mm_CSP2_weighted_schron/r_nn_CSP2_weighted_schron)*100,2)}", f"{round((r_mm_CSP2_weighted/r_nn_CSP2_weighted)*100,2)}", f"{round((r_mm_CSP2_23/r_nn_CSP2_23)*100,2)}", f"{round((r_mm_CSP2_22/r_nn_CSP2_22)*100,2)}", f"{round((r_mm_CSP2_21/r_nn_CSP2_21)*100,2)}", f"{round((r_mm_CSP2_20/r_nn_CSP2_20)*100,2)}", f"{round((r_mm_CSP2_19/r_nn_CSP2_19)*100,2)}", None, None, None, None, None, None, None, None, f"{round(((r_mm_CSP2_23/r_nn_CSP2_23)*100 + (r_mm_CSP2_22/r_nn_CSP2_22)*100 + (r_mm_CSP2_21/r_nn_CSP2_21)*100 + (r_mm_CSP2_20/r_nn_CSP2_20)*100 + (r_mm_CSP2_19/r_nn_CSP2_19)*100)/5,1)} +/- 0.44"],
    r'CSP3 $\eta$': [f"{round((r_mm_CSP3_CRNSDistance/r_nn_CSP3_CRNSDistance)*100,2)}",f"{round((r_mm_CSP3_weighted_schron/r_nn_CSP3_weighted_schron)*100, 2)}", f"{round((r_mm_CSP3_weighted_CRNSDistance/r_nn_CSP3_weighted_CRNSDistance)*100,2)}", f"{round((r_mm_CSP3_23/r_nn_CSP3_23)*100, 2)}", f"{round((r_mm_CSP3_22/r_nn_CSP3_22)*100, 2)}", f"{round((r_mm_CSP3_21/r_nn_CSP3_21)*100, 2)}", f"{round((r_mm_CSP3_20/r_nn_CSP3_20)*100, 2)}", f"{round((r_mm_CSP3_19/r_nn_CSP3_19)*100, 2)}", f"{round((r_mm_CSP3_18/r_nn_CSP3_18)*100, 2)}", f"{round((r_mm_CSP3_17/r_nn_CSP3_17)*100, 2)}", f"{round((r_mm_CSP3_16/r_nn_CSP3_16)*100, 2)}", f"{round((r_mm_CSP3_15/r_nn_CSP3_15)*100, 2)}", f"{round((r_mm_CSP3_14/r_nn_CSP3_14)*100, 2)}", f"{round((r_mm_CSP3_13/r_nn_CSP3_13)*100, 2)}", f"{round((r_mm_CSP3_12/r_nn_CSP3_12)*100, 2)}", f"{round((r_mm_CSP3_11/r_nn_CSP3_11)*100, 2)}", f"{round(((r_mm_CSP3_23/r_nn_CSP3_23)*100 + (r_mm_CSP3_22/r_nn_CSP3_22)*100 + (r_mm_CSP3_21/r_nn_CSP3_21)*100 + (r_mm_CSP3_20/r_nn_CSP3_20)*100 + (r_mm_CSP3_19/r_nn_CSP3_19)*100 + (r_mm_CSP3_18/r_nn_CSP3_18)*100 + (r_mm_CSP3_17/r_nn_CSP3_17)*100 + (r_mm_CSP3_16/r_nn_CSP3_16)*100 + (r_mm_CSP3_15/r_nn_CSP3_15)*100 + (r_mm_CSP3_14/r_nn_CSP3_14)*100 + (r_mm_CSP3_13/r_nn_CSP3_13)*100 + (r_mm_CSP3_12/r_nn_CSP3_12)*100 + (r_mm_CSP3_11/r_nn_CSP3_11)*100)/13,2)} +/- 0.41"]
}

df_table = pd.DataFrame(table_data)
fig, ax = plt.subplots(figsize=(10, 5))
table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
ax.axis('off')
table.auto_set_font_size(False)
table.set_fontsize(8)
#table.scale(1, 1)
col_widths = [max(len(str(x)), len(str(y))) for x, y in zip(df_table.columns, df_table.values[0])]
for i, width in enumerate(col_widths):
    table.auto_set_column_width(i)
table.scale(1, 1.7)
plt.savefig('YOUR PATH TO SAVE/Table_of_values.png', dpi=300, bbox_inches='tight')
plt.show()
