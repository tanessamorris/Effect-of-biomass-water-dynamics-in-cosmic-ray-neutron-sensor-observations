#import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
import six
import scipy.stats as stats


############################################
############################################
###################2023#####################
############################################
############################################

############################################
################## CSP1 ####################
############################################

## Using Schrön N0 (intercept) values ##
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1 = pd.read_csv('YOUR PATH/CSP1_Schron_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
SWC_CSP1 = GWC_CSP1['SWC']
a_CSP1 = SWC_CSP1

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
### CALCUATE N ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
#For this, N0 is the intercept
pwc_CSP1 = a_CSP1
lwc = 0.058
soc = 0.0092
N0_CSP1 = 2767.93

#use parameters to calculate N0
N_CSP1 = N0_CSP1*(((0.0808)/((pwc_CSP1+lwc+soc)+0.115))+0.372)
N_CSP1.to_csv('YOUR PATH TO MAKE/N_CSP1.csv')
N_CSP1_23 = pd.read_csv('YOUR PATH/N_CSP1.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP1_2023 = N_CSP1_23['SWC']

#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_weighted_schron = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1 = NPIV_CSP1_weighted_schron['NPIV']

#Calculate forward
fveg_CSP1 = NPIVCount_CSP1 / N_CSP1_2023
fveg_CSP1.to_csv('YOUR PATH TO MAKE/fveg_CSP1.csv')
fveg_CSP1_1 = pd.read_csv('YOUR PATH/fveg_CSP1.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_1 = fveg_CSP1_1['0']

#Import BWE for plotting
BWEcsv_CSP1 = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1 = BWEcsv_CSP1['BWE']

#Set up plotting for both foward and inverse
x_CSP1 = BWE_CSP1
y_CSP1 = 1/fveg_CSP1_1

#Least square regression
A_CSP1 = np.vstack([x_CSP1, np.ones(len(x_CSP1))]).T
y_CSP1 = np.array(y_CSP1) 
y_CSP1 = y_CSP1[:, np.newaxis]
alpha_CSP1 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1.T,A_CSP1)),A_CSP1.T)),y_CSP1)
m_CSP1 = alpha_CSP1[0]
n_CSP1 = alpha_CSP1[1]
mm_CSP1 = float(m_CSP1)
mmm_CSP1 = [float(m_CSP1)]
nn_CSP1 = float(n_CSP1)
r_mm_CSP1 = round(mm_CSP1,4)
r_nn_CSP1 = round(nn_CSP1,4)

#Calculate r-squared
x_CSP1_r2 = sm.add_constant(x_CSP1)
model = sm.OLS(y_CSP1, x_CSP1_r2)
results = model.fit()
r2_CSP1 = results.rsquared
r2_CSP1 = round(r2_CSP1,2)

#Plot data with best fit line, slope, and r-squared
a_CSP1, b_CSP1 = np.polyfit(x_CSP1,y_CSP1,1)
plt.scatter(x_CSP1, y_CSP1, marker = 'o', color = 'red')
plt.plot(x_CSP1, a_CSP1*x_CSP1+b_CSP1, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 Schrön Weighted '+ r'$f_b$'+ ' 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP1,r_nn_CSP1, r2_CSP1]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023fb.png', dpi=300, bbox_inches='tight')
plt.show()


############################################
################## CSP2 ####################
############################################

## Using Schron N0 (intercept) values ##
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP2 = pd.read_csv('YOUR PATH/CSP2_Schron_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
SWC_CSP2 = GWC_CSP2['SWC']
a_CSP2 = SWC_CSP2

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
### CALCUATE N ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
#For this, N0 is the intercept
pwc_CSP2 = a_CSP2
lwc = 0.058
soc = 0.0092
N0_CSP2 = 2864.75

#use parameters to calculate N0
N_CSP2 = N0_CSP2*(((0.0808)/((pwc_CSP2+lwc+soc)+0.115))+0.372)
N_CSP2.to_csv('YOUR PATH TO MAKE/N_CSP2.csv')
N_CSP2_23 = pd.read_csv('YOUR PATH/N_CSP2.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP2_2023 = N_CSP2_23['SWC']

#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_weighted_schron = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2 = NPIV_CSP2_weighted_schron['NPIV']

#Calculate forward
fveg_CSP2 = NPIVCount_CSP2 / N_CSP2_2023
fveg_CSP2.to_csv('YOUR PATH TO MAKE/fveg_CSP2.csv')
fveg_CSP2_1 = pd.read_csv('YOUR PATH/fveg_CSP2.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_1 = fveg_CSP2_1['0']

#Import BWE for plotting
BWEcsv_CSP2 = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2 = BWEcsv_CSP2['BWE']

#Set up plotting for both foward and inverse
x_CSP2 = BWE_CSP2
y_CSP2 = 1/fveg_CSP2_1

#Least square regression
A_CSP2 = np.vstack([x_CSP2, np.ones(len(x_CSP2))]).T
y_CSP2 = np.array(y_CSP2) 
y_CSP2 = y_CSP2[:, np.newaxis]
alpha_CSP2 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2.T,A_CSP2)),A_CSP2.T)),y_CSP2)
m_CSP2 = alpha_CSP2[0]
n_CSP2 = alpha_CSP2[1]
mm_CSP2 = float(m_CSP2)
mmm_CSP2 = [float(m_CSP2)]
nn_CSP2 = float(n_CSP2)
r_mm_CSP2 = round(mm_CSP2,4)
r_nn_CSP2 = round(nn_CSP2,4)

#Calculate r-squared
x_CSP2_r2 = sm.add_constant(x_CSP2)
model = sm.OLS(y_CSP2, x_CSP2_r2)
results = model.fit()
r2_CSP2 = results.rsquared
r2_CSP2 = round(r2_CSP2,2)

#Plot data with best fit line, slope, and r-squared
a_CSP2, b_CSP2 = np.polyfit(x_CSP2,y_CSP2,1)
plt.scatter(x_CSP2, y_CSP2, marker = 'o', color = 'red')
plt.plot(x_CSP2, a_CSP2*x_CSP2+b_CSP2, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 Schrön Weighted '+ r'$f_b$'+ ' 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP2,r_nn_CSP2, r2_CSP2]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2023fb.png', dpi=300, bbox_inches='tight')
plt.show()


############################################
################## CSP3 ####################
############################################

## Using Schron N0 (intercept) values ##
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3 = pd.read_csv('YOUR PATH/CSP3_Schron_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
SWC_CSP3 = GWC_CSP3['SWC']
a_CSP3 = SWC_CSP3

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
### CALCUATE N ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
#For this, N0 is the intercept
pwc_CSP3 = a_CSP3
lwc = 0.058
soc = 0.0092
N0_CSP3 = 1542.57

#use parameters to calculate N0
N_CSP3 = N0_CSP3*(((0.0808)/((pwc_CSP3+lwc+soc)+0.115))+0.372)
N_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3.csv')
N_CSP3_23 = pd.read_csv('YOUR PATH/N_CSP3.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
N_CSP3_2023 = N_CSP3_23['SWC']

#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_weighted_schron = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_23 = NPIV_CSP3_weighted_schron['NPIV']

#Calculate forward
fveg_CSP3 = NPIVCount_CSP3_23 / N_CSP3_2023
fveg_CSP3.to_csv('YOUR PATH TO MAKE/fveg_CSP3.csv')
fveg_CSP3_1 = pd.read_csv('YOUR PATH/fveg_CSP3.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_1 = fveg_CSP3_1['0']

#Import BWE for plotting
BWEcsv_CSP3 = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3 = BWEcsv_CSP3['BWE']

#Set up plotting for both foward and inverse
x_CSP3 = BWE_CSP3
y_CSP3 = 1/fveg_CSP3_1

#Least square regression
A_CSP3 = np.vstack([x_CSP3, np.ones(len(x_CSP3))]).T
y_CSP3 = np.array(y_CSP3) 
y_CSP3 = y_CSP3[:, np.newaxis]
alpha_CSP3 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3.T,A_CSP3)),A_CSP3.T)),y_CSP3)
m_CSP3 = alpha_CSP3[0]
n_CSP3 = alpha_CSP3[1]
mm_CSP3 = float(m_CSP3)
mmm_CSP3 = [float(m_CSP3)]
nn_CSP3 = float(n_CSP3)
r_mm_CSP3 = round(mm_CSP3,4)
r_nn_CSP3 = round(nn_CSP3,4)

#Calculate r-squared
x_CSP3_r2 = sm.add_constant(x_CSP3)
model = sm.OLS(y_CSP3, x_CSP3_r2)
results = model.fit()
r2_CSP3 = results.rsquared
r2_CSP3 = round(r2_CSP3,2)

#Plot data with best fit line, slope, and r-squared
a_CSP3, b_CSP3 = np.polyfit(x_CSP3,y_CSP3,1)
plt.scatter(x_CSP3, y_CSP3, marker = 'o', color = 'red')
plt.plot(x_CSP3, a_CSP3*x_CSP3+b_CSP3, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Schrön Weighted '+ r'$f_b$'+ ' 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP3,r_nn_CSP3, r2_CSP3]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2023fb.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
############ Table of Values #################
##############################################
#Baatz 2015 function 

############################################
################## CSP1 ####################
############################################

fvegbaatz_CSP1 = 1/((1-((-36.22/2767.93)*BWE_CSP1)))
y_baatz_CSP1 = 1/fvegbaatz_CSP1

#Least square regression
A_CSP1_baatz = np.vstack([x_CSP1, np.ones(len(x_CSP1))]).T
y_CSP1_baatz = np.array(y_baatz_CSP1)  
y_CSP1_baatz = y_baatz_CSP1.values[:, np.newaxis]
alpha_CSP1_baatz = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_baatz.T,A_CSP1_baatz)),A_CSP1_baatz.T)),y_CSP1_baatz)
m_CSP1_baatz = alpha_CSP1_baatz[0]
n_CSP1_baatz = alpha_CSP1_baatz[1]
mm_CSP1_baatz = float(m_CSP1_baatz)
mmm_CSP1_baatz = [float(m_CSP1_baatz)]
nn_CSP1_baatz = float(n_CSP1_baatz)
r_mm_CSP1_baatz = round(mm_CSP1_baatz,4)
r_nn_CSP1_baatz = round(nn_CSP1_baatz,4)

#Calculate r-squared   
x_CSP1_baatz = sm.add_constant(x_CSP1)
model = sm.OLS(y_CSP1_baatz, x_CSP1_baatz)
results = model.fit()
r2_CSP1_baatz = results.rsquared
r2_CSP1_baatz = round(r2_CSP1_baatz,2)

#Plot data with best fit line, slope, and r-squared
a_CSP1, b_CSP1 = np.polyfit(x_CSP1,y_CSP1,1)
c_CSP1, d_CSP1 = np.polyfit(x_CSP1,y_baatz_CSP1,1)
plt.scatter(x_CSP1, y_CSP1, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP1, a_CSP1*x_CSP1+b_CSP1, linestyle = '-', color = 'red')
plt.plot(x_CSP1, 0.005*x_CSP1 + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 Schrön Weighted '+ r'$f_b$'+ ' 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP1], [r2_CSP1]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP1_fveg_with_baatz.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
################## CSP2 ####################
############################################

fvegbaatz_CSP2 = 1/((1-((-49.36/2864.75)*BWE_CSP2)))
y_baatz_CSP2 = 1/fvegbaatz_CSP2

#Least square regression
A_CSP2_baatz = np.vstack([x_CSP2, np.ones(len(x_CSP2))]).T
y_CSP2_baatz = np.array(y_baatz_CSP2)  
y_CSP2_baatz = y_baatz_CSP2.values[:, np.newaxis]
alpha_CSP2_baatz = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_baatz.T,A_CSP2_baatz)),A_CSP2_baatz.T)),y_CSP2_baatz)
m_CSP2_baatz = alpha_CSP2_baatz[0]
n_CSP2_baatz = alpha_CSP2_baatz[1]
mm_CSP2_baatz = float(m_CSP2_baatz)
mmm_CSP2_baatz = [float(m_CSP2_baatz)]
nn_CSP2_baatz = float(n_CSP2_baatz)
r_mm_CSP2_baatz = round(mm_CSP2_baatz,4)
r_nn_CSP2_baatz = round(nn_CSP2_baatz,4)

#Calculate r-squared
x_CSP2_baatz = sm.add_constant(x_CSP2)
model = sm.OLS(y_CSP2_baatz, x_CSP2_baatz)
results = model.fit()
r2_CSP2_baatz = results.rsquared
r2_CSP2_baatz = round(r2_CSP2_baatz,2)

#Plot data with best fit line, slope, and r-squared
a_CSP2, b_CSP2 = np.polyfit(x_CSP2,y_CSP2,1)
c_CSP2, d_CSP2 = np.polyfit(x_CSP2,y_baatz_CSP2,1)
plt.scatter(x_CSP2, y_CSP2, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP2, a_CSP2*x_CSP2+b_CSP2, linestyle = '-', color = 'red')
plt.plot(x_CSP2, 0.005*x_CSP2 + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 Schrön Weighted '+ r'$f_b$'+ ' 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP2], [r2_CSP2]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('CYOUR PATH TO SAVE/CSP2_fveg_with_Baatz.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
################## CSP3 ####################
############################################

fvegbaatz_CSP3 = 1/((1-((-11.7/1542.57)*BWE_CSP3)))
y_baatz_CSP3 = 1/fvegbaatz_CSP3

#Least square regression
A_CSP3_baatz = np.vstack([x_CSP3, np.ones(len(x_CSP3))]).T
y_CSP3_baatz = np.array(y_baatz_CSP3)  
y_CSP3_baatz = y_baatz_CSP3.values[:, np.newaxis]
alpha_CSP3_baatz = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_baatz.T,A_CSP3_baatz)),A_CSP3_baatz.T)),y_CSP3_baatz)
m_CSP3_baatz = alpha_CSP3_baatz[0]
n_CSP3_baatz = alpha_CSP3_baatz[1]
mm_CSP3_baatz = float(m_CSP3_baatz)
mmm_CSP3_baatz = [float(m_CSP3_baatz)]
nn_CSP3_baatz = float(n_CSP3_baatz)
r_mm_CSP3_baatz = round(mm_CSP3_baatz,4)
r_nn_CSP3_baatz = round(nn_CSP3_baatz,4)

#Calculate r-squared
x_CSP3_baatz = sm.add_constant(x_CSP3)
model = sm.OLS(y_CSP3_baatz, x_CSP3_baatz)
results = model.fit()
r2_CSP3_baatz = results.rsquared
r2_CSP3_baatz = round(r2_CSP3_baatz,2)

#Plot data with best fit line, slope, and r-squared
a_CSP3, b_CSP3 = np.polyfit(x_CSP3,y_CSP3,1)
c_CSP3, d_CSP3 = np.polyfit(x_CSP3,y_baatz_CSP3,1)
plt.scatter(x_CSP3, y_CSP3, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP3, a_CSP3*x_CSP3+b_CSP3, linestyle = '-', color = 'red')
plt.plot(x_CSP3, 0.005*x_CSP3 + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Schrön Weighted '+ r'$f_b$'+ ' 2023')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3], [r2_CSP3]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_fveg_with_Baatz.png', dpi=300, bbox_inches='tight')
plt.show()

# Normalize according to Baatz 2015
normal_CSP1 = (36.22/2767.93)*100
normal_CSP2 = (49.36/2864.75)*100
normal_CSP3 = (11.7/1542.57)*100

##############################################
############ Table of Values #################
##############################################

data = {
    '': ['Slope', 'Intercept', 'Baatz 2015\nSlope', 'Baatz 2015\nIntercept', 'Slope\nDifference', 'Slope\nRatio', 'Neutron\nIntensity\nReduction(%)'],
    'CSP1': [f"{r_mm_CSP1}", f"{r_nn_CSP1}", f"{r_mm_CSP1_baatz}", f"{r_nn_CSP1_baatz}", f"{round((r_mm_CSP1-r_mm_CSP1_baatz),4)}", f"{round((r_mm_CSP1/r_mm_CSP1_baatz),4)}", f"{round((normal_CSP1),2)}"],
    'CSP2': [f"{r_mm_CSP2}", f"{r_nn_CSP2}", f"{r_mm_CSP2_baatz}", f"{r_nn_CSP2_baatz}", f"{round((r_mm_CSP2-r_mm_CSP2_baatz),4)}", f"{round((r_mm_CSP2/r_mm_CSP2_baatz),4)}", f"{round((normal_CSP2),2)}"],
    'CSP3': [f"{r_mm_CSP3}", f"{r_nn_CSP3}", f"{r_mm_CSP3_baatz}", f"{r_nn_CSP3_baatz}", f"{round((r_mm_CSP3-r_mm_CSP3_baatz),4)}", f"{round((r_mm_CSP3/r_mm_CSP3_baatz),4)}", f"{round((normal_CSP3),2)}"],
    'Average': [f"{round((r_mm_CSP1+r_mm_CSP2+r_mm_CSP3)/3,4)}", f"{round((r_nn_CSP1+r_nn_CSP2+r_nn_CSP3)/3,4)}", f"{round((r_mm_CSP1_baatz+r_mm_CSP2_baatz+r_mm_CSP3_baatz)/3, 4)}", f"{round((r_nn_CSP1_baatz+r_nn_CSP2_baatz+r_nn_CSP3_baatz)/3,4)}", f"{round((r_mm_CSP1-r_mm_CSP1_baatz+r_mm_CSP2-r_mm_CSP2_baatz+r_mm_CSP3-r_mm_CSP3_baatz)/3,4)}", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP3/r_mm_CSP3_baatz)/3,4)}", f"{round((normal_CSP1+normal_CSP2+normal_CSP3)/3,2)}"]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Function to render DataFrame as a table
def render_mpl_table(data, col_width=3.0, row_height=0.7, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(12,7))
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
plt.savefig('YOUR PATH TO SAVE/Summary_Table.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP1 ################################
##################################################################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2019_CSP1 = pd.read_csv('YOUR PATH/AmerifluxCSP12019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2020_CSP1 = pd.read_csv('YOUR PATH/AmerifluxCSP12020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP1 = pd.read_csv('YOUR PATH/AmerifluxCSP12021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP1 = pd.read_csv('YOUR PATH/AmerifluxCSP12022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP1 = pd.read_csv('YOUR PATH/AmerifluxCSP12023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC19_PI_F_1_1_1_CSP1 = GWC2019_CSP1['SWC_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP1 = GWC2019_CSP1['SWC_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP1 = GWC2019_CSP1['SWC_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP1 = GWC2019_CSP1['SWC_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP1 = GWC2019_CSP1['SWC_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP1 = GWC2019_CSP1['SWC_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP1 = GWC2019_CSP1['SWC_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP1 = GWC2019_CSP1['SWC_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP1 = GWC2019_CSP1['SWC_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP1 = GWC2019_CSP1['SWC_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP1 = GWC2019_CSP1['SWC_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP1 = GWC2019_CSP1['SWC_3_4_1']/1.42

SWC20_PI_F_1_1_1_CSP1 = GWC2020_CSP1['SWC_1_1_1']/1.42
SWC20_PI_F_1_2_1_CSP1 = GWC2020_CSP1['SWC_1_2_1']/1.42
SWC20_PI_F_1_3_1_CSP1 = GWC2020_CSP1['SWC_1_3_1']/1.42
SWC20_PI_F_1_4_1_CSP1 = GWC2020_CSP1['SWC_1_4_1']/1.42
SWC20_PI_F_2_1_1_CSP1 = GWC2020_CSP1['SWC_2_1_1']/1.42
SWC20_PI_F_2_2_1_CSP1 = GWC2020_CSP1['SWC_2_2_1']/1.42
SWC20_PI_F_2_3_1_CSP1 = GWC2020_CSP1['SWC_2_3_1']/1.42
SWC20_PI_F_2_4_1_CSP1 = GWC2020_CSP1['SWC_2_4_1']/1.42
SWC20_PI_F_3_1_1_CSP1 = GWC2020_CSP1['SWC_3_1_1']/1.42
SWC20_PI_F_3_2_1_CSP1 = GWC2020_CSP1['SWC_3_2_1']/1.42
SWC20_PI_F_3_3_1_CSP1 = GWC2020_CSP1['SWC_3_3_1']/1.42
SWC20_PI_F_3_4_1_CSP1 = GWC2020_CSP1['SWC_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP1 = GWC2021_CSP1['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP1 = GWC2021_CSP1['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP1 = GWC2021_CSP1['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP1 = GWC2021_CSP1['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP1 = GWC2021_CSP1['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP1 = GWC2021_CSP1['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP1 = GWC2021_CSP1['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP1 = GWC2021_CSP1['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP1 = GWC2021_CSP1['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP1 = GWC2021_CSP1['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_2_1_CSP1 = GWC2021_CSP1['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP1 = GWC2021_CSP1['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP1 = GWC2021_CSP1['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP1 = GWC2022_CSP1['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP1 = GWC2022_CSP1['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP1 = GWC2022_CSP1['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP1 = GWC2022_CSP1['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP1 = GWC2022_CSP1['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP1 = GWC2022_CSP1['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP1 = GWC2022_CSP1['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP1 = GWC2022_CSP1['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP1 = GWC2022_CSP1['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP1 = GWC2022_CSP1['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP1 = GWC2022_CSP1['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP1 = GWC2022_CSP1['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP1 = GWC2023_CSP1['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP1 = GWC2023_CSP1['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP1 = GWC2023_CSP1['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP1 = GWC2023_CSP1['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP1 = GWC2023_CSP1['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP1 = GWC2023_CSP1['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP1 = GWC2023_CSP1['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP1 = GWC2023_CSP1['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP1 = GWC2023_CSP1['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP1 = GWC2023_CSP1['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP1 = GWC2023_CSP1['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP1 = GWC2023_CSP1['SWC_F_3_4_1']/1.42

twosenwavgall2019_CSP1 = ((((SWC19_PI_F_1_1_1_CSP1 + SWC19_PI_F_2_1_1_CSP1 + SWC19_PI_F_3_1_1_CSP1)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP1 + SWC19_PI_F_2_2_1_CSP1 + SWC19_PI_F_3_2_1_CSP1)/3)*0.25)) / 100
twosenwavgall2020_CSP1 = ((((SWC20_PI_F_1_1_1_CSP1 + SWC20_PI_F_2_1_1_CSP1 + SWC20_PI_F_3_1_1_CSP1)/3)*0.75) + (((SWC20_PI_F_1_2_1_CSP1 + SWC20_PI_F_2_2_1_CSP1 + SWC20_PI_F_3_2_1_CSP1)/3)*0.25)) / 100
twosenwavgall2021_CSP1 = ((((SWC21_PI_F_1_1_1_CSP1 + SWC21_PI_F_2_1_1_CSP1 + SWC21_PI_F_3_1_1_CSP1)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP1 + SWC21_PI_F_2_2_1_CSP1 + SWC21_PI_F_3_2_1_CSP1)/3)*0.25)) / 100
twosenwavgall2022_CSP1 = ((((SWC22_PI_F_1_1_1_CSP1 + SWC22_PI_F_2_1_1_CSP1 + SWC22_PI_F_3_1_1_CSP1)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP1 + SWC22_PI_F_2_2_1_CSP1 + SWC22_PI_F_3_2_1_CSP1)/3)*0.25)) / 100
twosenwavgall2023_CSP1 = ((((SWC23_PI_F_1_1_1_CSP1 + SWC23_PI_F_2_1_1_CSP1 + SWC23_PI_F_3_1_1_CSP1)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP1 + SWC23_PI_F_2_2_1_CSP1 + SWC23_PI_F_3_2_1_CSP1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2019_CSP1 = twosenwavgall2019_CSP1.resample('D').mean() 
dailywavgSWC2020_CSP1 = twosenwavgall2020_CSP1.resample('D').mean() 
dailywavgSWC2021_CSP1 = twosenwavgall2021_CSP1.resample('D').mean() 
dailywavgSWC2022_CSP1 = twosenwavgall2022_CSP1.resample('D').mean() 
dailywavgSWC2023_CSP1 = twosenwavgall2023_CSP1.resample('D').mean()

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2019_CSP1 = pd.read_csv('YOUR PATH/CSP12019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2020_CSP1 = pd.read_csv('YOUR PATH/CSP12020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP1 = pd.read_csv('YOUR PATH/CSP12021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP1 = pd.read_csv('YOUR PATH/CSP12022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP1 = pd.read_csv('YOUR PATH/CSP12023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2019_CSP1 = NPIV2019_CSP1['NPIV']
NPIVCount2020_CSP1 = NPIV2020_CSP1['NPIV']
NPIVCount2021_CSP1 = NPIV2021_CSP1['NPIV']
NPIVCount2022_CSP1 = NPIV2022_CSP1['NPIV']
NPIVCount2023_CSP1 = NPIV2023_CSP1['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2019_CSP1 = NPIVCount2019_CSP1.resample('D').mean()
dailyNPIV2020_CSP1 = NPIVCount2020_CSP1.resample('D').mean()
dailyNPIV2021_CSP1 = NPIVCount2021_CSP1.resample('D').mean()
dailyNPIV2022_CSP1 = NPIVCount2022_CSP1.resample('D').mean()
dailyNPIV2023_CSP1 = NPIVCount2023_CSP1.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc19_CSP1 = dailywavgSWC2019_CSP1
pwc20_CSP1 = dailywavgSWC2020_CSP1
pwc21_CSP1 = dailywavgSWC2021_CSP1
pwc22_CSP1 = dailywavgSWC2022_CSP1
pwc23_CSP1 = dailywavgSWC2023_CSP1

lwc = 0.058
soc = 0.0092
N0_for_calc = 2767.93

N19_CSP1 = dailyNPIV2019_CSP1
N20_CSP1 = dailyNPIV2020_CSP1
N21_CSP1 = dailyNPIV2021_CSP1
N22_CSP1 = dailyNPIV2022_CSP1
N23_CSP1 = dailyNPIV2023_CSP1

#Calculate N
N0_CSP1_19 = N0_for_calc*(((0.0808)/((pwc19_CSP1+lwc+soc)+0.115))+0.372)
N0_CSP1_20 = N0_for_calc*(((0.0808)/((pwc20_CSP1+lwc+soc)+0.115))+0.372)
N0_CSP1_21 = N0_for_calc*(((0.0808)/((pwc21_CSP1+lwc+soc)+0.115))+0.372)
N0_CSP1_22 = N0_for_calc*(((0.0808)/((pwc22_CSP1+lwc+soc)+0.115))+0.372)
N0_CSP1_23 = N0_for_calc*(((0.0808)/((pwc23_CSP1+lwc+soc)+0.115))+0.372)

NCSP1_19= N0_CSP1_19.to_csv('YOUR PATH TO SAVE/N_CSP1_19.csv')
NCSP1_20= N0_CSP1_20.to_csv('YOUR PATH TO SAVE/N_CSP1_20.csv')
NCSP1_21= N0_CSP1_21.to_csv('YOUR PATH TO SAVE/N_CSP1_21.csv')
NCSP1_22= N0_CSP1_22.to_csv('YOUR PATH TO SAVE/N_CSP1_22.csv')
NCSP1_23= N0_CSP1_23.to_csv('YOUR PATH TO SAVE/N_CSP1_23.csv')

CSP1_19 = pd.read_csv('YOUR PATH/N_CSP1_19.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_20 = pd.read_csv('YOUR PATH/N_CSP1_20.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_21 = pd.read_csv('YOUR PATH/N_CSP1_21.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_22 = pd.read_csv('YOUR PATH/N_CSP1_22.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_23 = pd.read_csv('YOUR PATH/N_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP1_19_1 = CSP1_19['0']
N_CSP1_20_1 = CSP1_20['0']
N_CSP1_21_1 = CSP1_21['0']
N_CSP1_22_1 = CSP1_22['0']
N_CSP1_23_1 = CSP1_23['0']

fveg_CSP1_19 = dailyNPIV2019_CSP1 / N0_CSP1_19
fveg_CSP1_20 = dailyNPIV2020_CSP1 / N0_CSP1_20
fveg_CSP1_21 = dailyNPIV2021_CSP1 / N0_CSP1_21
fveg_CSP1_22 = dailyNPIV2022_CSP1 / N0_CSP1_22
fveg_CSP1_23 = dailyNPIV2023_CSP1 / N0_CSP1_23

fveg_CSP1_19.to_csv('YOUR PATH TO SAVE/fveg_CSP1_19.csv')
fveg_CSP1_20.to_csv('YOUR PATH TO SAVE/fveg_CSP1_20.csv')
fveg_CSP1_21.to_csv('YOUR PATH TO SAVE/fveg_CSP1_21.csv')
fveg_CSP1_22.to_csv('YOUR PATH TO SAVE/fveg_CSP1_22.csv')
fveg_CSP1_23.to_csv('YOUR PATH TO SAVE/fveg_CSP1_23.csv')

fveg_CSP1_19_1 = pd.read_csv('YOUR PATH/fveg_CSP1_19.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_20_1 = pd.read_csv('YOUR PATH/fveg_CSP1_20.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_21_1 = pd.read_csv('YOUR PATH/fveg_CSP1_21.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_22_1 = pd.read_csv('YOUR PATH/fveg_CSP1_22.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_23_1 = pd.read_csv('YOUR PATH/fveg_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP1_19_1 = fveg_CSP1_19_1['0']
fveg_CSP1_20_1 = fveg_CSP1_20_1['0']
fveg_CSP1_21_1 = fveg_CSP1_21_1['0']
fveg_CSP1_22_1 = fveg_CSP1_22_1['0']
fveg_CSP1_23_1 = fveg_CSP1_23_1['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2019_CSP1 = pd.read_csv('YOUR PATH/CSP12019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP1 = pd.read_csv('YOUR PATH/CSP12020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP1 = pd.read_csv('YOUR PATH/CSP12021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP1 = pd.read_csv('YOUR PATH/CSP12022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP1 = pd.read_csv('YOUR PATH/CSP12023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE19_CSP1 = BWEcsv2019_CSP1['BWE']
BWE20_CSP1 = BWEcsv2020_CSP1['BWE']
BWE21_CSP1 = BWEcsv2021_CSP1['BWE']
BWE22_CSP1 = BWEcsv2022_CSP1['BWE']
BWE23_CSP1 = BWEcsv2023_CSP1['BWE']

#Interpolate between dates
BWEinterp19_CSP1 = BWE19_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP1 = BWE20_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP1 = BWE21_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP1 = BWE22_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP1 = BWE23_CSP1.interpolate(method = 'linear', limit_direction = 'forward')

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP1 = BWEinterp19_CSP1
y19_CSP1 = 1/fveg_CSP1_19_1
x20_CSP1 = BWEinterp20_CSP1
y20_CSP1 = 1/fveg_CSP1_20_1
x21_CSP1 = BWEinterp21_CSP1
y21_CSP1 = 1/fveg_CSP1_21_1
x22_CSP1 = BWEinterp22_CSP1
y22_CSP1 = 1/fveg_CSP1_22_1
x23_CSP1 = BWEinterp23_CSP1
y23_CSP1 = 1/fveg_CSP1_23_1

# Concatenate all x and y values
x_all_CSP1 = np.concatenate([x19_CSP1, x20_CSP1, x21_CSP1, x22_CSP1, x23_CSP1])
y_all_CSP1 = np.concatenate([y19_CSP1, y20_CSP1, y21_CSP1, y22_CSP1, y23_CSP1])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP1 = x_all_CSP1.reshape(-1, 1)

# Perform linear regression
slope_CSP1, intercept_CSP1, r_value_CSP1, p_value_CSP1, std_err_CSP1 = stats.linregress(x_all_CSP1, y_all_CSP1)

# Calculate the t-value for a 95% confidence interval
t_CSP1 = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP1) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP1 = t_CSP1 * std_err_CSP1
intercept_conf_interval_CSP1 = t_CSP1 * std_err_CSP1 * np.sqrt((1/len(x_all_CSP1)) + (np.mean(x_all_CSP1)**2 / np.sum((x_all_CSP1 - np.mean(x_all_CSP1))**2)))

# Calculate R-squared value
r_squared_CSP1 = r_value_CSP1**2

shift = 1 - intercept_CSP1

plt.scatter(x19_CSP1, y19_CSP1 + shift, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP1, y20_CSP1 + shift, color = 'cyan', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP1, y21_CSP1 + shift, color = 'darkgreen', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP1, y22_CSP1 + shift, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP1, y23_CSP1 + shift, color = 'darkorange', marker = 'D', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP1, b_CSP1 = np.polyfit(x_CSP1,y_CSP1,1)
c_CSP1, d_CSP1 = np.polyfit(x_CSP1,y_baatz_CSP1,1)
plt.scatter(x_CSP1, y_CSP1, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
#plt.scatter(x_CSP1, y_baatz_CSP1, marker = 'o', color = 'blue', label = r'$f_b$ by\nBaatz et al. (2015)')
plt.plot(x_CSP1, 0.005*x_CSP1 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP1, a_CSP1*x_CSP1+b_CSP1, linestyle = '-', color = 'red', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 Schrön Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP1, round(slope_CSP1,2)], [r2_CSP1, round(r_squared_CSP1, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP1_with_TDR.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP2 ################################
##################################################################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2019_CSP2 = pd.read_csv('YOUR PATH/AmerifluxCSP22019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2020_CSP2 = pd.read_csv('YOUR PATH/AmerifluxCSP22020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP2 = pd.read_csv('YOUR PATH/AmerifluxCSP22021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP2 = pd.read_csv('YOUR PATH/AmerifluxCSP22022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP2 = pd.read_csv('YOUR PATH/AmerifluxCSP22023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC19_PI_F_1_1_1_CSP2 = GWC2019_CSP2['SWC_PI_F_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP2 = GWC2019_CSP2['SWC_PI_F_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP2 = GWC2019_CSP2['SWC_PI_F_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP2 = GWC2019_CSP2['SWC_PI_F_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP2 = GWC2019_CSP2['SWC_PI_F_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP2 = GWC2019_CSP2['SWC_PI_F_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP2 = GWC2019_CSP2['SWC_PI_F_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP2 = GWC2019_CSP2['SWC_PI_F_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP2 = GWC2019_CSP2['SWC_PI_F_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP2 = GWC2019_CSP2['SWC_PI_F_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP2 = GWC2019_CSP2['SWC_PI_F_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP2 = GWC2019_CSP2['SWC_PI_F_3_4_1']/1.42

SWC20_PI_F_1_1_1_CSP2 = GWC2020_CSP2['SWC_PI_F_1_1_1']/1.42
SWC20_PI_F_1_2_1_CSP2 = GWC2020_CSP2['SWC_PI_F_1_2_1']/1.42
SWC20_PI_F_1_3_1_CSP2 = GWC2020_CSP2['SWC_PI_F_1_3_1']/1.42
SWC20_PI_F_1_4_1_CSP2 = GWC2020_CSP2['SWC_PI_F_1_4_1']/1.42
SWC20_PI_F_2_1_1_CSP2 = GWC2020_CSP2['SWC_PI_F_2_1_1']/1.42
SWC20_PI_F_2_2_1_CSP2 = GWC2020_CSP2['SWC_PI_F_2_2_1']/1.42
SWC20_PI_F_2_3_1_CSP2 = GWC2020_CSP2['SWC_PI_F_2_3_1']/1.42
SWC20_PI_F_2_4_1_CSP2 = GWC2020_CSP2['SWC_PI_F_2_4_1']/1.42
SWC20_PI_F_3_1_1_CSP2 = GWC2020_CSP2['SWC_PI_F_3_1_1']/1.42
SWC20_PI_F_3_2_1_CSP2 = GWC2020_CSP2['SWC_PI_F_3_2_1']/1.42
SWC20_PI_F_3_3_1_CSP2 = GWC2020_CSP2['SWC_PI_F_3_3_1']/1.42
SWC20_PI_F_3_4_1_CSP2 = GWC2020_CSP2['SWC_PI_F_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP2 = GWC2021_CSP2['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP2 = GWC2021_CSP2['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP2 = GWC2021_CSP2['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP2 = GWC2021_CSP2['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP2 = GWC2021_CSP2['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP2 = GWC2021_CSP2['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP2 = GWC2021_CSP2['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP2 = GWC2021_CSP2['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP2 = GWC2021_CSP2['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP2 = GWC2021_CSP2['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP2 = GWC2021_CSP2['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP2 = GWC2021_CSP2['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP2 = GWC2022_CSP2['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP2 = GWC2022_CSP2['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP2 = GWC2022_CSP2['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP2 = GWC2022_CSP2['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP2 = GWC2022_CSP2['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP2 = GWC2022_CSP2['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP2 = GWC2022_CSP2['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP2 = GWC2022_CSP2['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP2 = GWC2022_CSP2['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP2 = GWC2022_CSP2['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP2 = GWC2022_CSP2['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP2 = GWC2022_CSP2['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP2 = GWC2023_CSP2['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP2 = GWC2023_CSP2['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP2 = GWC2023_CSP2['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP2 = GWC2023_CSP2['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP2 = GWC2023_CSP2['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP2 = GWC2023_CSP2['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP2 = GWC2023_CSP2['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP2 = GWC2023_CSP2['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP2 = GWC2023_CSP2['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP2 = GWC2023_CSP2['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP2 = GWC2023_CSP2['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP2 = GWC2023_CSP2['SWC_F_3_4_1']/1.42

twosenwavgall2019_CSP2 = ((((SWC19_PI_F_1_1_1_CSP2 + SWC19_PI_F_2_1_1_CSP2 + SWC19_PI_F_3_1_1_CSP2)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP2 + SWC19_PI_F_2_2_1_CSP2 + SWC19_PI_F_3_2_1_CSP2)/3)*0.25)) / 100
twosenwavgall2020_CSP2 = ((((SWC20_PI_F_1_1_1_CSP2 + SWC20_PI_F_2_1_1_CSP2 + SWC20_PI_F_3_1_1_CSP2)/3)*0.75) + (((SWC20_PI_F_1_2_1_CSP2 + SWC20_PI_F_2_2_1_CSP2 + SWC20_PI_F_3_2_1_CSP2)/3)*0.25)) / 100
twosenwavgall2021_CSP2 = ((((SWC21_PI_F_1_1_1_CSP2 + SWC21_PI_F_2_1_1_CSP2 + SWC21_PI_F_3_1_1_CSP2)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP2 + SWC21_PI_F_2_2_1_CSP2 + SWC21_PI_F_3_2_1_CSP2)/3)*0.25)) / 100
twosenwavgall2022_CSP2 = ((((SWC22_PI_F_1_1_1_CSP2 + SWC22_PI_F_2_1_1_CSP2 + SWC22_PI_F_3_1_1_CSP2)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP2 + SWC22_PI_F_2_2_1_CSP2 + SWC22_PI_F_3_2_1_CSP2)/3)*0.25)) / 100
twosenwavgall2023_CSP2 = ((((SWC23_PI_F_1_1_1_CSP2 + SWC23_PI_F_2_1_1_CSP2 + SWC23_PI_F_3_1_1_CSP2)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP2 + SWC23_PI_F_2_2_1_CSP2 + SWC23_PI_F_3_2_1_CSP2)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2019_CSP2 = twosenwavgall2019_CSP2.resample('D').mean() 
dailywavgSWC2020_CSP2 = twosenwavgall2020_CSP2.resample('D').mean() 
dailywavgSWC2021_CSP2 = twosenwavgall2021_CSP2.resample('D').mean() 
dailywavgSWC2022_CSP2 = twosenwavgall2022_CSP2.resample('D').mean() 
dailywavgSWC2023_CSP2 = twosenwavgall2023_CSP2.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2019_CSP2 = pd.read_csv('YOUR PATH/CSP22019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2020_CSP2 = pd.read_csv('YOUR PATH/CSP22020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP2 = pd.read_csv('YOUR PATH/CSP22021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP2 = pd.read_csv('YOUR PATH/CSP22022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP2 = pd.read_csv('YOUR PATH/CSP22023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2019_CSP2 = NPIV2019_CSP2['NPIV']
NPIVCount2020_CSP2 = NPIV2020_CSP2['NPIV']
NPIVCount2021_CSP2 = NPIV2021_CSP2['NPIV']
NPIVCount2022_CSP2 = NPIV2022_CSP2['NPIV']
NPIVCount2023_CSP2 = NPIV2023_CSP2['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2019_CSP2 = NPIVCount2019_CSP2.resample('D').mean()
dailyNPIV2020_CSP2 = NPIVCount2020_CSP2.resample('D').mean()
dailyNPIV2021_CSP2 = NPIVCount2021_CSP2.resample('D').mean()
dailyNPIV2022_CSP2 = NPIVCount2022_CSP2.resample('D').mean()
dailyNPIV2023_CSP2 = NPIVCount2023_CSP2.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc19_CSP2 = dailywavgSWC2019_CSP2
pwc20_CSP2 = dailywavgSWC2020_CSP2
pwc21_CSP2 = dailywavgSWC2021_CSP2
pwc22_CSP2 = dailywavgSWC2022_CSP2
pwc23_CSP2 = dailywavgSWC2023_CSP2

N19_CSP2 = dailyNPIV2019_CSP2
N20_CSP2 = dailyNPIV2020_CSP2
N21_CSP2 = dailyNPIV2021_CSP2
N22_CSP2 = dailyNPIV2022_CSP2
N23_CSP2 = dailyNPIV2023_CSP2

N0_for_calc_2 = 2864.75
lwc = 0.058
soc = 0.0092

#Calculate N
N0_CSP2_19 = N0_for_calc_2*(((0.0808)/((pwc19_CSP2+lwc+soc)+0.115))+0.372)
N0_CSP2_20 = N0_for_calc_2*(((0.0808)/((pwc20_CSP2+lwc+soc)+0.115))+0.372)
N0_CSP2_21 = N0_for_calc_2*(((0.0808)/((pwc21_CSP2+lwc+soc)+0.115))+0.372)
N0_CSP2_22 = N0_for_calc_2*(((0.0808)/((pwc22_CSP2+lwc+soc)+0.115))+0.372)
N0_CSP2_23 = N0_for_calc_2*(((0.0808)/((pwc23_CSP2+lwc+soc)+0.115))+0.372)

N0interp19_CSP2 = N0_CSP2_19.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP2 = N0_CSP2_20.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP2 = N0_CSP2_21.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP2 = N0_CSP2_22.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP2 = N0_CSP2_23.interpolate(method = 'linear', limit_direction = 'forward')

NCSP2_19= N0interp19_CSP2.to_csv('YOUR PATH TO MAKE/N_CSP2_19.csv')
NCSP2_20= N0interp20_CSP2.to_csv('YOUR PATH TO MAKE/N_CSP2_20.csv')
NCSP2_21= N0interp21_CSP2.to_csv('YOUR PATH TO MAKE/N_CSP2_21.csv')
NCSP2_22= N0interp22_CSP2.to_csv('YOUR PATH TO MAKE/N_CSP2_22.csv')
NCSP2_23= N0interp23_CSP2.to_csv('YOUR PATH TO MAKE/N_CSP2_23.csv')

CSP2_19 = pd.read_csv('YOUR PATH/N_CSP2_19.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_20 = pd.read_csv('YOUR PATH/N_CSP2_20.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_21 = pd.read_csv('YOUR PATH/N_CSP2_21.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_22 = pd.read_csv('YOUR PATH/N_CSP2_22.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_23 = pd.read_csv('YOUR PATH/N_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP2_19_1 = CSP2_19['0']
N_CSP2_20_1 = CSP2_20['0']
N_CSP2_21_1 = CSP2_21['0']
N_CSP2_22_1 = CSP2_22['0']
N_CSP2_23_1 = CSP2_23['0']

fveg_CSP2_19 = dailyNPIV2019_CSP2 / N0_CSP2_19
fveg_CSP2_20 = dailyNPIV2020_CSP2 / N0_CSP2_20
fveg_CSP2_21 = dailyNPIV2021_CSP2 / N0_CSP2_21
fveg_CSP2_22 = dailyNPIV2022_CSP2 / N0_CSP2_22
fveg_CSP2_23_11 = dailyNPIV2023_CSP2 / N0_CSP2_23

fveg_CSP2_19.to_csv('YOUR PATH TO MAKE/fveg_CSP2_19.csv')
fveg_CSP2_20.to_csv('YOUR PATH TO MAKE/fveg_CSP2_20.csv')
fveg_CSP2_21.to_csv('YOUR PATH TO MAKE/fveg_CSP2_21.csv')
fveg_CSP2_22.to_csv('YOUR PATH TO MAKE/fveg_CSP2_22.csv')
fveg_CSP2_23_11.to_csv('YOUR PATH TO MAKE/fveg_CSP2_23.csv')

fveg_CSP2_19_1 = pd.read_csv('YOUR PATH/fveg_CSP2_19.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_20_1 = pd.read_csv('YOUR PATH/fveg_CSP2_20.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_21_1 = pd.read_csv('YOUR PATH/fveg_CSP2_21.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_22_1 = pd.read_csv('YOUR PATH/fveg_CSP2_22.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_23_1 = pd.read_csv('YOUR PATH/fveg_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP2_19_1 = fveg_CSP2_19_1['0']
fveg_CSP2_20_1 = fveg_CSP2_20_1['0']
fveg_CSP2_21_1 = fveg_CSP2_21_1['0']
fveg_CSP2_22_1 = fveg_CSP2_22_1['0']
fveg_CSP2_23_1 = fveg_CSP2_23_1['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2019_CSP2 = pd.read_csv('YOUR PATH/CSP22019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP2 = pd.read_csv('YOUR PATH/CSP22020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP2 = pd.read_csv('YOUR PATH/CSP22021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP2 = pd.read_csv('YOUR PATH/CSP22022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP2 = pd.read_csv('YOUR PATH/CSP22023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE19_CSP2 = BWEcsv2019_CSP2['BWE']
BWE20_CSP2 = BWEcsv2020_CSP2['BWE']
BWE21_CSP2 = BWEcsv2021_CSP2['BWE']
BWE22_CSP2 = BWEcsv2022_CSP2['BWE']
BWE23_CSP2 = BWEcsv2023_CSP2['BWE']

#Interpolate between dates
BWEinterp19_CSP2 = BWE19_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP2 = BWE20_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP2 = BWE21_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP2 = BWE22_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP2 = BWE23_CSP2.interpolate(method = 'linear', limit_direction = 'forward')

fveg_CSP2_19_1 = fveg_CSP2_19_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_20_1 = fveg_CSP2_20_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_21_1 = fveg_CSP2_21_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_22_1 = fveg_CSP2_22_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_23_1 = fveg_CSP2_23_1.replace(np.inf, np.nan).replace(-np.inf, np.nan)

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP2 = BWEinterp19_CSP2
y19_CSP2 = 1/fveg_CSP2_19_1
x20_CSP2 = BWEinterp20_CSP2
y20_CSP2 = 1/fveg_CSP2_20_1
x21_CSP2 = BWEinterp21_CSP2
y21_CSP2 = 1/fveg_CSP2_21_1
x22_CSP2 = BWEinterp22_CSP2
y22_CSP2 = 1/fveg_CSP2_22_1
x23_CSP2 = BWEinterp23_CSP2
y23_CSP2 = 1/fveg_CSP2_23_1

def fill_nan_with_avg(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr
y23_CSP2 = fill_nan_with_avg(y23_CSP2)
# Concatenate all x and y values
x_all_CSP2 = np.concatenate([x19_CSP2, x20_CSP2, x21_CSP2, x22_CSP2, x23_CSP2])
y_all_CSP2 = np.concatenate([y19_CSP2, y20_CSP2, y21_CSP2, y22_CSP2, y23_CSP2])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP2 = x_all_CSP2.reshape(-1, 1)

# Perform linear regression
slope_CSP2, intercept_CSP2, r_value_CSP2, p_value_CSP2, std_err_CSP2 = stats.linregress(x_all_CSP2, y_all_CSP2)

# Calculate the t-value for a 95% confidence interval
t_CSP2 = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP2) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP2 = t_CSP2 * std_err_CSP2
intercept_conf_interval_CSP2 = t_CSP2 * std_err_CSP2 * np.sqrt((1/len(x_all_CSP2)) + (np.mean(x_all_CSP2)**2 / np.sum((x_all_CSP2 - np.mean(x_all_CSP2))**2)))

# Calculate R-squared value
r_squared_CSP2 = r_value_CSP2**2

shift_CSP2 = 1 - intercept_CSP2

plt.scatter(x19_CSP2, y19_CSP2 + shift_CSP2, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP2, y20_CSP2 + shift_CSP2, color = 'cyan', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP2, y21_CSP2 + shift_CSP2, color = 'darkgreen', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP2, y22_CSP2 + shift_CSP2, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP2, y23_CSP2 + shift_CSP2, color = 'darkorange', marker = 'D', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP2, b_CSP2 = np.polyfit(x_CSP2,y_CSP2,1)
c_CSP2, d_CSP2 = np.polyfit(x_CSP2,y_baatz_CSP2,1)
plt.scatter(x_CSP2, y_CSP2, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
#plt.scatter(x_CSP2, y_baatz_CSP2, marker = 'o', color = 'blue', label = r'$f_b$ by\nBaatz et al. (2015)')
plt.plot(x_CSP2, 0.005*x_CSP2 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP2, a_CSP2*x_CSP2+b_CSP2, linestyle = '-', color = 'red', linewidth=2.0)

plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 Schrön Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP2, round(slope_CSP2,2)], [r2_CSP2, round(r_squared_CSP2, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP2_with_TDR.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
##################################################################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2011_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2012_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2013_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2014_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2015_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2016_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2017_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2018_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2019_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2020_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP3 = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC11_PI_F_1_1_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC11_PI_F_1_2_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC11_PI_F_1_3_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC11_PI_F_1_4_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_5_1']/1.42
SWC11_PI_F_2_1_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC11_PI_F_2_2_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC11_PI_F_2_3_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC11_PI_F_2_4_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_5_1']/1.42
SWC11_PI_F_3_1_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC11_PI_F_3_2_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC11_PI_F_3_3_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC11_PI_F_3_4_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_3_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3 = GWC2011_CSP3_1['SWC_PI_F_1_5_1']/1.42

SWC12_PI_F_1_1_1_CSP3 = GWC2012_CSP3['SWC_PI_F_1_1_1']/1.42
SWC12_PI_F_1_2_1_CSP3 = GWC2012_CSP3['SWC_PI_F_1_2_1']/1.42
SWC12_PI_F_1_3_1_CSP3 = GWC2012_CSP3['SWC_PI_F_1_3_1']/1.42
SWC12_PI_F_1_4_1_CSP3 = GWC2012_CSP3['SWC_PI_F_1_4_1']/1.42
SWC12_PI_F_2_1_1_CSP3 = GWC2012_CSP3['SWC_PI_F_2_1_1']/1.42
SWC12_PI_F_2_2_1_CSP3 = GWC2012_CSP3['SWC_PI_F_2_2_1']/1.42
SWC12_PI_F_2_3_1_CSP3 = GWC2012_CSP3['SWC_PI_F_2_3_1']/1.42
SWC12_PI_F_2_4_1_CSP3 = GWC2012_CSP3['SWC_PI_F_2_4_1']/1.42
SWC12_PI_F_3_1_1_CSP3 = GWC2012_CSP3['SWC_PI_F_3_1_1']/1.42
SWC12_PI_F_3_2_1_CSP3 = GWC2012_CSP3['SWC_PI_F_3_2_1']/1.42
SWC12_PI_F_3_3_1_CSP3 = GWC2012_CSP3['SWC_PI_F_3_3_1']/1.42
SWC12_PI_F_3_4_1_CSP3 = GWC2012_CSP3['SWC_PI_F_3_4_1']/1.42

SWC13_PI_F_1_1_1_CSP3 = GWC2013_CSP3['SWC_PI_F_1_1_1']/1.42
SWC13_PI_F_1_2_1_CSP3 = GWC2013_CSP3['SWC_PI_F_1_2_1']/1.42
SWC13_PI_F_1_3_1_CSP3 = GWC2013_CSP3['SWC_PI_F_1_3_1']/1.42
SWC13_PI_F_1_4_1_CSP3 = GWC2013_CSP3['SWC_PI_F_1_4_1']/1.42
SWC13_PI_F_2_1_1_CSP3 = GWC2013_CSP3['SWC_PI_F_2_1_1']/1.42
SWC13_PI_F_2_2_1_CSP3 = GWC2013_CSP3['SWC_PI_F_2_2_1']/1.42
SWC13_PI_F_2_3_1_CSP3 = GWC2013_CSP3['SWC_PI_F_2_3_1']/1.42
SWC13_PI_F_2_4_1_CSP3 = GWC2013_CSP3['SWC_PI_F_2_4_1']/1.42
SWC13_PI_F_3_1_1_CSP3 = GWC2013_CSP3['SWC_PI_F_3_1_1']/1.42
SWC13_PI_F_3_2_1_CSP3 = GWC2013_CSP3['SWC_PI_F_3_2_1']/1.42
SWC13_PI_F_3_3_1_CSP3 = GWC2013_CSP3['SWC_PI_F_3_3_1']/1.42
SWC13_PI_F_3_4_1_CSP3 = GWC2013_CSP3['SWC_PI_F_3_4_1']/1.42

SWC14_PI_F_1_1_1_CSP3 = GWC2014_CSP3['SWC_PI_F_1_1_1']/1.42
SWC14_PI_F_1_2_1_CSP3 = GWC2014_CSP3['SWC_PI_F_1_2_1']/1.42
SWC14_PI_F_1_3_1_CSP3 = GWC2014_CSP3['SWC_PI_F_1_3_1']/1.42
SWC14_PI_F_1_4_1_CSP3 = GWC2014_CSP3['SWC_PI_F_1_4_1']/1.42
SWC14_PI_F_2_1_1_CSP3 = GWC2014_CSP3['SWC_PI_F_2_1_1']/1.42
SWC14_PI_F_2_2_1_CSP3 = GWC2014_CSP3['SWC_PI_F_2_2_1']/1.42
SWC14_PI_F_2_3_1_CSP3 = GWC2014_CSP3['SWC_PI_F_2_3_1']/1.42
SWC14_PI_F_2_4_1_CSP3 = GWC2014_CSP3['SWC_PI_F_2_4_1']/1.42
SWC14_PI_F_3_1_1_CSP3 = GWC2014_CSP3['SWC_PI_F_3_1_1']/1.42
SWC14_PI_F_3_2_1_CSP3 = GWC2014_CSP3['SWC_PI_F_3_2_1']/1.42
SWC14_PI_F_3_3_1_CSP3 = GWC2014_CSP3['SWC_PI_F_3_3_1']/1.42
SWC14_PI_F_3_4_1_CSP3 = GWC2014_CSP3['SWC_PI_F_3_4_1']/1.42

SWC15_PI_F_1_1_1_CSP3 = GWC2015_CSP3['SWC_PI_F_1_1_1']/1.42
SWC15_PI_F_1_2_1_CSP3 = GWC2015_CSP3['SWC_PI_F_1_2_1']/1.42
SWC15_PI_F_1_3_1_CSP3 = GWC2015_CSP3['SWC_PI_F_1_3_1']/1.42
SWC15_PI_F_1_4_1_CSP3 = GWC2015_CSP3['SWC_PI_F_1_4_1']/1.42
SWC15_PI_F_2_1_1_CSP3 = GWC2015_CSP3['SWC_PI_F_2_1_1']/1.42
SWC15_PI_F_2_2_1_CSP3 = GWC2015_CSP3['SWC_PI_F_2_2_1']/1.42
SWC15_PI_F_2_3_1_CSP3 = GWC2015_CSP3['SWC_PI_F_2_3_1']/1.42
SWC15_PI_F_2_4_1_CSP3 = GWC2015_CSP3['SWC_PI_F_2_4_1']/1.42
SWC15_PI_F_3_1_1_CSP3 = GWC2015_CSP3['SWC_PI_F_3_1_1']/1.42
SWC15_PI_F_3_2_1_CSP3 = GWC2015_CSP3['SWC_PI_F_3_2_1']/1.42
SWC15_PI_F_3_3_1_CSP3 = GWC2015_CSP3['SWC_PI_F_3_3_1']/1.42
SWC15_PI_F_3_4_1_CSP3 = GWC2015_CSP3['SWC_PI_F_3_4_1']/1.42

SWC16_PI_F_1_1_1_CSP3 = GWC2016_CSP3['SWC_PI_F_1_1_1']/1.42
SWC16_PI_F_1_2_1_CSP3 = GWC2016_CSP3['SWC_PI_F_1_2_1']/1.42
SWC16_PI_F_1_3_1_CSP3 = GWC2016_CSP3['SWC_PI_F_1_3_1']/1.42
SWC16_PI_F_1_4_1_CSP3 = GWC2016_CSP3['SWC_PI_F_1_4_1']/1.42
SWC16_PI_F_2_1_1_CSP3 = GWC2016_CSP3['SWC_PI_F_2_1_1']/1.42
SWC16_PI_F_2_2_1_CSP3 = GWC2016_CSP3['SWC_PI_F_2_2_1']/1.42
SWC16_PI_F_2_3_1_CSP3 = GWC2016_CSP3['SWC_PI_F_2_3_1']/1.42
SWC16_PI_F_2_4_1_CSP3 = GWC2016_CSP3['SWC_PI_F_2_4_1']/1.42
SWC16_PI_F_3_1_1_CSP3 = GWC2016_CSP3['SWC_PI_F_3_1_1']/1.42
SWC16_PI_F_3_2_1_CSP3 = GWC2016_CSP3['SWC_PI_F_3_2_1']/1.42
SWC16_PI_F_3_3_1_CSP3 = GWC2016_CSP3['SWC_PI_F_3_3_1']/1.42
SWC16_PI_F_3_4_1_CSP3 = GWC2016_CSP3['SWC_PI_F_3_4_1']/1.42

SWC17_PI_F_1_1_1_CSP3 = GWC2017_CSP3['SWC_PI_F_1_1_1']/1.42
SWC17_PI_F_1_2_1_CSP3 = GWC2017_CSP3['SWC_PI_F_1_2_1']/1.42
SWC17_PI_F_1_3_1_CSP3 = GWC2017_CSP3['SWC_PI_F_1_3_1']/1.42
SWC17_PI_F_1_4_1_CSP3 = GWC2017_CSP3['SWC_PI_F_1_4_1']/1.42
SWC17_PI_F_2_1_1_CSP3 = GWC2017_CSP3['SWC_PI_F_2_1_1']/1.42
SWC17_PI_F_2_2_1_CSP3 = GWC2017_CSP3['SWC_PI_F_2_2_1']/1.42
SWC17_PI_F_2_3_1_CSP3 = GWC2017_CSP3['SWC_PI_F_2_3_1']/1.42
SWC17_PI_F_2_4_1_CSP3 = GWC2017_CSP3['SWC_PI_F_2_4_1']/1.42
SWC17_PI_F_3_1_1_CSP3 = GWC2017_CSP3['SWC_PI_F_3_1_1']/1.42
SWC17_PI_F_3_2_1_CSP3 = GWC2017_CSP3['SWC_PI_F_3_2_1']/1.42
SWC17_PI_F_3_3_1_CSP3 = GWC2017_CSP3['SWC_PI_F_3_3_1']/1.42
SWC17_PI_F_3_4_1_CSP3 = GWC2017_CSP3['SWC_PI_F_3_4_1']/1.42

SWC18_PI_F_1_1_1_CSP3 = GWC2018_CSP3['SWC_PI_F_1_1_1']/1.42
SWC18_PI_F_1_2_1_CSP3 = GWC2018_CSP3['SWC_PI_F_1_2_1']/1.42
SWC18_PI_F_1_3_1_CSP3 = GWC2018_CSP3['SWC_PI_F_1_3_1']/1.42
SWC18_PI_F_1_4_1_CSP3 = GWC2018_CSP3['SWC_PI_F_1_4_1']/1.42
SWC18_PI_F_2_1_1_CSP3 = GWC2018_CSP3['SWC_PI_F_2_1_1']/1.42
SWC18_PI_F_2_2_1_CSP3 = GWC2018_CSP3['SWC_PI_F_2_2_1']/1.42
SWC18_PI_F_2_3_1_CSP3 = GWC2018_CSP3['SWC_PI_F_2_3_1']/1.42
SWC18_PI_F_2_4_1_CSP3 = GWC2018_CSP3['SWC_PI_F_2_4_1']/1.42
SWC18_PI_F_3_1_1_CSP3 = GWC2018_CSP3['SWC_PI_F_3_1_1']/1.42
SWC18_PI_F_3_2_1_CSP3 = GWC2018_CSP3['SWC_PI_F_3_2_1']/1.42
SWC18_PI_F_3_3_1_CSP3 = GWC2018_CSP3['SWC_PI_F_3_3_1']/1.42
SWC18_PI_F_3_4_1_CSP3 = GWC2018_CSP3['SWC_PI_F_3_4_1']/1.42

SWC19_PI_F_1_1_1_CSP3 = GWC2019_CSP3['SWC_PI_F_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP3 = GWC2019_CSP3['SWC_PI_F_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP3 = GWC2019_CSP3['SWC_PI_F_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP3 = GWC2019_CSP3['SWC_PI_F_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP3 = GWC2019_CSP3['SWC_PI_F_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP3 = GWC2019_CSP3['SWC_PI_F_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP3 = GWC2019_CSP3['SWC_PI_F_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP3 = GWC2019_CSP3['SWC_PI_F_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP3 = GWC2019_CSP3['SWC_PI_F_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP3 = GWC2019_CSP3['SWC_PI_F_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP3 = GWC2019_CSP3['SWC_PI_F_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP3 = GWC2019_CSP3['SWC_PI_F_3_4_1']/1.42

SWC20_PI_F_1_1_1_CSP3 = GWC2020_CSP3['SWC_PI_F_1_1_1']/1.42
SWC20_PI_F_1_2_1_CSP3 = GWC2020_CSP3['SWC_PI_F_1_2_1']/1.42
SWC20_PI_F_1_3_1_CSP3 = GWC2020_CSP3['SWC_PI_F_1_3_1']/1.42
SWC20_PI_F_1_4_1_CSP3 = GWC2020_CSP3['SWC_PI_F_1_4_1']/1.42
SWC20_PI_F_2_1_1_CSP3 = GWC2020_CSP3['SWC_PI_F_2_1_1']/1.42
SWC20_PI_F_2_2_1_CSP3 = GWC2020_CSP3['SWC_PI_F_2_2_1']/1.42
SWC20_PI_F_2_3_1_CSP3 = GWC2020_CSP3['SWC_PI_F_2_3_1']/1.42
SWC20_PI_F_2_4_1_CSP3 = GWC2020_CSP3['SWC_PI_F_2_4_1']/1.42
SWC20_PI_F_3_1_1_CSP3 = GWC2020_CSP3['SWC_PI_F_3_1_1']/1.42
SWC20_PI_F_3_2_1_CSP3 = GWC2020_CSP3['SWC_PI_F_3_2_1']/1.42
SWC20_PI_F_3_3_1_CSP3 = GWC2020_CSP3['SWC_PI_F_3_3_1']/1.42
SWC20_PI_F_3_4_1_CSP3 = GWC2020_CSP3['SWC_PI_F_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP3 = GWC2021_CSP3['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP3 = GWC2021_CSP3['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP3 = GWC2021_CSP3['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP3 = GWC2021_CSP3['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP3 = GWC2021_CSP3['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP3 = GWC2021_CSP3['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP3 = GWC2021_CSP3['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP3 = GWC2021_CSP3['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP3 = GWC2021_CSP3['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP3 = GWC2021_CSP3['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP3 = GWC2021_CSP3['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP3 = GWC2021_CSP3['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP3 = GWC2022_CSP3['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP3 = GWC2022_CSP3['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP3 = GWC2022_CSP3['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP3 = GWC2022_CSP3['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP3 = GWC2022_CSP3['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP3 = GWC2022_CSP3['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP3 = GWC2022_CSP3['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP3 = GWC2022_CSP3['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP3 = GWC2022_CSP3['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP3 = GWC2022_CSP3['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP3 = GWC2022_CSP3['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP3 = GWC2022_CSP3['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP3 = GWC2023_CSP3['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP3 = GWC2023_CSP3['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP3 = GWC2023_CSP3['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP3 = GWC2023_CSP3['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP3 = GWC2023_CSP3['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP3 = GWC2023_CSP3['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP3 = GWC2023_CSP3['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP3 = GWC2023_CSP3['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP3 = GWC2023_CSP3['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP3 = GWC2023_CSP3['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP3 = GWC2023_CSP3['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP3 = GWC2023_CSP3['SWC_F_3_4_1']/1.42

twosenwavgall2011_CSP3 = ((((SWC11_PI_F_1_1_1_CSP3 + SWC11_PI_F_2_1_1_CSP3 + SWC11_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC11_PI_F_1_2_1_CSP3 + SWC11_PI_F_2_2_1_CSP3 + SWC11_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2012_CSP3 = ((((SWC12_PI_F_1_1_1_CSP3 + SWC12_PI_F_2_1_1_CSP3 + SWC12_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC12_PI_F_1_2_1_CSP3 + SWC12_PI_F_2_2_1_CSP3 + SWC12_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2013_CSP3 = ((((SWC13_PI_F_1_1_1_CSP3 + SWC13_PI_F_2_1_1_CSP3 + SWC13_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC13_PI_F_1_2_1_CSP3 + SWC13_PI_F_2_2_1_CSP3 + SWC13_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2014_CSP3 = ((((SWC14_PI_F_1_1_1_CSP3 + SWC14_PI_F_2_1_1_CSP3 + SWC14_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC14_PI_F_1_2_1_CSP3 + SWC14_PI_F_2_2_1_CSP3 + SWC14_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2015_CSP3 = ((((SWC15_PI_F_1_1_1_CSP3 + SWC15_PI_F_2_1_1_CSP3 + SWC15_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC15_PI_F_1_2_1_CSP3 + SWC15_PI_F_2_2_1_CSP3 + SWC15_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2016_CSP3 = ((((SWC16_PI_F_1_1_1_CSP3 + SWC16_PI_F_2_1_1_CSP3 + SWC16_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC16_PI_F_1_2_1_CSP3 + SWC16_PI_F_2_2_1_CSP3 + SWC16_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2017_CSP3 = ((((SWC17_PI_F_1_1_1_CSP3 + SWC17_PI_F_2_1_1_CSP3 + SWC17_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC17_PI_F_1_2_1_CSP3 + SWC17_PI_F_2_2_1_CSP3 + SWC17_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2018_CSP3 = ((((SWC18_PI_F_1_1_1_CSP3 + SWC18_PI_F_2_1_1_CSP3 + SWC18_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC18_PI_F_1_2_1_CSP3 + SWC18_PI_F_2_2_1_CSP3 + SWC18_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2019_CSP3 = ((((SWC19_PI_F_1_1_1_CSP3 + SWC19_PI_F_2_1_1_CSP3 + SWC19_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP3 + SWC19_PI_F_2_2_1_CSP3 + SWC19_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2020_CSP3 = ((((SWC20_PI_F_1_1_1_CSP3 + SWC20_PI_F_2_1_1_CSP3 + SWC20_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC20_PI_F_1_2_1_CSP3 + SWC20_PI_F_2_2_1_CSP3 + SWC20_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2021_CSP3 = ((((SWC21_PI_F_1_1_1_CSP3 + SWC21_PI_F_2_1_1_CSP3 + SWC21_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP3 + SWC21_PI_F_2_2_1_CSP3 + SWC21_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2022_CSP3 = ((((SWC22_PI_F_1_1_1_CSP3 + SWC22_PI_F_2_1_1_CSP3 + SWC22_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP3 + SWC22_PI_F_2_2_1_CSP3 + SWC22_PI_F_3_2_1_CSP3)/3)*0.25)) / 100
twosenwavgall2023_CSP3 = ((((SWC23_PI_F_1_1_1_CSP3 + SWC23_PI_F_2_1_1_CSP3 + SWC23_PI_F_3_1_1_CSP3)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP3 + SWC23_PI_F_2_2_1_CSP3 + SWC23_PI_F_3_2_1_CSP3)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2011_CSP3 = twosenwavgall2011_CSP3.resample('D').mean() 
dailywavgSWC2012_CSP3 = twosenwavgall2012_CSP3.resample('D').mean() 
dailywavgSWC2013_CSP3 = twosenwavgall2013_CSP3.resample('D').mean() 
dailywavgSWC2014_CSP3 = twosenwavgall2014_CSP3.resample('D').mean()
dailywavgSWC2015_CSP3 = twosenwavgall2015_CSP3.resample('D').mean() 
dailywavgSWC2016_CSP3 = twosenwavgall2016_CSP3.resample('D').mean() 
dailywavgSWC2017_CSP3 = twosenwavgall2017_CSP3.resample('D').mean() 
dailywavgSWC2018_CSP3 = twosenwavgall2018_CSP3.resample('D').mean()
dailywavgSWC2019_CSP3 = twosenwavgall2019_CSP3.resample('D').mean() 
dailywavgSWC2020_CSP3 = twosenwavgall2020_CSP3.resample('D').mean() 
dailywavgSWC2021_CSP3 = twosenwavgall2021_CSP3.resample('D').mean() 
dailywavgSWC2022_CSP3 = twosenwavgall2022_CSP3.resample('D').mean() 
dailywavgSWC2023_CSP3 = twosenwavgall2023_CSP3.resample('D').mean()

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2011_CSP3 = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2012_CSP3 = pd.read_csv('YOUR PATH/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2013_CSP3 = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2014_CSP3 = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2015_CSP3 = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2016_CSP3 = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2017_CSP3 = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2018_CSP3 = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2019_CSP3 = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2020_CSP3 = pd.read_csv('YOUR PATH/CSP32020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP3 = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP3 = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP3 = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2011_CSP3 = NPIV2011_CSP3['NPIV']
NPIVCount2012_CSP3 = NPIV2012_CSP3['NPIV']
NPIVCount2013_CSP3 = NPIV2013_CSP3['NPIV']
NPIVCount2014_CSP3 = NPIV2014_CSP3['NPIV']
NPIVCount2015_CSP3 = NPIV2015_CSP3['NPIV']
NPIVCount2016_CSP3 = NPIV2016_CSP3['NPIV']
NPIVCount2017_CSP3 = NPIV2017_CSP3['NPIV']
NPIVCount2018_CSP3 = NPIV2018_CSP3['NPIV']
NPIVCount2019_CSP3 = NPIV2019_CSP3['NPIV']
NPIVCount2020_CSP3 = NPIV2020_CSP3['NPIV']
NPIVCount2021_CSP3 = NPIV2021_CSP3['NPIV']
NPIVCount2022_CSP3 = NPIV2022_CSP3['NPIV']
NPIVCount2023_CSP3 = NPIV2023_CSP3['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2011_CSP3 = NPIVCount2011_CSP3.resample('D').mean()
dailyNPIV2012_CSP3 = NPIVCount2012_CSP3.resample('D').mean()
dailyNPIV2013_CSP3 = NPIVCount2013_CSP3.resample('D').mean()
dailyNPIV2014_CSP3 = NPIVCount2014_CSP3.resample('D').mean()
dailyNPIV2015_CSP3 = NPIVCount2015_CSP3.resample('D').mean()
dailyNPIV2016_CSP3 = NPIVCount2016_CSP3.resample('D').mean()
dailyNPIV2017_CSP3 = NPIVCount2017_CSP3.resample('D').mean()
dailyNPIV2018_CSP3 = NPIVCount2018_CSP3.resample('D').mean()
dailyNPIV2019_CSP3 = NPIVCount2019_CSP3.resample('D').mean()
dailyNPIV2020_CSP3 = NPIVCount2020_CSP3.resample('D').mean()
dailyNPIV2021_CSP3 = NPIVCount2021_CSP3.resample('D').mean()
dailyNPIV2022_CSP3 = NPIVCount2022_CSP3.resample('D').mean()
dailyNPIV2023_CSP3 = NPIVCount2023_CSP3.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc11_CSP3 = dailywavgSWC2011_CSP3
pwc12_CSP3 = dailywavgSWC2012_CSP3
pwc13_CSP3 = dailywavgSWC2013_CSP3
pwc14_CSP3 = dailywavgSWC2014_CSP3
pwc15_CSP3 = dailywavgSWC2015_CSP3
pwc16_CSP3 = dailywavgSWC2016_CSP3
pwc17_CSP3 = dailywavgSWC2017_CSP3
pwc18_CSP3 = dailywavgSWC2018_CSP3
pwc19_CSP3 = dailywavgSWC2019_CSP3
pwc20_CSP3 = dailywavgSWC2020_CSP3
pwc21_CSP3 = dailywavgSWC2021_CSP3
pwc22_CSP3 = dailywavgSWC2022_CSP3
pwc23_CSP3 = dailywavgSWC2023_CSP3

N11_CSP3 = dailyNPIV2011_CSP3
N12_CSP3 = dailyNPIV2012_CSP3
N13_CSP3 = dailyNPIV2013_CSP3
N14_CSP3 = dailyNPIV2014_CSP3
N15_CSP3 = dailyNPIV2015_CSP3
N16_CSP3 = dailyNPIV2016_CSP3
N17_CSP3 = dailyNPIV2017_CSP3
N18_CSP3 = dailyNPIV2018_CSP3
N19_CSP3 = dailyNPIV2019_CSP3
N20_CSP3 = dailyNPIV2020_CSP3
N21_CSP3 = dailyNPIV2021_CSP3
N22_CSP3 = dailyNPIV2022_CSP3
N23_CSP3 = dailyNPIV2023_CSP3

N0_for_calc_3 = 1633.36
lwc = 0.058
soc = 0.0092

#Calculate N
N0_CSP3_11 = N0_for_calc_3*(((0.0808)/((pwc11_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_12 = N0_for_calc_3*(((0.0808)/((pwc12_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_13 = N0_for_calc_3*(((0.0808)/((pwc13_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_14 = N0_for_calc_3*(((0.0808)/((pwc14_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_15 = N0_for_calc_3*(((0.0808)/((pwc15_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_16 = N0_for_calc_3*(((0.0808)/((pwc16_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_17 = N0_for_calc_3*(((0.0808)/((pwc17_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_18 = N0_for_calc_3*(((0.0808)/((pwc18_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_19 = N0_for_calc_3*(((0.0808)/((pwc19_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_20 = N0_for_calc_3*(((0.0808)/((pwc20_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_21 = N0_for_calc_3*(((0.0808)/((pwc21_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_22 = N0_for_calc_3*(((0.0808)/((pwc22_CSP3+lwc+soc)+0.115))+0.372)
N0_CSP3_23 = N0_for_calc_3*(((0.0808)/((pwc23_CSP3+lwc+soc)+0.115))+0.372)

N0interp11_CSP3 = N0_CSP3_11.interpolate(method = 'linear', limit_direction = 'forward')
N0interp12_CSP3 = N0_CSP3_12.interpolate(method = 'linear', limit_direction = 'forward')
N0interp13_CSP3 = N0_CSP3_13.interpolate(method = 'linear', limit_direction = 'forward')
N0interp14_CSP3 = N0_CSP3_14.interpolate(method = 'linear', limit_direction = 'forward')
N0interp15_CSP3 = N0_CSP3_15.interpolate(method = 'linear', limit_direction = 'forward')
N0interp16_CSP3 = N0_CSP3_16.interpolate(method = 'linear', limit_direction = 'forward')
N0interp17_CSP3 = N0_CSP3_17.interpolate(method = 'linear', limit_direction = 'forward')
N0interp18_CSP3 = N0_CSP3_18.interpolate(method = 'linear', limit_direction = 'forward')
N0interp19_CSP3 = N0_CSP3_19.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP3 = N0_CSP3_20.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP3 = N0_CSP3_21.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP3 = N0_CSP3_22.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP3 = N0_CSP3_23.interpolate(method = 'linear', limit_direction = 'forward')

NCSP3_11= N0interp11_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_11.csv')
NCSP3_12= N0interp12_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_12.csv')
NCSP3_13= N0interp13_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_13.csv')
NCSP3_14= N0interp14_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_14.csv')
NCSP3_15= N0interp15_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_15.csv')
NCSP3_16= N0interp16_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_16.csv')
NCSP3_17= N0interp17_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_17.csv')
NCSP3_18= N0interp18_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_18.csv')
NCSP3_19= N0interp19_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_19.csv')
NCSP3_20= N0interp20_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_20.csv')
NCSP3_21= N0interp21_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_21.csv')
NCSP3_22= N0interp22_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_22.csv')
NCSP3_23= N0interp23_CSP3.to_csv('YOUR PATH TO MAKE/N_CSP3_23.csv')

CSP3_11 = pd.read_csv('YOUR PATH/N_CSP3_11.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_12 = pd.read_csv('YOUR PATH/N_CSP3_12.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_13 = pd.read_csv('YOUR PATH/N_CSP3_13.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_14 = pd.read_csv('YOUR PATH/N_CSP3_14.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_15 = pd.read_csv('YOUR PATH/N_CSP3_15.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_16 = pd.read_csv('YOUR PATH/N_CSP3_16.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_17 = pd.read_csv('YOUR PATH/N_CSP3_17.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_18 = pd.read_csv('YOUR PATH/N_CSP3_18.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_19 = pd.read_csv('YOUR PATH/N_CSP3_19.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_20 = pd.read_csv('YOUR PATH/N_CSP3_20.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_21 = pd.read_csv('YOUR PATH/N_CSP3_21.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_22 = pd.read_csv('YOUR PATH/N_CSP3_22.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_23 = pd.read_csv('YOUR PATH/N_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP3_11_1 = CSP3_11['0']
N_CSP3_12_1 = CSP3_12['0']
N_CSP3_13_1 = CSP3_13['0']
N_CSP3_14_1 = CSP3_14['0']
N_CSP3_15_1 = CSP3_15['0']
N_CSP3_16_1 = CSP3_16['0']
N_CSP3_17_1 = CSP3_17['0']
N_CSP3_18_1 = CSP3_18['0']
N_CSP3_19_1 = CSP3_19['0']
N_CSP3_20_1 = CSP3_20['0']
N_CSP3_21_1 = CSP3_21['0']
N_CSP3_22_1 = CSP3_22['0']
N_CSP3_23_1 = CSP3_23['0']

fveg_CSP3_11 = dailyNPIV2011_CSP3 / N0_CSP3_11
fveg_CSP3_12 = dailyNPIV2012_CSP3 / N0_CSP3_12
fveg_CSP3_13 = dailyNPIV2013_CSP3 / N0_CSP3_13
fveg_CSP3_14 = dailyNPIV2014_CSP3 / N0_CSP3_14
fveg_CSP3_15_1 = dailyNPIV2015_CSP3 / N0_CSP3_15
fveg_CSP3_16 = dailyNPIV2016_CSP3 / N0_CSP3_16
fveg_CSP3_17 = dailyNPIV2017_CSP3 / N0_CSP3_17
fveg_CSP3_18 = dailyNPIV2018_CSP3 / N0_CSP3_18
fveg_CSP3_19 = dailyNPIV2019_CSP3 / N0_CSP3_19
fveg_CSP3_20_1 = dailyNPIV2020_CSP3 / N0_CSP3_20
fveg_CSP3_21 = dailyNPIV2021_CSP3 / N0_CSP3_21
fveg_CSP3_22_1 = dailyNPIV2022_CSP3 / N0_CSP3_22
fveg_CSP3_23_1 = dailyNPIV2023_CSP3 / N0_CSP3_23

fveg_CSP3_11.to_csv('YOUR PATH TO MAKE/fveg_CSP3_11.csv')
fveg_CSP3_12.to_csv('YOUR PATH TO MAKE/fveg_CSP3_12.csv')
fveg_CSP3_13.to_csv('YOUR PATH TO MAKE/fveg_CSP3_13.csv')
fveg_CSP3_14.to_csv('YOUR PATH TO MAKE/fveg_CSP3_14.csv')
fveg_CSP3_15_1.to_csv('YOUR PATH TO MAKE/fveg_CSP3_15_1.csv')
fveg_CSP3_16.to_csv('YOUR PATH TO MAKE/fveg_CSP3_16.csv')
fveg_CSP3_17.to_csv('YOUR PATH TO MAKE/fveg_CSP3_17.csv')
fveg_CSP3_18.to_csv('YOUR PATH TO MAKE/fveg_CSP3_18.csv')
fveg_CSP3_19.to_csv('YOUR PATH TO MAKE/fveg_CSP3_19.csv')
fveg_CSP3_20_1.to_csv('YOUR PATH TO MAKE/fveg_CSP3_20_1.csv')
fveg_CSP3_21.to_csv('YOUR PATH TO MAKE/fveg_CSP3_21.csv')
fveg_CSP3_22_1.to_csv('YOUR PATH TO MAKE/fveg_CSP3_22_1.csv')
fveg_CSP3_23_1.to_csv('YOUR PATH TO MAKE/fveg_CSP3_23_1.csv')

fveg_CSP3_11_1 = pd.read_csv('YOUR PATH/fveg_CSP3_11.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_12_1 = pd.read_csv('YOUR PATH/fveg_CSP3_12.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_13_1 = pd.read_csv('YOUR PATH/fveg_CSP3_13.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_14_1 = pd.read_csv('YOUR PATH/fveg_CSP3_14.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_15_1 = pd.read_csv('YOUR PATH/fveg_CSP3_15.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_16_1 = pd.read_csv('YOUR PATH/fveg_CSP3_16.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_17_1 = pd.read_csv('YOUR PATH/fveg_CSP3_17.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_18_1 = pd.read_csv('YOUR PATH/fveg_CSP3_18.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_19_1 = pd.read_csv('YOUR PATH/fveg_CSP3_19.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_20_1 = pd.read_csv('YOUR PATH/fveg_CSP3_20.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_21_1 = pd.read_csv('YOUR PATH/fveg_CSP3_21.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_22_1 = pd.read_csv('YOUR PATH/fveg_CSP3_22.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_23_1 = pd.read_csv('YOUR PATH/fveg_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP3_11_1 = fveg_CSP3_11_1['0']
fveg_CSP3_12_1 = fveg_CSP3_12_1['0']
fveg_CSP3_13_1 = fveg_CSP3_13_1['0']
fveg_CSP3_14_1 = fveg_CSP3_14_1['0']
fveg_CSP3_15_1 = fveg_CSP3_15_1['0']
fveg_CSP3_16_1 = fveg_CSP3_16_1['0']
fveg_CSP3_17_1 = fveg_CSP3_17_1['0']
fveg_CSP3_18_1 = fveg_CSP3_18_1['0']
fveg_CSP3_19_1 = fveg_CSP3_19_1['0']
fveg_CSP3_20_1 = fveg_CSP3_20_1['0']
fveg_CSP3_21_1 = fveg_CSP3_21_1['0']
fveg_CSP3_22_1 = fveg_CSP3_22_1['0']
fveg_CSP3_23_1 = fveg_CSP3_23_1['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2011_CSP3 = pd.read_csv('YOUR PATH/CSP32011BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2012_CSP3 = pd.read_csv('YOUR PATH/CSP32012BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2013_CSP3 = pd.read_csv('YOUR PATH/CSP32013BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2014_CSP3 = pd.read_csv('YOUR PATH/CSP32014BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2015_CSP3 = pd.read_csv('YOUR PATH/CSP32015BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2016_CSP3 = pd.read_csv('YOUR PATH/CSP32016BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2017_CSP3 = pd.read_csv('YOUR PATH/CSP32017BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2018_CSP3 = pd.read_csv('YOUR PATH/CSP32018BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2019_CSP3 = pd.read_csv('YOUR PATH/CSP32019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP3 = pd.read_csv('YOUR PATH/CSP32020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP3 = pd.read_csv('YOUR PATH/CSP32021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP3 = pd.read_csv('YOUR PATH/CSP32022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP3 = pd.read_csv('YOUR PATH/CSP32023BWE_plotdate.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE11_CSP3 = BWEcsv2011_CSP3['BWE']
BWE12_CSP3 = BWEcsv2012_CSP3['BWE']
BWE13_CSP3 = BWEcsv2013_CSP3['BWE']
BWE14_CSP3 = BWEcsv2014_CSP3['BWE']
BWE15_CSP3 = BWEcsv2015_CSP3['BWE']
BWE16_CSP3 = BWEcsv2016_CSP3['BWE']
BWE17_CSP3 = BWEcsv2017_CSP3['BWE']
BWE18_CSP3 = BWEcsv2018_CSP3['BWE']
BWE19_CSP3 = BWEcsv2019_CSP3['BWE']
BWE20_CSP3 = BWEcsv2020_CSP3['BWE']
BWE21_CSP3 = BWEcsv2021_CSP3['BWE']
BWE22_CSP3 = BWEcsv2022_CSP3['BWE']
BWE23_CSP3 = BWEcsv2023_CSP3['BWE']

#Interpolate between dates
BWEinterp11_CSP3 = BWE11_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp12_CSP3 = BWE12_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp13_CSP3 = BWE13_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp14_CSP3 = BWE14_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp15_CSP3 = BWE15_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp16_CSP3 = BWE16_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp17_CSP3 = BWE17_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp18_CSP3 = BWE18_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp19_CSP3 = BWE19_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP3 = BWE20_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP3 = BWE21_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP3 = BWE22_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP3 = BWE23_CSP3.interpolate(method = 'linear', limit_direction = 'forward')

BWEFillNaN11_CSP3 = BWEinterp11_CSP3.fillna(int(BWEinterp11_CSP3.mean()), inplace=True)
BWEFillNaN12_CSP3 = BWEinterp12_CSP3.fillna(int(BWEinterp12_CSP3.mean()), inplace=True)
BWEFillNaN13_CSP3 = BWEinterp13_CSP3.fillna(int(BWEinterp13_CSP3.mean()), inplace=True)
BWEFillNaN14_CSP3 = BWEinterp14_CSP3.fillna(int(BWEinterp14_CSP3.mean()), inplace=True)
BWEFillNaN15_CSP3 = BWEinterp15_CSP3.fillna(int(BWEinterp15_CSP3.mean()), inplace=True)
BWEFillNaN16_CSP3 = BWEinterp16_CSP3.fillna(int(BWEinterp16_CSP3.mean()), inplace=True)
BWEFillNaN17_CSP3 = BWEinterp17_CSP3.fillna(int(BWEinterp17_CSP3.mean()), inplace=True)
BWEFillNaN18_CSP3 = BWEinterp18_CSP3.fillna(int(BWEinterp18_CSP3.mean()), inplace=True)
BWEFillNaN19_CSP3 = BWEinterp19_CSP3.fillna(int(BWEinterp19_CSP3.mean()), inplace=True)
BWEFillNaN20_CSP3 = BWEinterp20_CSP3.fillna(int(BWEinterp20_CSP3.mean()), inplace=True)
BWEFillNaN21_CSP3 = BWEinterp21_CSP3.fillna(int(BWEinterp21_CSP3.mean()), inplace=True)
BWEFillNaN22_CSP3 = BWEinterp22_CSP3.fillna(int(BWEinterp22_CSP3.mean()), inplace=True)
BWEFillNaN23_CSP3 = BWEinterp23_CSP3.fillna(int(BWEinterp23_CSP3.mean()), inplace=True)

fveg_CSP3_11_1 = fveg_CSP3_11_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_12_1 = fveg_CSP3_12_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_13_1 = fveg_CSP3_13_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_14_1 = fveg_CSP3_14_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_15_1 = fveg_CSP3_15_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_16_1 = fveg_CSP3_16_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_17_1 = fveg_CSP3_17_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_18_1 = fveg_CSP3_18_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_19_1 = fveg_CSP3_19_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_20_1 = fveg_CSP3_20_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_21_1 = fveg_CSP3_21_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_22_1 = fveg_CSP3_22_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_23_1 = fveg_CSP3_23_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x11_CSP3 = BWEinterp11_CSP3
y11_CSP3 = 1/fveg_CSP3_11_1
x12_CSP3 = BWEinterp12_CSP3
y12_CSP3 = 1/fveg_CSP3_12_1
x13_CSP3 = BWEinterp13_CSP3
y13_CSP3 = 1/fveg_CSP3_13_1
x14_CSP3 = BWEinterp14_CSP3
y14_CSP3 = 1/fveg_CSP3_14_1
x15_CSP3 = BWEinterp15_CSP3
y15_CSP3 = 1/fveg_CSP3_15_1
x16_CSP3 = BWEinterp16_CSP3
y16_CSP3 = 1/fveg_CSP3_16_1
x17_CSP3 = BWEinterp17_CSP3
y17_CSP3 = 1/fveg_CSP3_17_1
x18_CSP3 = BWEinterp18_CSP3
y18_CSP3 = 1/fveg_CSP3_18_1
x19_CSP3 = BWEinterp19_CSP3
y19_CSP3 = 1/fveg_CSP3_19_1
x20_CSP3 = BWEinterp20_CSP3
y20_CSP3 = 1/fveg_CSP3_20_1
x21_CSP3 = BWEinterp21_CSP3
y21_CSP3 = 1/fveg_CSP3_21_1
x22_CSP3 = BWEinterp22_CSP3
y22_CSP3 = 1/fveg_CSP3_22_1
x23_CSP3 = BWEinterp23_CSP3
y23_CSP3 = 1/fveg_CSP3_23_1

y15_CSP3 = fill_nan_with_avg(y15_CSP3)
y20_CSP3 = fill_nan_with_avg(y20_CSP3)
y22_CSP3 = fill_nan_with_avg(y22_CSP3)
y23_CSP3 = fill_nan_with_avg(y23_CSP3)

# Concatenate all x and y values
x_all_CSP3 = np.concatenate([x11_CSP3, x12_CSP3, x13_CSP3, x14_CSP3, x15_CSP3, x16_CSP3, x17_CSP3, x18_CSP3, x19_CSP3, x20_CSP3, x21_CSP3, x22_CSP3, x23_CSP3])
y_all_CSP3 = np.concatenate([y11_CSP3, y12_CSP3, y13_CSP3, y14_CSP3, y15_CSP3, y16_CSP3, y17_CSP3, y18_CSP3, y19_CSP3, y20_CSP3, y21_CSP3, y22_CSP3, y23_CSP3])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP3 = x_all_CSP3.reshape(-1, 1)

# Perform linear regression
slope_CSP3, intercept_CSP3, r_value_CSP3, p_value_CSP3, std_err_CSP3 = stats.linregress(x_all_CSP3, y_all_CSP3)

# Calculate the t-value for a 95% confidence interval
t_CSP3 = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP3) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP3 = t_CSP3 * std_err_CSP3
intercept_conf_interval_CSP3 = t_CSP3 * std_err_CSP3 * np.sqrt((1/len(x_all_CSP3)) + (np.mean(x_all_CSP3)**2 / np.sum((x_all_CSP3 - np.mean(x_all_CSP3))**2)))

# Calculate R-squared value
r_squared_CSP3 = r_value_CSP3**2
shift_CSP3 = 1 - intercept_CSP3

plt.scatter(x11_CSP3, y11_CSP3 + shift_CSP3, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.3)
plt.scatter(x12_CSP3, y12_CSP3 + shift_CSP3, color = 'indianred', marker = 'v', label = '2012', alpha=0.3)
plt.scatter(x13_CSP3, y13_CSP3 + shift_CSP3, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.3)
plt.scatter(x14_CSP3, y14_CSP3 + shift_CSP3, color = 'peru', marker = 's', label = '2014', alpha=0.3)
plt.scatter(x15_CSP3, y15_CSP3 + shift_CSP3, color = 'gold', marker = '<', label = '2015', alpha=0.3)
plt.scatter(x16_CSP3, y16_CSP3 + shift_CSP3, color = 'pink', marker = '>', label = '2016', alpha=0.3)
plt.scatter(x17_CSP3, y17_CSP3 + shift_CSP3, color = 'rosybrown', marker = '.', label = '2017', alpha=0.3)
plt.scatter(x18_CSP3, y18_CSP3 + shift_CSP3, color = 'fuchsia', marker = '1', label = '2018', alpha=0.3)
plt.scatter(x19_CSP3, y19_CSP3 + shift_CSP3, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP3, y20_CSP3 + shift_CSP3, color = 'aquamarine', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP3, y21_CSP3 + shift_CSP3, color = 'greenyellow', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP3, y22_CSP3 + shift_CSP3, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP3, y23_CSP3 + shift_CSP3, color = 'darkorange', marker = 'p', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP3, b_CSP3 = np.polyfit(x_CSP3,y_CSP3,1)
c_CSP3, d_CSP3 = np.polyfit(x_CSP3,y_baatz_CSP3,1)
plt.plot(x_CSP3, a_CSP3*x_CSP3+b_CSP3, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP3, 0.005*x_CSP3 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
plt.scatter(x_CSP3, y_CSP3, marker = 'o', color = 'red', label = r'$f_b$ Grav. 2023')
# Plot the line of best fit
plt.plot(x_all_CSP3, slope_CSP3*x_all_CSP3 + 1, color='black', label='Best fit', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Schrön Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3, round(slope_CSP3,2)], [r2_CSP3, round(r_squared_CSP3, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_fveg_with_BaatznTDR.png', dpi=300, bbox_inches='tight')
plt.show()


shift_CSP3 = 1 - intercept_CSP3

plt.scatter(x11_CSP3, y11_CSP3 + shift_CSP3, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.3)
plt.scatter(x12_CSP3, y12_CSP3 + shift_CSP3, color = 'indianred', marker = 'v', label = '2012', alpha=0.3)
plt.scatter(x13_CSP3, y13_CSP3 + shift_CSP3, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.3)
plt.scatter(x14_CSP3, y14_CSP3 + shift_CSP3, color = 'peru', marker = 's', label = '2014', alpha=0.3)
plt.scatter(x15_CSP3, y15_CSP3 + shift_CSP3, color = 'gold', marker = '<', label = '2015', alpha=0.3)
plt.scatter(x16_CSP3, y16_CSP3 + shift_CSP3, color = 'pink', marker = '>', label = '2016', alpha=0.3)
plt.scatter(x17_CSP3, y17_CSP3 + shift_CSP3, color = 'rosybrown', marker = '.', label = '2017', alpha=0.3)
plt.scatter(x18_CSP3, y18_CSP3 + shift_CSP3, color = 'fuchsia', marker = '1', label = '2018', alpha=0.3)
plt.scatter(x19_CSP3, y19_CSP3 + shift_CSP3, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP3, y20_CSP3 + shift_CSP3, color = 'aquamarine', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP3, y21_CSP3 + shift_CSP3, color = 'greenyellow', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP3, y22_CSP3 + shift_CSP3, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP3, y23_CSP3 + shift_CSP3, color = 'darkorange', marker = 'p', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP3, b_CSP3 = np.polyfit(x_CSP3,y_CSP3,1)
c_CSP3, d_CSP3 = np.polyfit(x_CSP3,y_baatz_CSP3,1)
plt.scatter(x_CSP3, y_CSP3, marker = 'o', color = 'red', label = r'$f_b$ Grav. 2023')
plt.plot(x_CSP3, a_CSP3*x_CSP3+b_CSP3, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP3, 0.005*x_CSP3 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP3, slope_CSP3*x_all_CSP3 + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Schrön Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3, round(slope_CSP3,2)], [r2_CSP3, round(r_squared_CSP3, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_with_TDR_no2014.png', dpi=300, bbox_inches='tight')
plt.show()


# Normalize according to Baatz 2015
normal_CSP1_all = (26.14/2893.9)*100
normal_CSP2_all = (32.47/2949.04)*100
normal_CSP3_all = (9.45/1621.86)*100

data = {
    '': ['Slope', 'Intercept', 'Baatz 2015\nSlope', 'Baatz 2015\nIntercept', 'Slope\nDifference', 'Slope\nRatio', 'Neutron\nIntensity\nReduction(%)'],
    'CSP1': [f"{r_mm_CSP1}", f"{r_nn_CSP1}", f"{r_mm_CSP1_baatz}", f"{r_nn_CSP1_baatz}", f"{round((r_mm_CSP1-r_mm_CSP1_baatz),4)}", f"{round((r_mm_CSP1/r_mm_CSP1_baatz),4)}", f"{round((normal_CSP1),2)}"],
    'CSP2': [f"{r_mm_CSP2}", f"{r_nn_CSP2}", f"{r_mm_CSP2_baatz}", f"{r_nn_CSP2_baatz}", f"{round((r_mm_CSP2-r_mm_CSP2_baatz),4)}", f"{round((r_mm_CSP2/r_mm_CSP2_baatz),4)}", f"{round((normal_CSP2),2)}"],
    'CSP3': [f"{r_mm_CSP3}", f"{r_nn_CSP3}", f"{r_mm_CSP3_baatz}", f"{r_nn_CSP3_baatz}", f"{round((r_mm_CSP3-r_mm_CSP3_baatz),4)}", f"{round((r_mm_CSP3/r_mm_CSP3_baatz),4)}", f"{round((normal_CSP3),2)}"],
    'CSP1\nTDR': [f"{round(val, 4)}" for val in [slope_CSP1, intercept_CSP1, r_mm_CSP1_baatz, r_nn_CSP1_baatz, slope_CSP1-r_mm_CSP1_baatz, slope_CSP1/r_mm_CSP1_baatz]] + [f"{round(normal_CSP1_all, 2)}"],
    'CSP2\nTDR': [f"{round(val, 4)}" for val in [slope_CSP2, intercept_CSP2, r_mm_CSP2_baatz, r_nn_CSP2_baatz, slope_CSP2-r_mm_CSP2_baatz, slope_CSP2/r_mm_CSP2_baatz]] + [f"{round(normal_CSP2_all, 2)}"],
    'CSP3\nTDR': [f"{round(val, 4)}" for val in [slope_CSP3, intercept_CSP3, r_mm_CSP3_baatz, r_nn_CSP3_baatz, slope_CSP3-r_mm_CSP3_baatz, slope_CSP3/r_mm_CSP3_baatz]] + [f"{round(normal_CSP3_all, 2)}"],
    'CRS-2000/B\nAverage': [f"{round(val, 4)}" for val in [(r_mm_CSP1+r_mm_CSP2+slope_CSP1+slope_CSP2)/4, (r_nn_CSP1+r_nn_CSP2+intercept_CSP1+intercept_CSP2)/4, (r_mm_CSP1_baatz+r_mm_CSP2_baatz)/2, (r_nn_CSP1_baatz+r_nn_CSP2_baatz)/2, ((r_mm_CSP1-r_mm_CSP1_baatz)+(r_mm_CSP2-r_mm_CSP2_baatz)+(slope_CSP1-r_mm_CSP1_baatz)+(slope_CSP2-r_mm_CSP2_baatz))/4, ((r_mm_CSP1/r_mm_CSP1_baatz)+(r_mm_CSP2/r_mm_CSP2_baatz)+(slope_CSP1/r_mm_CSP1_baatz)+(slope_CSP2/r_mm_CSP2_baatz))/4]] + [f"{round((normal_CSP1+normal_CSP2+normal_CSP1_all+normal_CSP2_all)/4, 2)}"],
    'CRS-1000/B\nAverage': [f"{round(val, 4)}" for val in [(slope_CSP3+r_mm_CSP3)/2, (intercept_CSP3+r_nn_CSP3)/2, r_mm_CSP3_baatz, r_nn_CSP3_baatz, ((slope_CSP3-r_mm_CSP3_baatz)+(r_mm_CSP3-r_mm_CSP3_baatz))/2, ((slope_CSP3/r_mm_CSP3_baatz)+(r_mm_CSP3/r_mm_CSP3_baatz))/2]] + [f"{round((normal_CSP3_all+normal_CSP3)/2, 2)}"],
    'Gravimetric\nAverage': [f"{round(val, 4)}" for val in [(r_mm_CSP1+r_mm_CSP2+r_mm_CSP3)/3, (r_nn_CSP1+r_nn_CSP2+r_nn_CSP3)/3, (r_mm_CSP1_baatz+r_mm_CSP2_baatz+r_mm_CSP3_baatz)/3, (r_nn_CSP1_baatz+r_nn_CSP2_baatz+r_nn_CSP3_baatz)/3, ((r_mm_CSP1-r_mm_CSP1_baatz)+(r_mm_CSP2-r_mm_CSP2_baatz)+(r_mm_CSP3-r_mm_CSP3_baatz))/3, ((r_mm_CSP1/r_mm_CSP1_baatz)+(r_mm_CSP2/r_mm_CSP2_baatz)+(r_mm_CSP3/r_mm_CSP3_baatz))/3]] + [f"{round((normal_CSP1+normal_CSP2+normal_CSP3)/3, 2)}"], 
    'TDR\nAverage': [f"{round(val, 4)}" for val in [(slope_CSP1+slope_CSP2+slope_CSP3)/3, (intercept_CSP1+intercept_CSP2+intercept_CSP3)/3, (r_mm_CSP1_baatz+r_mm_CSP2_baatz+r_mm_CSP3_baatz)/3, (r_nn_CSP1_baatz+r_nn_CSP2_baatz+r_nn_CSP3_baatz)/3, ((r_mm_CSP1-r_mm_CSP1_baatz)+(r_mm_CSP2-r_mm_CSP2_baatz)+(r_mm_CSP3-r_mm_CSP3_baatz))/3, ((slope_CSP1/r_mm_CSP1_baatz)+(slope_CSP2/r_mm_CSP2_baatz)+(slope_CSP3/r_mm_CSP3_baatz))/3]] + [f"{round((normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/3, 2)}"],
    'Overall\nAverage': [f"{round((val1 + val2) / 2, 4)}" for val1, val2 in zip(
    [(r_mm_CSP1+r_mm_CSP2+r_mm_CSP3)/3, (r_nn_CSP1+r_nn_CSP2+r_nn_CSP3)/3, (r_mm_CSP1_baatz+r_mm_CSP2_baatz+r_mm_CSP3_baatz)/3, (r_nn_CSP1_baatz+r_nn_CSP2_baatz+r_nn_CSP3_baatz)/3, ((r_mm_CSP1-r_mm_CSP1_baatz)+(r_mm_CSP2-r_mm_CSP2_baatz)+(r_mm_CSP3-r_mm_CSP3_baatz))/3, ((r_mm_CSP1/r_mm_CSP1_baatz)+(r_mm_CSP2/r_mm_CSP2_baatz)+(r_mm_CSP3/r_mm_CSP3_baatz))/3, (normal_CSP1+normal_CSP2+normal_CSP3)/3],
    [(slope_CSP1+slope_CSP2+slope_CSP3)/3, (intercept_CSP1+intercept_CSP2+intercept_CSP3)/3, (r_mm_CSP1_baatz+r_mm_CSP2_baatz+r_mm_CSP3_baatz)/3, (r_nn_CSP1_baatz+r_nn_CSP2_baatz+r_nn_CSP3_baatz)/3, ((r_mm_CSP1-r_mm_CSP1_baatz)+(r_mm_CSP2-r_mm_CSP2_baatz)+(r_mm_CSP3-r_mm_CSP3_baatz))/3, ((slope_CSP1/r_mm_CSP1_baatz)+(slope_CSP2/r_mm_CSP2_baatz)+(slope_CSP3/r_mm_CSP3_baatz))/3, (normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/3]
)]
}
# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Function to render DataFrame as a table
def render_mpl_table(data, col_width=6.0, row_height=0.7, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(22,8))
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

render_mpl_table(df, header_columns=0, col_width=6.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Summary_Table_with_TDR.png', dpi=300, bbox_inches='tight')
plt.show()

data_1 = {
    '': ['CSP1', 'CSP2', 'CSP3', 'CSP1\nTDR', 'CSP2\nTDR', 'CSP3\nTDR', 'CRS-2000/B\nAverage', 'CRS-1000/B\nAverage', 'Gravimetric\nAverage', 'TDR\nAverage', 'Overall\nAverage'],
    'Neutron Intensity\nReduction\n(% per mm BWE)': [f"{round((normal_CSP1),3)}", f"{round((normal_CSP2),3)}", f"{round((normal_CSP3),3)}", f"{round((normal_CSP1_all),3)}", f"{round(normal_CSP2_all, 3)}", f"{round(normal_CSP3_all, 3)}", f"{round((normal_CSP1+normal_CSP2+normal_CSP1_all+normal_CSP2_all)/4, 3)} +/- 0.12", f"{round((normal_CSP3_all+normal_CSP3)/2, 3)} +/- 0.30", f"{round((normal_CSP1+normal_CSP2+normal_CSP3)/3, 3)} +/- 0.45", f"{round((normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/3, 3)} +/- 0.24", f"{round((normal_CSP1+normal_CSP2+normal_CSP3+normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/6, 3)} +/- 0.30"],
    'Slope\nRatio': [f"{round((r_mm_CSP1/r_mm_CSP1_baatz),3)}", f"{round((r_mm_CSP2/r_mm_CSP2_baatz),3)}", f"{round((r_mm_CSP3/r_mm_CSP3_baatz),3)}", f"{round(slope_CSP1/r_mm_CSP1_baatz, 3)}", f"{round(slope_CSP2/r_mm_CSP2_baatz, 3)}", f"{round(slope_CSP3/r_mm_CSP3_baatz, 3)}", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz)/4, 3)} +/- 0.16", f"{round((r_mm_CSP3/r_mm_CSP3_baatz+slope_CSP3/r_mm_CSP3_baatz)/2,3)} +/- 0.21", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP3/r_mm_CSP3_baatz)/3, 3)} +/- 0.06", f"{round((slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/3, 3)} +/- 0.05", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP3/r_mm_CSP3_baatz+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/6, 3)} +/- 0.15"],
}

df_1 = pd.DataFrame(data_1)

def render_mpl_table(data_1, col_width=2.8, row_height=0.6, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_1.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(8,12))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_1.values, bbox=bbox, colLabels=data_1.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            cell.set_height(0.6)  # Set the height of header row to 0.6
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.set_height(0.5)  # Set the height of non-header rows to 0.2
    ax.set_title('Schrön Weighted Neutron Reduction')  # Set the title of the table
    return ax

render_mpl_table(df_1, header_columns=0, col_width=2.8)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Update_Summary_Table_with_TDR.png', dpi=300, bbox_inches='tight')
plt.show()


############################################
############################################
############################################
############################################
############################################
#ARITHMETIC FVEG
############################################
############################################
###################2023#####################
############################################
############################################

############################################
################## CSP1 ####################
############################################

## Using Schron N0 (intercept) values ##
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP1_arith = pd.read_csv('YOUR PATH/2023_CSP1_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
SWC_CSP1_arith = GWC_CSP1_arith['SWC_1_1_1']
a_CSP1_arith = SWC_CSP1_arith

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
### CALCUATE N ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
#For this, N0 is the intercept
pwc_CSP1_arith = a_CSP1_arith
lwc = 0.058
soc = 0.0092
N0_CSP1_arith = 2863.69

#use parameters to calculate N0
N_CSP1_arith = N0_CSP1_arith*(((0.0808)/((pwc_CSP1_arith+lwc+soc)+0.115))+0.372)
N_CSP1_arith.to_csv('YOUR PATH TO MAKE/Solve_for_N/N_CSP1_arithmetic.csv')
N_CSP1_23_arith = pd.read_csv('YOUR PATH/Solve_for_N/N_CSP1_arithmetic.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP1_2023_arith = N_CSP1_23_arith['SWC_1_1_1']

#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_weighted_schron_arith = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_arith = NPIV_CSP1_weighted_schron_arith['NPIV']

#Calculate forward
fveg_CSP1_arith = NPIVCount_CSP1_arith / N_CSP1_2023_arith
fveg_CSP1_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP1_arithmetic.csv')
fveg_CSP1_1_arith = pd.read_csv('YOUR PATH/fveg_CSP1_arithmetic.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_1_arith = fveg_CSP1_1_arith['0']

#Import BWE for plotting
BWEcsv_CSP1_arith = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_arith = BWEcsv_CSP1_arith['BWE']

#Set up plotting for both foward and inverse
x_CSP1_arith = BWE_CSP1_arith
y_CSP1_arith = 1/fveg_CSP1_1_arith

#Least square regression
A_CSP1_arith = np.vstack([x_CSP1_arith, np.ones(len(x_CSP1_arith))]).T
y_CSP1_arith = np.array(y_CSP1_arith) 
y_CSP1_arith = y_CSP1_arith[:, np.newaxis]
alpha_CSP1_arith = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_arith.T,A_CSP1_arith)),A_CSP1_arith.T)),y_CSP1_arith)
m_CSP1_arith = alpha_CSP1_arith[0]
n_CSP1_arith = alpha_CSP1_arith[1]
mm_CSP1_arith = float(m_CSP1_arith)
mmm_CSP1_arith = [float(m_CSP1_arith)]
nn_CSP1_arith = float(n_CSP1_arith)
r_mm_CSP1_arith = round(mm_CSP1_arith,4)
r_nn_CSP1_arith = round(nn_CSP1_arith,4)

#Calculate r-squared
x_CSP1_r2_arith = sm.add_constant(x_CSP1_arith)
model_arith = sm.OLS(y_CSP1_arith, x_CSP1_r2_arith)
results_arith = model_arith.fit()
r2_CSP1_arith = results_arith.rsquared
r2_CSP1_arith = round(r2_CSP1_arith,2)

#Plot data with best fit line, slope, and r-squared
a_CSP1_arith, b_CSP1_arith = np.polyfit(x_CSP1_arith,y_CSP1_arith,1)
plt.scatter(x_CSP1_arith, y_CSP1_arith, marker = 'o', color = 'red')
plt.plot(x_CSP1_arith, a_CSP1_arith*x_CSP1_arith+b_CSP1_arith, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP1_arith,r_nn_CSP1_arith, r2_CSP1_arith]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023_Arith.png', dpi=300, bbox_inches='tight')
plt.show()


############################################
################## CSP2 ####################
############################################

## Using Schron N0 (intercept) values ##
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP2_arith = pd.read_csv('YOUR PATH/2023_CSP2_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
SWC_CSP2_arith = GWC_CSP2_arith['SWC_1_1_1']
a_CSP2_arith = SWC_CSP2_arith

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
### CALCUATE N ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
#For this, N0 is the intercept
pwc_CSP2_arith = a_CSP2_arith
lwc = 0.058
soc = 0.0092
N0_CSP2_arith = 2951.22

#use parameters to calculate N0
N_CSP2_arith = N0_CSP2_arith*(((0.0808)/((pwc_CSP2_arith+lwc+soc)+0.115))+0.372)
N_CSP2_arith.to_csv('YOUR PATH TO MAKE/N_CSP2_arithmetic.csv')
N_CSP2_23_arith = pd.read_csv('YOUR PATH/N_CSP2_arithmetic.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP2_2023_arith = N_CSP2_23_arith['SWC_1_1_1']

#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_weighted_schron_arith = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_arith = NPIV_CSP2_weighted_schron_arith['NPIV']

#Calculate forward
fveg_CSP2_arith = NPIVCount_CSP2_arith / N_CSP2_2023_arith
fveg_CSP2_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP2_arithmetic.csv')
fveg_CSP2_1_arith = pd.read_csv('YOUR PATH/fveg_CSP2_arithmetic.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_1_arith = fveg_CSP2_1_arith['0']

#Import BWE for plotting
BWEcsv_CSP2_arith = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_arith = BWEcsv_CSP2_arith['BWE']

#Set up plotting for both foward and inverse
x_CSP2_arith = BWE_CSP2_arith
y_CSP2_arith = 1/fveg_CSP2_1_arith

#Least square regression
A_CSP2_arith = np.vstack([x_CSP2_arith, np.ones(len(x_CSP2_arith))]).T
y_CSP2_arith = np.array(y_CSP2_arith) 
y_CSP2_arith = y_CSP2_arith[:, np.newaxis]
alpha_CSP2_arith = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_arith.T,A_CSP2_arith)),A_CSP2_arith.T)),y_CSP2_arith)
m_CSP2_arith = alpha_CSP2_arith[0]
n_CSP2_arith = alpha_CSP2_arith[1]
mm_CSP2_arith = float(m_CSP2_arith)
mmm_CSP2_arith = [float(m_CSP2_arith)]
nn_CSP2_arith = float(n_CSP2_arith)
r_mm_CSP2_arith = round(mm_CSP2_arith,4)
r_nn_CSP2_arith = round(nn_CSP2_arith,4)

#Calculate r-squared
x_CSP2_r2_arith = sm.add_constant(x_CSP2_arith)
model_arith = sm.OLS(y_CSP2_arith, x_CSP2_r2_arith)
results_arith = model_arith.fit()
r2_CSP2_arith = results_arith.rsquared
r2_CSP2_arith = round(r2_CSP2_arith,2)

#Plot data with best fit line, slope, and r-squared
a_CSP2_arith, b_CSP2_arith = np.polyfit(x_CSP2_arith,y_CSP2_arith,1)
plt.scatter(x_CSP2_arith, y_CSP2_arith, marker = 'o', color = 'red')
plt.plot(x_CSP2_arith, a_CSP2_arith*x_CSP2_arith+b_CSP2_arith, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP2_arith,r_nn_CSP2_arith, r2_CSP2_arith]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2023_Arith.png', dpi=300, bbox_inches='tight')
plt.show()


############################################
################## CSP3 ####################
############################################

## Using Schron N0 (intercept) values ##
### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC_CSP3_arith = pd.read_csv('YOUR PATH/2023_CSP3_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
SWC_CSP3_arith = GWC_CSP3_arith['CRNS_SampleDepth']
a_CSP3_arith = SWC_CSP3_arith

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
### CALCUATE N ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
#For this, N0 is the intercept
pwc_CSP3_arith = a_CSP3_arith
lwc = 0.058
soc = 0.0092
N0_CSP3_arith = 1601.43

#use parameters to calculate N0
N_CSP3_arith = N0_CSP3_arith*(((0.0808)/((pwc_CSP3_arith+lwc+soc)+0.115))+0.372)
N_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_arithmetic.csv')
N_CSP3_23_arith = pd.read_csv('YOUR PATH/N_CSP3_arithmetic.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
N_CSP3_2023_arith = N_CSP3_23_arith['CRNS_SampleDepth']

#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_weighted_schron_arith = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_23_arith = NPIV_CSP3_weighted_schron_arith['NPIV']

#Calculate forward
fveg_CSP3_arith = NPIVCount_CSP3_23_arith / N_CSP3_2023_arith
fveg_CSP3_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_arithmetic.csv')
fveg_CSP3_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_arithmetic.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_1_arith = fveg_CSP3_1_arith['0']

#Import BWE for plotting
BWEcsv_CSP3_arith = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_arith = BWEcsv_CSP3_arith['BWE']

#Set up plotting for both foward and inverse
x_CSP3_arith = BWE_CSP3_arith
y_CSP3_arith = 1/fveg_CSP3_1_arith

#Least square regression
A_CSP3_arith = np.vstack([x_CSP3_arith, np.ones(len(x_CSP3_arith))]).T
y_CSP3_arith = np.array(y_CSP3_arith) 
y_CSP3_arith = y_CSP3_arith[:, np.newaxis]
alpha_CSP3_arith = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_arith.T,A_CSP3_arith)),A_CSP3_arith.T)),y_CSP3_arith)
m_CSP3_arith = alpha_CSP3_arith[0]
n_CSP3_arith = alpha_CSP3_arith[1]
mm_CSP3_arith = float(m_CSP3_arith)
mmm_CSP3_arith = [float(m_CSP3_arith)]
nn_CSP3_arith = float(n_CSP3_arith)
r_mm_CSP3_arith = round(mm_CSP3_arith,4)
r_nn_CSP3_arith = round(nn_CSP3_arith,4)

#Calculate r-squared
x_CSP3_r2_arith = sm.add_constant(x_CSP3_arith)
model_arith = sm.OLS(y_CSP3_arith, x_CSP3_r2_arith)
results_arith = model_arith.fit()
r2_CSP3_arith = results_arith.rsquared
r2_CSP3_arith = round(r2_CSP3_arith,2)

#Plot data with best fit line, slope, and r-squared
a_CSP3_arith, b_CSP3_arith = np.polyfit(x_CSP3_arith,y_CSP3_arith,1)
plt.scatter(x_CSP3_arith, y_CSP3_arith, marker = 'o', color = 'red')
plt.plot(x_CSP3_arith, a_CSP3_arith*x_CSP3_arith+b_CSP3_arith, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
#plt.xlim(-0.15, 2.25)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP3_arith,r_nn_CSP3_arith, r2_CSP3_arith]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2023_Arith.png', dpi=300, bbox_inches='tight')
plt.show()


##############################################
############ Table of Values #################
##############################################
#Baatz 2015 function 

############################################
################## CSP1 ####################
############################################

fvegbaatz_CSP1_arith = 1/((1-((-36.22/2767.93)*BWE_CSP1_arith)))
y_baatz_CSP1_arith = 1/fvegbaatz_CSP1_arith

#Least square regression
A_CSP1_baatz_arith = np.vstack([x_CSP1_arith, np.ones(len(x_CSP1_arith))]).T
y_CSP1_baatz_arith = np.array(y_baatz_CSP1_arith)  
y_CSP1_baatz_arith = y_baatz_CSP1_arith.values[:, np.newaxis]
alpha_CSP1_baatz_arith = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_baatz_arith.T,A_CSP1_baatz_arith)),A_CSP1_baatz_arith.T)),y_CSP1_baatz_arith)
m_CSP1_baatz_arith = alpha_CSP1_baatz_arith[0]
n_CSP1_baatz_arith = alpha_CSP1_baatz_arith[1]
mm_CSP1_baatz_arith = float(m_CSP1_baatz_arith)
mmm_CSP1_baatz_arith = [float(m_CSP1_baatz_arith)]
nn_CSP1_baatz_arith = float(n_CSP1_baatz_arith)
r_mm_CSP1_baatz_arith = round(mm_CSP1_baatz_arith,4)
r_nn_CSP1_baatz_arith = round(nn_CSP1_baatz_arith,4)

#Calculate r-squared   
x_CSP1_baatz_arith = sm.add_constant(x_CSP1_arith)
model_arith = sm.OLS(y_CSP1_baatz_arith, x_CSP1_baatz_arith)
results_arith = model_arith.fit()
r2_CSP1_baatz_arith = results_arith.rsquared
r2_CSP1_baatz_arith = round(r2_CSP1_baatz_arith,2)

#Plot data with best fit line, slope, and r-squared
a_CSP1_arith, b_CSP1_arith = np.polyfit(x_CSP1_arith,y_CSP1_arith,1)
c_CSP1_arith, d_CSP1_arith = np.polyfit(x_CSP1_arith,y_baatz_CSP1_arith,1)
plt.scatter(x_CSP1_arith, y_CSP1_arith, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP1_arith, a_CSP1_arith*x_CSP1_arith+b_CSP1_arith, linestyle = '-', color = 'red')
plt.plot(x_CSP1_arith, 0.005*x_CSP1_arith + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP1_arith], [r2_CSP1_arith]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP1_fveg_with_baatz_Arith.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
################## CSP2 ####################
############################################

fvegbaatz_CSP2_arith = 1/((1-((-49.36/2864.75)*BWE_CSP2_arith)))
y_baatz_CSP2_arith = 1/fvegbaatz_CSP2_arith

#Least square regression
A_CSP2_baatz_arith = np.vstack([x_CSP2_arith, np.ones(len(x_CSP2_arith))]).T
y_CSP2_baatz_arith = np.array(y_baatz_CSP2_arith)  
y_CSP2_baatz_arith = y_baatz_CSP2_arith.values[:, np.newaxis]
alpha_CSP2_baatz_arith = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_baatz_arith.T,A_CSP2_baatz_arith)),A_CSP2_baatz_arith.T)),y_CSP2_baatz_arith)
m_CSP2_baatz_arith = alpha_CSP2_baatz_arith[0]
n_CSP2_baatz_arith = alpha_CSP2_baatz_arith[1]
mm_CSP2_baatz_arith = float(m_CSP2_baatz_arith)
mmm_CSP2_baatz_arith = [float(m_CSP2_baatz_arith)]
nn_CSP2_baatz_arith = float(n_CSP2_baatz_arith)
r_mm_CSP2_baatz_arith = round(mm_CSP2_baatz_arith,4)
r_nn_CSP2_baatz_arith = round(nn_CSP2_baatz_arith,4)

#Calculate r-squared
x_CSP2_baatz_arith = sm.add_constant(x_CSP2_arith)
model_arith = sm.OLS(y_CSP2_baatz_arith, x_CSP2_baatz_arith)
results_arith = model_arith.fit()
r2_CSP2_baatz_arith = results_arith.rsquared
r2_CSP2_baatz_arith = round(r2_CSP2_baatz_arith,2)

#Plot data with best fit line, slope, and r-squared
a_CSP2_arith, b_CSP2_arith = np.polyfit(x_CSP2_arith,y_CSP2_arith,1)
c_CSP2_arith, d_CSP2_arith = np.polyfit(x_CSP2_arith,y_baatz_CSP2_arith,1)
plt.scatter(x_CSP2_arith, y_CSP2_arith, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP2_arith, a_CSP2_arith*x_CSP2_arith+b_CSP2_arith, linestyle = '-', color = 'red')
plt.plot(x_CSP2_arith, 0.005*x_CSP2_arith + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP2_arith], [r2_CSP2_arith]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP2_fveg_with_Baatz_Arith.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
################## CSP3 ####################
############################################

fvegbaatz_CSP3_arith = 1/((1-((-11.7/1542.57)*BWE_CSP3_arith)))
y_baatz_CSP3_arith = 1/fvegbaatz_CSP3_arith

#Least square regression
A_CSP3_baatz_arith = np.vstack([x_CSP3_arith, np.ones(len(x_CSP3_arith))]).T
y_CSP3_baatz_arith = np.array(y_baatz_CSP3_arith)  
y_CSP3_baatz_arith = y_baatz_CSP3_arith.values[:, np.newaxis]
alpha_CSP3_baatz_arith = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_baatz_arith.T,A_CSP3_baatz_arith)),A_CSP3_baatz_arith.T)),y_CSP3_baatz_arith)
m_CSP3_baatz_arith = alpha_CSP3_baatz_arith[0]
n_CSP3_baatz_arith = alpha_CSP3_baatz_arith[1]
mm_CSP3_baatz_arith = float(m_CSP3_baatz_arith)
mmm_CSP3_baatz_arith = [float(m_CSP3_baatz_arith)]
nn_CSP3_baatz_arith = float(n_CSP3_baatz_arith)
r_mm_CSP3_baatz_arith = round(mm_CSP3_baatz_arith,4)
r_nn_CSP3_baatz_arith = round(nn_CSP3_baatz_arith,4)

#Calculate r-squared
x_CSP3_baatz_arith = sm.add_constant(x_CSP3_arith)
model_arith = sm.OLS(y_CSP3_baatz_arith, x_CSP3_baatz_arith)
results_arith = model.fit()
r2_CSP3_baatz_arith = results_arith.rsquared
r2_CSP3_baatz_arith = round(r2_CSP3_baatz_arith,2)

#Plot data with best fit line, slope, and r-squared
a_CSP3_arith, b_CSP3_arith = np.polyfit(x_CSP3_arith,y_CSP3_arith,1)
c_CSP3_arith, d_CSP3_arith = np.polyfit(x_CSP3_arith,y_baatz_CSP3_arith,1)
plt.scatter(x_CSP3_arith, y_CSP3_arith, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP3_arith, a_CSP3_arith*x_CSP3_arith+b_CSP3_arith, linestyle = '-', color = 'red')
plt.plot(x_CSP3_arith, 0.005*x_CSP3_arith + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3_arith], [r2_CSP3_arith]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_fveg_with_Baatz_Arith.png', dpi=300, bbox_inches='tight')
plt.show()

# Normalize according to Baatz 2015
normal_CSP1_arith = (27.84/2863.69)*100
normal_CSP2_arith = (48.36/2951.22)*100
normal_CSP3_arith = (8.18/1601.43)*100

##############################################
############ Table of Values #################
##############################################

data_arith = {
    '': ['Slope', 'Intercept', 'Baatz 2015\nSlope', 'Baatz 2015\nIntercept', 'Slope\nDifference', 'Slope\nRatio', 'Neutron\nIntensity\nReduction(%)'],
    'CSP1': [f"{r_mm_CSP1_arith}", f"{r_nn_CSP1_arith}", f"{r_mm_CSP1_baatz_arith}", f"{r_nn_CSP1_baatz_arith}", f"{round((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith),4)}", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith),4)}", f"{round((normal_CSP1_arith),2)}"],
    'CSP2': [f"{r_mm_CSP2_arith}", f"{r_nn_CSP2_arith}", f"{r_mm_CSP2_baatz_arith}", f"{r_nn_CSP2_baatz_arith}", f"{round((r_mm_CSP2_arith-r_mm_CSP2_baatz_arith),4)}", f"{round((r_mm_CSP2_arith/r_mm_CSP2_baatz_arith),4)}", f"{round((normal_CSP2_arith),2)}"],
    'CSP3': [f"{r_mm_CSP3_arith}", f"{r_nn_CSP3_arith}", f"{r_mm_CSP3_baatz_arith}", f"{r_nn_CSP3_baatz_arith}", f"{round((r_mm_CSP3_arith-r_mm_CSP3_baatz_arith),4)}", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith),4)}", f"{round((normal_CSP3_arith),2)}"],
    'Average': [f"{round((r_mm_CSP1_arith+r_mm_CSP2_arith+r_mm_CSP3_arith)/3,4)}", f"{round((r_nn_CSP1_arith+r_nn_CSP2_arith+r_nn_CSP3_arith)/3,4)}", f"{round((r_mm_CSP1_baatz_arith+r_mm_CSP2_baatz_arith+r_mm_CSP3_baatz_arith)/3, 4)}", f"{round((r_nn_CSP1_baatz_arith+r_nn_CSP2_baatz_arith+r_nn_CSP3_baatz_arith)/3,4)}", f"{round((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith+r_mm_CSP2_arith-r_mm_CSP2_baatz_arith+r_mm_CSP3_arith-r_mm_CSP3_baatz_arith)/3,4)}", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP3_arith/r_mm_CSP3_baatz_arith)/3,4)}", f"{round((normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith)/3,2)}"]
}

# Convert the dictionary to a pandas DataFrame
df_arith = pd.DataFrame(data_arith)

# Function to render DataFrame as a table
def render_mpl_table(data_arith, col_width=3.0, row_height=0.7, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_arith.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(12,7))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_arith.values, bbox=bbox, colLabels=data_arith.columns, **kwargs)

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

render_mpl_table(df_arith, header_columns=0, col_width=2.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Summary_Table_Arith.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP1 ################################
##################################################################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2019_CSP1_arith = pd.read_csv('YOUR PATH/AmerifluxCSP12019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2020_CSP1_arith = pd.read_csv('YOUR PATH/AmerifluxCSP12020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP1_arith = pd.read_csv('YOUR PATH/AmerifluxCSP12021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP1_arith = pd.read_csv('YOUR PATH/AmerifluxCSP12022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP1_arith = pd.read_csv('YOUR PATH/AmerifluxCSP12023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC19_PI_F_1_1_1_CSP1_arith = GWC2019_CSP1_arith['SWC_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP1_arith = GWC2019_CSP1_arith['SWC_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP1_arith = GWC2019_CSP1_arith['SWC_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP1_arith = GWC2019_CSP1_arith['SWC_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP1_arith = GWC2019_CSP1_arith['SWC_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP1_arith = GWC2019_CSP1_arith['SWC_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP1_arith = GWC2019_CSP1_arith['SWC_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP1_arith = GWC2019_CSP1_arith['SWC_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP1_arith = GWC2019_CSP1_arith['SWC_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP1_arith = GWC2019_CSP1_arith['SWC_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP1_arith = GWC2019_CSP1_arith['SWC_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP1_arith = GWC2019_CSP1_arith['SWC_3_4_1']/1.42

SWC20_PI_F_1_1_1_CSP1_arith = GWC2020_CSP1_arith['SWC_1_1_1']/1.42
SWC20_PI_F_1_2_1_CSP1_arith = GWC2020_CSP1_arith['SWC_1_2_1']/1.42
SWC20_PI_F_1_3_1_CSP1_arith = GWC2020_CSP1_arith['SWC_1_3_1']/1.42
SWC20_PI_F_1_4_1_CSP1_arith = GWC2020_CSP1_arith['SWC_1_4_1']/1.42
SWC20_PI_F_2_1_1_CSP1_arith = GWC2020_CSP1_arith['SWC_2_1_1']/1.42
SWC20_PI_F_2_2_1_CSP1_arith = GWC2020_CSP1_arith['SWC_2_2_1']/1.42
SWC20_PI_F_2_3_1_CSP1_arith = GWC2020_CSP1_arith['SWC_2_3_1']/1.42
SWC20_PI_F_2_4_1_CSP1_arith = GWC2020_CSP1_arith['SWC_2_4_1']/1.42
SWC20_PI_F_3_1_1_CSP1_arith = GWC2020_CSP1_arith['SWC_3_1_1']/1.42
SWC20_PI_F_3_2_1_CSP1_arith = GWC2020_CSP1_arith['SWC_3_2_1']/1.42
SWC20_PI_F_3_3_1_CSP1_arith = GWC2020_CSP1_arith['SWC_3_3_1']/1.42
SWC20_PI_F_3_4_1_CSP1_arith = GWC2020_CSP1_arith['SWC_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_2_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP1_arith = GWC2021_CSP1_arith['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP1_arith = GWC2022_CSP1_arith['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP1_arith = GWC2023_CSP1_arith['SWC_F_3_4_1']/1.42

twosenwavgall2019_CSP1_arith = ((((SWC19_PI_F_1_1_1_CSP1_arith + SWC19_PI_F_2_1_1_CSP1_arith + SWC19_PI_F_3_1_1_CSP1_arith)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP1_arith + SWC19_PI_F_2_2_1_CSP1_arith + SWC19_PI_F_3_2_1_CSP1_arith)/3)*0.25)) / 100
twosenwavgall2020_CSP1_arith = ((((SWC20_PI_F_1_1_1_CSP1_arith + SWC20_PI_F_2_1_1_CSP1_arith + SWC20_PI_F_3_1_1_CSP1_arith)/3)*0.75) + (((SWC20_PI_F_1_2_1_CSP1_arith + SWC20_PI_F_2_2_1_CSP1_arith + SWC20_PI_F_3_2_1_CSP1_arith)/3)*0.25)) / 100
twosenwavgall2021_CSP1_arith = ((((SWC21_PI_F_1_1_1_CSP1_arith + SWC21_PI_F_2_1_1_CSP1_arith + SWC21_PI_F_3_1_1_CSP1_arith)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP1_arith + SWC21_PI_F_2_2_1_CSP1_arith + SWC21_PI_F_3_2_1_CSP1_arith)/3)*0.25)) / 100
twosenwavgall2022_CSP1_arith = ((((SWC22_PI_F_1_1_1_CSP1_arith + SWC22_PI_F_2_1_1_CSP1_arith + SWC22_PI_F_3_1_1_CSP1_arith)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP1_arith + SWC22_PI_F_2_2_1_CSP1_arith + SWC22_PI_F_3_2_1_CSP1_arith)/3)*0.25)) / 100
twosenwavgall2023_CSP1_arith = ((((SWC23_PI_F_1_1_1_CSP1_arith + SWC23_PI_F_2_1_1_CSP1_arith + SWC23_PI_F_3_1_1_CSP1_arith)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP1_arith + SWC23_PI_F_2_2_1_CSP1_arith + SWC23_PI_F_3_2_1_CSP1_arith)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2019_CSP1_arith = twosenwavgall2019_CSP1_arith.resample('D').mean() 
dailywavgSWC2020_CSP1_arith = twosenwavgall2020_CSP1_arith.resample('D').mean() 
dailywavgSWC2021_CSP1_arith = twosenwavgall2021_CSP1_arith.resample('D').mean() 
dailywavgSWC2022_CSP1_arith = twosenwavgall2022_CSP1_arith.resample('D').mean() 
dailywavgSWC2023_CSP1_arith = twosenwavgall2023_CSP1_arith.resample('D').mean()

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2019_CSP1_arith = pd.read_csv('YOUR PATH/CSP12019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2020_CSP1_arith = pd.read_csv('YOUR PATH/CSP12020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP1_arith = pd.read_csv('YOUR PATH/CSP12021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP1_arith = pd.read_csv('YOUR PATH/CSP12022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP1_arith = pd.read_csv('YOUR PATH/CSP12023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2019_CSP1_arith = NPIV2019_CSP1_arith['NPIV']
NPIVCount2020_CSP1_arith = NPIV2020_CSP1_arith['NPIV']
NPIVCount2021_CSP1_arith = NPIV2021_CSP1_arith['NPIV']
NPIVCount2022_CSP1_arith = NPIV2022_CSP1_arith['NPIV']
NPIVCount2023_CSP1_arith = NPIV2023_CSP1_arith['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2019_CSP1_arith = NPIVCount2019_CSP1_arith.resample('D').mean()
dailyNPIV2020_CSP1_arith = NPIVCount2020_CSP1_arith.resample('D').mean()
dailyNPIV2021_CSP1_arith = NPIVCount2021_CSP1_arith.resample('D').mean()
dailyNPIV2022_CSP1_arith = NPIVCount2022_CSP1_arith.resample('D').mean()
dailyNPIV2023_CSP1_arith = NPIVCount2023_CSP1_arith.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc19_CSP1_arith = dailywavgSWC2019_CSP1_arith
pwc20_CSP1_arith = dailywavgSWC2020_CSP1_arith
pwc21_CSP1_arith = dailywavgSWC2021_CSP1_arith
pwc22_CSP1_arith = dailywavgSWC2022_CSP1_arith
pwc23_CSP1_arith = dailywavgSWC2023_CSP1_arith

lwc = 0.058
soc = 0.0092
N0_for_calc_arith = 2863.69

N19_CSP1_arith = dailyNPIV2019_CSP1_arith
N20_CSP1_arith = dailyNPIV2020_CSP1_arith
N21_CSP1_arith = dailyNPIV2021_CSP1_arith
N22_CSP1_arith = dailyNPIV2022_CSP1_arith
N23_CSP1_arith = dailyNPIV2023_CSP1_arith

#Calculate N
N0_CSP1_19_arith = N0_for_calc_arith*(((0.0808)/((pwc19_CSP1_arith+lwc+soc)+0.115))+0.372)
N0_CSP1_20_arith = N0_for_calc_arith*(((0.0808)/((pwc20_CSP1_arith+lwc+soc)+0.115))+0.372)
N0_CSP1_21_arith = N0_for_calc_arith*(((0.0808)/((pwc21_CSP1_arith+lwc+soc)+0.115))+0.372)
N0_CSP1_22_arith = N0_for_calc_arith*(((0.0808)/((pwc22_CSP1_arith+lwc+soc)+0.115))+0.372)
N0_CSP1_23_arith = N0_for_calc_arith*(((0.0808)/((pwc23_CSP1_arith+lwc+soc)+0.115))+0.372)

NCSP1_19_arith= N0_CSP1_19_arith.to_csv('YOUR PATH TO MAKE/N_CSP1_19_arith.csv')
NCSP1_20_arith= N0_CSP1_20_arith.to_csv('YOUR PATH TO MAKE/N_CSP1_20_arith.csv')
NCSP1_21_arith= N0_CSP1_21_arith.to_csv('YOUR PATH TO MAKE/N_CSP1_21_arith.csv')
NCSP1_22_arith= N0_CSP1_22_arith.to_csv('YOUR PATH TO MAKE/N_CSP1_22_arith.csv')
NCSP1_23_arith= N0_CSP1_23_arith.to_csv('YOUR PATH TO MAKE/N_CSP1_23_arith.csv')

CSP1_19_arith = pd.read_csv('YOUR PATH/N_CSP1_19_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_20_arith = pd.read_csv('YOUR PATH/N_CSP1_20_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_21_arith = pd.read_csv('YOUR PATH/N_CSP1_21_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_22_arith = pd.read_csv('YOUR PATH/N_CSP1_22_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_23_arith = pd.read_csv('YOUR PATH/N_CSP1_23_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP1_19_1_arith = CSP1_19_arith['0']
N_CSP1_20_1_arith = CSP1_20_arith['0']
N_CSP1_21_1_arith = CSP1_21_arith['0']
N_CSP1_22_1_arith = CSP1_22_arith['0']
N_CSP1_23_1_arith = CSP1_23_arith['0']

fveg_CSP1_19_arith = dailyNPIV2019_CSP1_arith / N0_CSP1_19_arith
fveg_CSP1_20_arith = dailyNPIV2020_CSP1_arith / N0_CSP1_20_arith
fveg_CSP1_21_arith = dailyNPIV2021_CSP1_arith / N0_CSP1_21_arith
fveg_CSP1_22_arith = dailyNPIV2022_CSP1_arith / N0_CSP1_22_arith
fveg_CSP1_23_arith = dailyNPIV2023_CSP1_arith / N0_CSP1_23_arith

fveg_CSP1_19_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP1_19_arith.csv')
fveg_CSP1_20_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP1_20_arith.csv')
fveg_CSP1_21_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP1_21_arith.csv')
fveg_CSP1_22_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP1_22_arith.csv')
fveg_CSP1_23_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP1_23_arith.csv')

fveg_CSP1_19_1_arith = pd.read_csv('YOUR PATH/fveg_CSP1_19_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_20_1_arith = pd.read_csv('YOUR PATH/fveg_CSP1_20_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_21_1_arith = pd.read_csv('YOUR PATH/fveg_CSP1_21_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_22_1_arith = pd.read_csv('YOUR PATH/fveg_CSP1_22_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_23_1_arith = pd.read_csv('YOUR PATH/fveg_CSP1_23_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP1_19_1_arith = fveg_CSP1_19_1_arith['0']
fveg_CSP1_20_1_arith = fveg_CSP1_20_1_arith['0']
fveg_CSP1_21_1_arith = fveg_CSP1_21_1_arith['0']
fveg_CSP1_22_1_arith = fveg_CSP1_22_1_arith['0']
fveg_CSP1_23_1_arith = fveg_CSP1_23_1_arith['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2019_CSP1_arith = pd.read_csv('YOUR PATH/CSP12019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP1_arith = pd.read_csv('YOUR PATH/CSP12020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP1_arith = pd.read_csv('YOUR PATH/CSP12021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP1_arith = pd.read_csv('YOUR PATH/CSP12022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP1_arith = pd.read_csv('YOUR PATH/CSP12023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE19_CSP1_arith = BWEcsv2019_CSP1_arith['BWE']
BWE20_CSP1_arith = BWEcsv2020_CSP1_arith['BWE']
BWE21_CSP1_arith = BWEcsv2021_CSP1_arith['BWE']
BWE22_CSP1_arith = BWEcsv2022_CSP1_arith['BWE']
BWE23_CSP1_arith = BWEcsv2023_CSP1_arith['BWE']

#Interpolate between dates
BWEinterp19_CSP1_arith = BWE19_CSP1_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP1_arith = BWE20_CSP1_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP1_arith = BWE21_CSP1_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP1_arith = BWE22_CSP1_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP1_arith = BWE23_CSP1_arith.interpolate(method = 'linear', limit_direction = 'forward')

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP1_arith = BWEinterp19_CSP1_arith
y19_CSP1_arith = 1/fveg_CSP1_19_1_arith
x20_CSP1_arith = BWEinterp20_CSP1_arith
y20_CSP1_arith = 1/fveg_CSP1_20_1_arith
x21_CSP1_arith = BWEinterp21_CSP1_arith
y21_CSP1_arith = 1/fveg_CSP1_21_1_arith
x22_CSP1_arith = BWEinterp22_CSP1_arith
y22_CSP1_arith = 1/fveg_CSP1_22_1_arith
x23_CSP1_arith = BWEinterp23_CSP1_arith
y23_CSP1_arith = 1/fveg_CSP1_23_1_arith

# Concatenate all x and y values
x_all_CSP1_arith = np.concatenate([x19_CSP1_arith, x20_CSP1_arith, x21_CSP1_arith, x22_CSP1_arith, x23_CSP1_arith])
y_all_CSP1_arith = np.concatenate([y19_CSP1_arith, y20_CSP1_arith, y21_CSP1_arith, y22_CSP1_arith, y23_CSP1_arith])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP1_arith = x_all_CSP1_arith.reshape(-1, 1)

# Perform linear regression
slope_CSP1_arith, intercept_CSP1_arith, r_value_CSP1_arith, p_value_CSP1_arith, std_err_CSP1_arith = stats.linregress(x_all_CSP1_arith, y_all_CSP1_arith)

# Calculate the t-value for a 95% confidence interval
t_CSP1_arith = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP1_arith) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP1_arith = t_CSP1_arith * std_err_CSP1_arith
intercept_conf_interval_CSP1_arith = t_CSP1_arith * std_err_CSP1_arith * np.sqrt((1/len(x_all_CSP1_arith)) + (np.mean(x_all_CSP1_arith)**2 / np.sum((x_all_CSP1_arith - np.mean(x_all_CSP1_arith))**2)))

# Calculate R-squared value
r_squared_CSP1_arith = r_value_CSP1_arith**2

shift_arith = 1 - intercept_CSP1_arith

plt.scatter(x19_CSP1_arith, y19_CSP1_arith + shift_arith, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP1_arith, y20_CSP1_arith + shift_arith, color = 'cyan', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP1_arith, y21_CSP1_arith + shift_arith, color = 'darkgreen', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP1_arith, y22_CSP1_arith + shift_arith, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP1_arith, y23_CSP1_arith + shift_arith, color = 'darkorange', marker = 'D', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP1_arith, b_CSP1_arith = np.polyfit(x_CSP1_arith,y_CSP1_arith,1)
c_CSP1_arith, d_CSP1_arith = np.polyfit(x_CSP1_arith,y_baatz_CSP1_arith,1)
plt.scatter(x_CSP1_arith, y_CSP1_arith, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP1_arith, 0.005*x_CSP1_arith + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP1_arith, slope_CSP1_arith*x_all_CSP1_arith + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP1_arith, a_CSP1_arith*x_CSP1_arith+b_CSP1_arith, linestyle = '-', color = 'red', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP1_arith, round(slope_CSP1_arith,2)], [r2_CSP1_arith, round(r_squared_CSP1_arith, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP1_with_TDR_arith.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP2 ################################
##################################################################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2019_CSP2_arith = pd.read_csv('YOUR PATH/AmerifluxCSP22019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2020_CSP2_arith = pd.read_csv('YOUR PATH/AmerifluxCSP22020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP2_arith = pd.read_csv('YOUR PATH/AmerifluxCSP22021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP2_arith = pd.read_csv('YOUR PATH/AmerifluxCSP22022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP2_arith = pd.read_csv('YOUR PATH/AmerifluxCSP22023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC19_PI_F_1_1_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP2_arith = GWC2019_CSP2_arith['SWC_PI_F_3_4_1']/1.42

SWC20_PI_F_1_1_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_1_1_1']/1.42
SWC20_PI_F_1_2_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_1_2_1']/1.42
SWC20_PI_F_1_3_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_1_3_1']/1.42
SWC20_PI_F_1_4_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_1_4_1']/1.42
SWC20_PI_F_2_1_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_2_1_1']/1.42
SWC20_PI_F_2_2_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_2_2_1']/1.42
SWC20_PI_F_2_3_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_2_3_1']/1.42
SWC20_PI_F_2_4_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_2_4_1']/1.42
SWC20_PI_F_3_1_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_3_1_1']/1.42
SWC20_PI_F_3_2_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_3_2_1']/1.42
SWC20_PI_F_3_3_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_3_3_1']/1.42
SWC20_PI_F_3_4_1_CSP2_arith = GWC2020_CSP2_arith['SWC_PI_F_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP2_arith = GWC2021_CSP2_arith['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP2_arith = GWC2022_CSP2_arith['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP2_arith = GWC2023_CSP2_arith['SWC_F_3_4_1']/1.42

twosenwavgall2019_CSP2_arith = ((((SWC19_PI_F_1_1_1_CSP2_arith + SWC19_PI_F_2_1_1_CSP2_arith + SWC19_PI_F_3_1_1_CSP2_arith)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP2_arith + SWC19_PI_F_2_2_1_CSP2_arith + SWC19_PI_F_3_2_1_CSP2_arith)/3)*0.25)) / 100
twosenwavgall2020_CSP2_arith = ((((SWC20_PI_F_1_1_1_CSP2_arith + SWC20_PI_F_2_1_1_CSP2_arith + SWC20_PI_F_3_1_1_CSP2_arith)/3)*0.75) + (((SWC20_PI_F_1_2_1_CSP2_arith + SWC20_PI_F_2_2_1_CSP2_arith + SWC20_PI_F_3_2_1_CSP2_arith)/3)*0.25)) / 100
twosenwavgall2021_CSP2_arith = ((((SWC21_PI_F_1_1_1_CSP2_arith + SWC21_PI_F_2_1_1_CSP2_arith + SWC21_PI_F_3_1_1_CSP2_arith)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP2_arith + SWC21_PI_F_2_2_1_CSP2_arith + SWC21_PI_F_3_2_1_CSP2_arith)/3)*0.25)) / 100
twosenwavgall2022_CSP2_arith = ((((SWC22_PI_F_1_1_1_CSP2_arith + SWC22_PI_F_2_1_1_CSP2_arith + SWC22_PI_F_3_1_1_CSP2_arith)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP2_arith + SWC22_PI_F_2_2_1_CSP2_arith + SWC22_PI_F_3_2_1_CSP2_arith)/3)*0.25)) / 100
twosenwavgall2023_CSP2_arith = ((((SWC23_PI_F_1_1_1_CSP2_arith + SWC23_PI_F_2_1_1_CSP2_arith + SWC23_PI_F_3_1_1_CSP2_arith)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP2_arith + SWC23_PI_F_2_2_1_CSP2_arith + SWC23_PI_F_3_2_1_CSP2_arith)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2019_CSP2_arith = twosenwavgall2019_CSP2_arith.resample('D').mean() 
dailywavgSWC2020_CSP2_arith = twosenwavgall2020_CSP2_arith.resample('D').mean() 
dailywavgSWC2021_CSP2_arith = twosenwavgall2021_CSP2_arith.resample('D').mean() 
dailywavgSWC2022_CSP2_arith = twosenwavgall2022_CSP2_arith.resample('D').mean() 
dailywavgSWC2023_CSP2_arith = twosenwavgall2023_CSP2_arith.resample('D').mean() 

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2019_CSP2_arith = pd.read_csv('YOUR PATH/CSP22019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2020_CSP2_arith = pd.read_csv('YOUR PATH/CSP22020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP2_arith = pd.read_csv('YOUR PATH/CSP22021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP2_arith = pd.read_csv('YOUR PATH/CSP22022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP2_arith = pd.read_csv('YOUR PATH/CSP22023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2019_CSP2_arith = NPIV2019_CSP2_arith['NPIV']
NPIVCount2020_CSP2_arith = NPIV2020_CSP2_arith['NPIV']
NPIVCount2021_CSP2_arith = NPIV2021_CSP2_arith['NPIV']
NPIVCount2022_CSP2_arith = NPIV2022_CSP2_arith['NPIV']
NPIVCount2023_CSP2_arith = NPIV2023_CSP2_arith['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2019_CSP2_arith = NPIVCount2019_CSP2_arith.resample('D').mean()
dailyNPIV2020_CSP2_arith = NPIVCount2020_CSP2_arith.resample('D').mean()
dailyNPIV2021_CSP2_arith = NPIVCount2021_CSP2_arith.resample('D').mean()
dailyNPIV2022_CSP2_arith = NPIVCount2022_CSP2_arith.resample('D').mean()
dailyNPIV2023_CSP2_arith = NPIVCount2023_CSP2_arith.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc19_CSP2_arith = dailywavgSWC2019_CSP2_arith
pwc20_CSP2_arith = dailywavgSWC2020_CSP2_arith
pwc21_CSP2_arith = dailywavgSWC2021_CSP2_arith
pwc22_CSP2_arith = dailywavgSWC2022_CSP2_arith
pwc23_CSP2_arith = dailywavgSWC2023_CSP2_arith

N19_CSP2_arith = dailyNPIV2019_CSP2_arith
N20_CSP2_arith = dailyNPIV2020_CSP2_arith
N21_CSP2_arith = dailyNPIV2021_CSP2_arith
N22_CSP2_arith = dailyNPIV2022_CSP2_arith
N23_CSP2_arith = dailyNPIV2023_CSP2_arith

N0_for_calc_2_arith = 2951.22
lwc = 0.058
soc = 0.0092

#Calculate N
N0_CSP2_19_arith = N0_for_calc_2_arith*(((0.0808)/((pwc19_CSP2_arith+lwc+soc)+0.115))+0.372)
N0_CSP2_20_arith = N0_for_calc_2_arith*(((0.0808)/((pwc20_CSP2_arith+lwc+soc)+0.115))+0.372)
N0_CSP2_21_arith = N0_for_calc_2_arith*(((0.0808)/((pwc21_CSP2_arith+lwc+soc)+0.115))+0.372)
N0_CSP2_22_arith = N0_for_calc_2_arith*(((0.0808)/((pwc22_CSP2_arith+lwc+soc)+0.115))+0.372)
N0_CSP2_23_arith = N0_for_calc_2_arith*(((0.0808)/((pwc23_CSP2_arith+lwc+soc)+0.115))+0.372)

N0interp19_CSP2_arith = N0_CSP2_19_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP2_arith = N0_CSP2_20_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP2_arith = N0_CSP2_21_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP2_arith = N0_CSP2_22_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP2_arith = N0_CSP2_23_arith.interpolate(method = 'linear', limit_direction = 'forward')

NCSP2_19_arith= N0interp19_CSP2_arith.to_csv('YOUR PATH TO MAKE/N_CSP2_19_arith.csv')
NCSP2_20_arith= N0interp20_CSP2_arith.to_csv('YOUR PATH TO MAKE/N_CSP2_20_arith.csv')
NCSP2_21_arith= N0interp21_CSP2_arith.to_csv('YOUR PATH TO MAKE/N_CSP2_21_arith.csv')
NCSP2_22_arith= N0interp22_CSP2_arith.to_csv('YOUR PATH TO MAKE/N_CSP2_22_arith.csv')
NCSP2_23_arith= N0interp23_CSP2_arith.to_csv('YOUR PATH TO MAKE/N_CSP2_23_arith.csv')

CSP2_19_arith = pd.read_csv('YOUR PATH/N_CSP2_19_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_20_arith = pd.read_csv('YOUR PATH/N_CSP2_20_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_21_arith = pd.read_csv('YOUR PATH/N_CSP2_21_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_22_arith = pd.read_csv('YOUR PATH/N_CSP2_22_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_23_arith = pd.read_csv('YOUR PATH/N_CSP2_23_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP2_19_1_arith = CSP2_19_arith['0']
N_CSP2_20_1_arith = CSP2_20_arith['0']
N_CSP2_21_1_arith = CSP2_21_arith['0']
N_CSP2_22_1_arith = CSP2_22_arith['0']
N_CSP2_23_1_arith = CSP2_23_arith['0']

fveg_CSP2_19_arith = dailyNPIV2019_CSP2_arith / N0_CSP2_19_arith
fveg_CSP2_20_arith = dailyNPIV2020_CSP2_arith / N0_CSP2_20_arith
fveg_CSP2_21_arith = dailyNPIV2021_CSP2_arith / N0_CSP2_21_arith
fveg_CSP2_22_arith = dailyNPIV2022_CSP2_arith / N0_CSP2_22_arith
fveg_CSP2_23_11_arith = dailyNPIV2023_CSP2_arith / N0_CSP2_23_arith

fveg_CSP2_19_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP2_19_arith.csv')
fveg_CSP2_20_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP2_20_arith.csv')
fveg_CSP2_21_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP2_21_arith.csv')
fveg_CSP2_22_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP2_22_arith.csv')
fveg_CSP2_23_11_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP2_23_arith.csv')

fveg_CSP2_19_1_arith = pd.read_csv('YOUR PATH/fveg_CSP2_19_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_20_1_arith = pd.read_csv('YOUR PATH/fveg_CSP2_20_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_21_1_arith = pd.read_csv('YOUR PATH/fveg_CSP2_21_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_22_1_arith = pd.read_csv('YOUR PATH/fveg_CSP2_22_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_23_1_arith = pd.read_csv('YOUR PATH/fveg_CSP2_23_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP2_19_1_arith = fveg_CSP2_19_1_arith['0']
fveg_CSP2_20_1_arith = fveg_CSP2_20_1_arith['0']
fveg_CSP2_21_1_arith = fveg_CSP2_21_1_arith['0']
fveg_CSP2_22_1_arith = fveg_CSP2_22_1_arith['0']
fveg_CSP2_23_1_arith = fveg_CSP2_23_1_arith['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2019_CSP2_arith = pd.read_csv('YOUR PATH/CSP22019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP2_arith = pd.read_csv('YOUR PATH/CSP22020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP2_arith = pd.read_csv('YOUR PATH/CSP22021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP2_arith = pd.read_csv('YOUR PATH/CSP22022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP2_arith = pd.read_csv('YOUR PATH/CSP22023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE19_CSP2_arith = BWEcsv2019_CSP2_arith['BWE']
BWE20_CSP2_arith = BWEcsv2020_CSP2_arith['BWE']
BWE21_CSP2_arith = BWEcsv2021_CSP2_arith['BWE']
BWE22_CSP2_arith = BWEcsv2022_CSP2_arith['BWE']
BWE23_CSP2_arith = BWEcsv2023_CSP2_arith['BWE']

#Interpolate between dates
BWEinterp19_CSP2_arith = BWE19_CSP2_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP2_arith = BWE20_CSP2_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP2_arith = BWE21_CSP2_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP2_arith = BWE22_CSP2_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP2_arith = BWE23_CSP2_arith.interpolate(method = 'linear', limit_direction = 'forward')

fveg_CSP2_19_1_arith = fveg_CSP2_19_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_20_1_arith = fveg_CSP2_20_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_21_1_arith = fveg_CSP2_21_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_22_1_arith = fveg_CSP2_22_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_23_1_arith = fveg_CSP2_23_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan)

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP2_arith = BWEinterp19_CSP2_arith
y19_CSP2_arith = 1/fveg_CSP2_19_1_arith
x20_CSP2_arith = BWEinterp20_CSP2_arith
y20_CSP2_arith = 1/fveg_CSP2_20_1_arith
x21_CSP2_arith = BWEinterp21_CSP2_arith
y21_CSP2_arith = 1/fveg_CSP2_21_1_arith
x22_CSP2_arith = BWEinterp22_CSP2_arith
y22_CSP2_arith = 1/fveg_CSP2_22_1_arith
x23_CSP2_arith = BWEinterp23_CSP2_arith
y23_CSP2_arith = 1/fveg_CSP2_23_1_arith

def fill_nan_with_avg(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr
y23_CSP2_arith = fill_nan_with_avg(y23_CSP2_arith)
# Concatenate all x and y values
x_all_CSP2_arith = np.concatenate([x19_CSP2_arith, x20_CSP2_arith, x21_CSP2_arith, x22_CSP2_arith, x23_CSP2_arith])
y_all_CSP2_arith = np.concatenate([y19_CSP2_arith, y20_CSP2_arith, y21_CSP2_arith, y22_CSP2_arith, y23_CSP2_arith])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP2_arith = x_all_CSP2_arith.reshape(-1, 1)

# Perform linear regression
slope_CSP2_arith, intercept_CSP2_arith, r_value_CSP2_arith, p_value_CSP2_arith, std_err_CSP2_arith = stats.linregress(x_all_CSP2_arith, y_all_CSP2_arith)

# Calculate the t-value for a 95% confidence interval
t_CSP2_arith = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP2_arith) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP2_arith = t_CSP2_arith * std_err_CSP2_arith
intercept_conf_interval_CSP2_arith = t_CSP2_arith * std_err_CSP2_arith * np.sqrt((1/len(x_all_CSP2_arith)) + (np.mean(x_all_CSP2_arith)**2 / np.sum((x_all_CSP2_arith - np.mean(x_all_CSP2_arith))**2)))

# Calculate R-squared value
r_squared_CSP2_arith = r_value_CSP2_arith**2

shift_CSP2_arith = 1 - intercept_CSP2_arith

plt.scatter(x19_CSP2_arith, y19_CSP2_arith + shift_CSP2_arith, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP2_arith, y20_CSP2_arith + shift_CSP2_arith, color = 'cyan', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP2_arith, y21_CSP2_arith + shift_CSP2_arith, color = 'darkgreen', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP2_arith, y22_CSP2_arith + shift_CSP2_arith, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP2_arith, y23_CSP2_arith + shift_CSP2_arith, color = 'darkorange', marker = 'D', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP2_arith, b_CSP2_arith = np.polyfit(x_CSP2_arith,y_CSP2_arith,1)
c_CSP2_arith, d_CSP2_arith = np.polyfit(x_CSP2_arith,y_baatz_CSP2_arith,1)
plt.scatter(x_CSP2_arith, y_CSP2_arith, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP2_arith, 0.005*x_CSP2_arith + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP2_arith, slope_CSP2_arith*x_all_CSP2_arith + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP2_arith, a_CSP2_arith*x_CSP2_arith+b_CSP2_arith, linestyle = '-', color = 'red', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP2_arith, round(slope_CSP2_arith,2)], [r2_CSP2_arith, round(r_squared_CSP2_arith, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP2_with_TDR_arith.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
##################################################################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2011_CSP3_1_arith = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2012_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2013_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2014_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2015_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2016_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2017_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2018_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2019_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2020_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP3_arith = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC11_PI_F_1_1_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_1_1']/1.42
SWC11_PI_F_1_2_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_2_1']/1.42
SWC11_PI_F_1_3_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_3_1']/1.42
SWC11_PI_F_1_4_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_5_1']/1.42
SWC11_PI_F_2_1_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_2_1_1']/1.42
SWC11_PI_F_2_2_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_2_2_1']/1.42
SWC11_PI_F_2_3_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_2_3_1']/1.42
SWC11_PI_F_2_4_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_2_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_5_1']/1.42
SWC11_PI_F_3_1_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_3_1_1']/1.42
SWC11_PI_F_3_2_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_3_2_1']/1.42
SWC11_PI_F_3_3_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_3_3_1']/1.42
SWC11_PI_F_3_4_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_3_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3_arith = GWC2011_CSP3_1_arith['SWC_PI_F_1_5_1']/1.42

SWC12_PI_F_1_1_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC12_PI_F_1_2_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC12_PI_F_1_3_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC12_PI_F_1_4_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC12_PI_F_2_1_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC12_PI_F_2_2_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC12_PI_F_2_3_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC12_PI_F_2_4_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC12_PI_F_3_1_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC12_PI_F_3_2_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC12_PI_F_3_3_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC12_PI_F_3_4_1_CSP3_arith = GWC2012_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC13_PI_F_1_1_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC13_PI_F_1_2_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC13_PI_F_1_3_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC13_PI_F_1_4_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC13_PI_F_2_1_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC13_PI_F_2_2_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC13_PI_F_2_3_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC13_PI_F_2_4_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC13_PI_F_3_1_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC13_PI_F_3_2_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC13_PI_F_3_3_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC13_PI_F_3_4_1_CSP3_arith = GWC2013_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC14_PI_F_1_1_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC14_PI_F_1_2_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC14_PI_F_1_3_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC14_PI_F_1_4_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC14_PI_F_2_1_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC14_PI_F_2_2_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC14_PI_F_2_3_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC14_PI_F_2_4_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC14_PI_F_3_1_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC14_PI_F_3_2_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC14_PI_F_3_3_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC14_PI_F_3_4_1_CSP3_arith = GWC2014_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC15_PI_F_1_1_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC15_PI_F_1_2_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC15_PI_F_1_3_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC15_PI_F_1_4_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC15_PI_F_2_1_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC15_PI_F_2_2_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC15_PI_F_2_3_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC15_PI_F_2_4_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC15_PI_F_3_1_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC15_PI_F_3_2_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC15_PI_F_3_3_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC15_PI_F_3_4_1_CSP3_arith = GWC2015_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC16_PI_F_1_1_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC16_PI_F_1_2_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC16_PI_F_1_3_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC16_PI_F_1_4_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC16_PI_F_2_1_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC16_PI_F_2_2_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC16_PI_F_2_3_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC16_PI_F_2_4_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC16_PI_F_3_1_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC16_PI_F_3_2_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC16_PI_F_3_3_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC16_PI_F_3_4_1_CSP3_arith = GWC2016_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC17_PI_F_1_1_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC17_PI_F_1_2_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC17_PI_F_1_3_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC17_PI_F_1_4_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC17_PI_F_2_1_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC17_PI_F_2_2_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC17_PI_F_2_3_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC17_PI_F_2_4_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC17_PI_F_3_1_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC17_PI_F_3_2_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC17_PI_F_3_3_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC17_PI_F_3_4_1_CSP3_arith = GWC2017_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC18_PI_F_1_1_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC18_PI_F_1_2_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC18_PI_F_1_3_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC18_PI_F_1_4_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC18_PI_F_2_1_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC18_PI_F_2_2_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC18_PI_F_2_3_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC18_PI_F_2_4_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC18_PI_F_3_1_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC18_PI_F_3_2_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC18_PI_F_3_3_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC18_PI_F_3_4_1_CSP3_arith = GWC2018_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC19_PI_F_1_1_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP3_arith = GWC2019_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC20_PI_F_1_1_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC20_PI_F_1_2_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC20_PI_F_1_3_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC20_PI_F_1_4_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC20_PI_F_2_1_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC20_PI_F_2_2_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC20_PI_F_2_3_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC20_PI_F_2_4_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC20_PI_F_3_1_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC20_PI_F_3_2_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC20_PI_F_3_3_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC20_PI_F_3_4_1_CSP3_arith = GWC2020_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP3_arith = GWC2021_CSP3_arith['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP3_arith = GWC2022_CSP3_arith['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP3_arith = GWC2023_CSP3_arith['SWC_F_3_4_1']/1.42

twosenwavgall2011_CSP3_arith = ((((SWC11_PI_F_1_1_1_CSP3_arith + SWC11_PI_F_2_1_1_CSP3_arith + SWC11_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC11_PI_F_1_2_1_CSP3_arith + SWC11_PI_F_2_2_1_CSP3_arith + SWC11_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2012_CSP3_arith = ((((SWC12_PI_F_1_1_1_CSP3_arith + SWC12_PI_F_2_1_1_CSP3_arith + SWC12_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC12_PI_F_1_2_1_CSP3_arith + SWC12_PI_F_2_2_1_CSP3_arith + SWC12_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2013_CSP3_arith = ((((SWC13_PI_F_1_1_1_CSP3_arith + SWC13_PI_F_2_1_1_CSP3_arith + SWC13_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC13_PI_F_1_2_1_CSP3_arith + SWC13_PI_F_2_2_1_CSP3_arith + SWC13_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2014_CSP3_arith = ((((SWC14_PI_F_1_1_1_CSP3_arith + SWC14_PI_F_2_1_1_CSP3_arith + SWC14_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC14_PI_F_1_2_1_CSP3_arith + SWC14_PI_F_2_2_1_CSP3_arith + SWC14_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2015_CSP3_arith = ((((SWC15_PI_F_1_1_1_CSP3_arith + SWC15_PI_F_2_1_1_CSP3_arith + SWC15_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC15_PI_F_1_2_1_CSP3_arith + SWC15_PI_F_2_2_1_CSP3_arith + SWC15_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2016_CSP3_arith = ((((SWC16_PI_F_1_1_1_CSP3_arith + SWC16_PI_F_2_1_1_CSP3_arith + SWC16_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC16_PI_F_1_2_1_CSP3_arith + SWC16_PI_F_2_2_1_CSP3_arith + SWC16_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2017_CSP3_arith = ((((SWC17_PI_F_1_1_1_CSP3_arith + SWC17_PI_F_2_1_1_CSP3_arith + SWC17_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC17_PI_F_1_2_1_CSP3_arith + SWC17_PI_F_2_2_1_CSP3_arith + SWC17_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2018_CSP3_arith = ((((SWC18_PI_F_1_1_1_CSP3_arith + SWC18_PI_F_2_1_1_CSP3_arith + SWC18_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC18_PI_F_1_2_1_CSP3_arith + SWC18_PI_F_2_2_1_CSP3_arith + SWC18_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2019_CSP3_arith = ((((SWC19_PI_F_1_1_1_CSP3_arith + SWC19_PI_F_2_1_1_CSP3_arith + SWC19_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP3_arith + SWC19_PI_F_2_2_1_CSP3_arith + SWC19_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2020_CSP3_arith = ((((SWC20_PI_F_1_1_1_CSP3_arith + SWC20_PI_F_2_1_1_CSP3_arith + SWC20_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC20_PI_F_1_2_1_CSP3_arith + SWC20_PI_F_2_2_1_CSP3_arith + SWC20_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2021_CSP3_arith = ((((SWC21_PI_F_1_1_1_CSP3_arith + SWC21_PI_F_2_1_1_CSP3_arith + SWC21_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP3_arith + SWC21_PI_F_2_2_1_CSP3_arith + SWC21_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2022_CSP3_arith = ((((SWC22_PI_F_1_1_1_CSP3_arith + SWC22_PI_F_2_1_1_CSP3_arith + SWC22_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP3_arith + SWC22_PI_F_2_2_1_CSP3_arith + SWC22_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100
twosenwavgall2023_CSP3_arith = ((((SWC23_PI_F_1_1_1_CSP3_arith + SWC23_PI_F_2_1_1_CSP3_arith + SWC23_PI_F_3_1_1_CSP3_arith)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP3_arith + SWC23_PI_F_2_2_1_CSP3_arith + SWC23_PI_F_3_2_1_CSP3_arith)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2011_CSP3_arith = twosenwavgall2011_CSP3_arith.resample('D').mean() 
dailywavgSWC2012_CSP3_arith = twosenwavgall2012_CSP3_arith.resample('D').mean() 
dailywavgSWC2013_CSP3_arith = twosenwavgall2013_CSP3_arith.resample('D').mean() 
dailywavgSWC2014_CSP3_arith = twosenwavgall2014_CSP3_arith.resample('D').mean()
dailywavgSWC2015_CSP3_arith = twosenwavgall2015_CSP3_arith.resample('D').mean() 
dailywavgSWC2016_CSP3_arith = twosenwavgall2016_CSP3_arith.resample('D').mean() 
dailywavgSWC2017_CSP3_arith = twosenwavgall2017_CSP3_arith.resample('D').mean() 
dailywavgSWC2018_CSP3_arith = twosenwavgall2018_CSP3_arith.resample('D').mean()
dailywavgSWC2019_CSP3_arith = twosenwavgall2019_CSP3_arith.resample('D').mean() 
dailywavgSWC2020_CSP3_arith = twosenwavgall2020_CSP3_arith.resample('D').mean() 
dailywavgSWC2021_CSP3_arith = twosenwavgall2021_CSP3_arith.resample('D').mean() 
dailywavgSWC2022_CSP3_arith = twosenwavgall2022_CSP3_arith.resample('D').mean() 
dailywavgSWC2023_CSP3_arith = twosenwavgall2023_CSP3_arith.resample('D').mean()

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2011_CSP3_arith = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2012_CSP3_arith = pd.read_csv('YOUR PATH/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2013_CSP3_arith = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2014_CSP3_arith = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2015_CSP3_arith = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2016_CSP3_arith = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2017_CSP3_arith = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2018_CSP3_arith = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2019_CSP3_arith = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2020_CSP3_arith = pd.read_csv('YOUR PATH/CSP32020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP3_arith = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP3_arith = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP3_arith = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2011_CSP3_arith = NPIV2011_CSP3_arith['NPIV']
NPIVCount2012_CSP3_arith = NPIV2012_CSP3_arith['NPIV']
NPIVCount2013_CSP3_arith = NPIV2013_CSP3_arith['NPIV']
NPIVCount2014_CSP3_arith = NPIV2014_CSP3_arith['NPIV']
NPIVCount2015_CSP3_arith = NPIV2015_CSP3_arith['NPIV']
NPIVCount2016_CSP3_arith = NPIV2016_CSP3_arith['NPIV']
NPIVCount2017_CSP3_arith = NPIV2017_CSP3_arith['NPIV']
NPIVCount2018_CSP3_arith = NPIV2018_CSP3_arith['NPIV']
NPIVCount2019_CSP3_arith = NPIV2019_CSP3_arith['NPIV']
NPIVCount2020_CSP3_arith = NPIV2020_CSP3_arith['NPIV']
NPIVCount2021_CSP3_arith = NPIV2021_CSP3_arith['NPIV']
NPIVCount2022_CSP3_arith = NPIV2022_CSP3_arith['NPIV']
NPIVCount2023_CSP3_arith = NPIV2023_CSP3_arith['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2011_CSP3_arith = NPIVCount2011_CSP3_arith.resample('D').mean()
dailyNPIV2012_CSP3_arith = NPIVCount2012_CSP3_arith.resample('D').mean()
dailyNPIV2013_CSP3_arith = NPIVCount2013_CSP3_arith.resample('D').mean()
dailyNPIV2014_CSP3_arith = NPIVCount2014_CSP3_arith.resample('D').mean()
dailyNPIV2015_CSP3_arith = NPIVCount2015_CSP3_arith.resample('D').mean()
dailyNPIV2016_CSP3_arith = NPIVCount2016_CSP3_arith.resample('D').mean()
dailyNPIV2017_CSP3_arith = NPIVCount2017_CSP3_arith.resample('D').mean()
dailyNPIV2018_CSP3_arith = NPIVCount2018_CSP3_arith.resample('D').mean()
dailyNPIV2019_CSP3_arith = NPIVCount2019_CSP3_arith.resample('D').mean()
dailyNPIV2020_CSP3_arith = NPIVCount2020_CSP3_arith.resample('D').mean()
dailyNPIV2021_CSP3_arith = NPIVCount2021_CSP3_arith.resample('D').mean()
dailyNPIV2022_CSP3_arith = NPIVCount2022_CSP3_arith.resample('D').mean()
dailyNPIV2023_CSP3_arith = NPIVCount2023_CSP3_arith.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc11_CSP3_arith = dailywavgSWC2011_CSP3
pwc12_CSP3_arith = dailywavgSWC2012_CSP3
pwc13_CSP3_arith = dailywavgSWC2013_CSP3
pwc14_CSP3_arith = dailywavgSWC2014_CSP3
pwc15_CSP3_arith = dailywavgSWC2015_CSP3
pwc16_CSP3_arith = dailywavgSWC2016_CSP3
pwc17_CSP3_arith = dailywavgSWC2017_CSP3
pwc18_CSP3_arith = dailywavgSWC2018_CSP3
pwc19_CSP3_arith = dailywavgSWC2019_CSP3
pwc20_CSP3_arith = dailywavgSWC2020_CSP3
pwc21_CSP3_arith = dailywavgSWC2021_CSP3
pwc22_CSP3_arith = dailywavgSWC2022_CSP3
pwc23_CSP3_arith = dailywavgSWC2023_CSP3

N11_CSP3_arith = dailyNPIV2011_CSP3
N12_CSP3_arith = dailyNPIV2012_CSP3
N13_CSP3_arith = dailyNPIV2013_CSP3
N14_CSP3_arith = dailyNPIV2014_CSP3
N15_CSP3_arith = dailyNPIV2015_CSP3
N16_CSP3_arith = dailyNPIV2016_CSP3
N17_CSP3_arith = dailyNPIV2017_CSP3
N18_CSP3_arith = dailyNPIV2018_CSP3
N19_CSP3_arith = dailyNPIV2019_CSP3
N20_CSP3_arith = dailyNPIV2020_CSP3
N21_CSP3_arith = dailyNPIV2021_CSP3
N22_CSP3_arith = dailyNPIV2022_CSP3
N23_CSP3_arith = dailyNPIV2023_CSP3

N0_for_calc_3_arith = 1601.43
lwc = 0.058
soc = 0.0092

#Calculate N
N0_CSP3_11_arith = N0_for_calc_3_arith*(((0.0808)/((pwc11_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_12_arith = N0_for_calc_3_arith*(((0.0808)/((pwc12_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_13_arith = N0_for_calc_3_arith*(((0.0808)/((pwc13_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_14_arith = N0_for_calc_3_arith*(((0.0808)/((pwc14_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_15_arith = N0_for_calc_3_arith*(((0.0808)/((pwc15_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_16_arith = N0_for_calc_3_arith*(((0.0808)/((pwc16_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_17_arith = N0_for_calc_3_arith*(((0.0808)/((pwc17_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_18_arith = N0_for_calc_3_arith*(((0.0808)/((pwc18_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_19_arith = N0_for_calc_3_arith*(((0.0808)/((pwc19_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_20_arith = N0_for_calc_3_arith*(((0.0808)/((pwc20_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_21_arith = N0_for_calc_3_arith*(((0.0808)/((pwc21_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_22_arith = N0_for_calc_3_arith*(((0.0808)/((pwc22_CSP3_arith+lwc+soc)+0.115))+0.372)
N0_CSP3_23_arith = N0_for_calc_3_arith*(((0.0808)/((pwc23_CSP3_arith+lwc+soc)+0.115))+0.372)

N0interp11_CSP3_arith = N0_CSP3_11_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp12_CSP3_arith = N0_CSP3_12_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp13_CSP3_arith = N0_CSP3_13_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp14_CSP3_arith = N0_CSP3_14_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp15_CSP3_arith = N0_CSP3_15_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp16_CSP3_arith = N0_CSP3_16_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp17_CSP3_arith = N0_CSP3_17_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp18_CSP3_arith = N0_CSP3_18_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp19_CSP3_arith = N0_CSP3_19_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP3_arith = N0_CSP3_20_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP3_arith = N0_CSP3_21_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP3_arith = N0_CSP3_22_arith.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP3_arith = N0_CSP3_23_arith.interpolate(method = 'linear', limit_direction = 'forward')

NCSP3_11_arith= N0interp11_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_11_arith.csv')
NCSP3_12_arith= N0interp12_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_12_arith.csv')
NCSP3_13_arith= N0interp13_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_13_arith.csv')
NCSP3_14_arith= N0interp14_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_14_arith.csv')
NCSP3_15_arith= N0interp15_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_15_arith.csv')
NCSP3_16_arith= N0interp16_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_16_arith.csv')
NCSP3_17_arith= N0interp17_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_17_arith.csv')
NCSP3_18_arith= N0interp18_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_18_arith.csv')
NCSP3_19_arith= N0interp19_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_19_arith.csv')
NCSP3_20_arith= N0interp20_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_20_arith.csv')
NCSP3_21_arith= N0interp21_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_21_arith.csv')
NCSP3_22_arith= N0interp22_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_22_arith.csv')
NCSP3_23_arith= N0interp23_CSP3_arith.to_csv('YOUR PATH TO MAKE/N_CSP3_23_arith.csv')

CSP3_11_arith = pd.read_csv('YOUR PATH/N_CSP3_11_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_12_arith = pd.read_csv('YOUR PATH/N_CSP3_12_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_13_arith = pd.read_csv('YOUR PATH/N_CSP3_13_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_14_arith = pd.read_csv('YOUR PATH/N_CSP3_14_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_15_arith = pd.read_csv('YOUR PATH/N_CSP3_15_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_16_arith = pd.read_csv('YOUR PATH/N_CSP3_16_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_17_arith = pd.read_csv('YOUR PATH/N_CSP3_17_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_18_arith = pd.read_csv('YOUR PATH/N_CSP3_18_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_19_arith = pd.read_csv('YOUR PATH/N_CSP3_19_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_20_arith = pd.read_csv('YOUR PATH/N_CSP3_20_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_21_arith = pd.read_csv('YOUR PATH/N_CSP3_21_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_22_arith = pd.read_csv('YOUR PATH/N_CSP3_22_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_23_arith = pd.read_csv('YOUR PATH/N_CSP3_23_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP3_11_1_arith = CSP3_11_arith['0']
N_CSP3_12_1_arith = CSP3_12_arith['0']
N_CSP3_13_1_arith = CSP3_13_arith['0']
N_CSP3_14_1_arith = CSP3_14_arith['0']
N_CSP3_15_1_arith = CSP3_15_arith['0']
N_CSP3_16_1_arith = CSP3_16_arith['0']
N_CSP3_17_1_arith = CSP3_17_arith['0']
N_CSP3_18_1_arith = CSP3_18_arith['0']
N_CSP3_19_1_arith = CSP3_19_arith['0']
N_CSP3_20_1_arith = CSP3_20_arith['0']
N_CSP3_21_1_arith = CSP3_21_arith['0']
N_CSP3_22_1_arith = CSP3_22_arith['0']
N_CSP3_23_1_arith = CSP3_23_arith['0']

fveg_CSP3_11_arith = dailyNPIV2011_CSP3_arith / N0_CSP3_11_arith
fveg_CSP3_12_arith = dailyNPIV2012_CSP3_arith / N0_CSP3_12_arith
fveg_CSP3_13_arith = dailyNPIV2013_CSP3_arith / N0_CSP3_13_arith
fveg_CSP3_14_arith = dailyNPIV2014_CSP3_arith / N0_CSP3_14_arith
fveg_CSP3_15_1_arith = dailyNPIV2015_CSP3_arith / N0_CSP3_15_arith
fveg_CSP3_16_arith = dailyNPIV2016_CSP3_arith / N0_CSP3_16_arith
fveg_CSP3_17_arith = dailyNPIV2017_CSP3_arith / N0_CSP3_17_arith
fveg_CSP3_18_arith = dailyNPIV2018_CSP3_arith / N0_CSP3_18_arith
fveg_CSP3_19_arith = dailyNPIV2019_CSP3_arith / N0_CSP3_19_arith
fveg_CSP3_20_1_arith = dailyNPIV2020_CSP3_arith / N0_CSP3_20_arith
fveg_CSP3_21_arith = dailyNPIV2021_CSP3_arith / N0_CSP3_21_arith
fveg_CSP3_22_1_arith = dailyNPIV2022_CSP3_arith / N0_CSP3_22_arith
fveg_CSP3_23_1_arith = dailyNPIV2023_CSP3_arith / N0_CSP3_23_arith

fveg_CSP3_11_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_11_arith.csv')
fveg_CSP3_12_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_12_arith.csv')
fveg_CSP3_13_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_13_arith.csv')
fveg_CSP3_14_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_14_arith.csv')
fveg_CSP3_15_1_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_15_1_arith.csv')
fveg_CSP3_16_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_16_arith.csv')
fveg_CSP3_17_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_17_arith.csv')
fveg_CSP3_18_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_18_arith.csv')
fveg_CSP3_19_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_19_arith.csv')
fveg_CSP3_20_1_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_20_1_arith.csv')
fveg_CSP3_21_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_21_arith.csv')
fveg_CSP3_22_1_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_22_1_arith.csv')
fveg_CSP3_23_1_arith.to_csv('YOUR PATH TO MAKE/fveg_CSP3_23_1_arith.csv')

fveg_CSP3_11_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_11_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_12_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_12_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_13_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_13_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_14_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_14_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_15_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_15_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_16_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_16_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_17_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_17_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_18_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_18_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_19_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_19_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_20_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_20_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_21_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_21_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_22_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_22_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_23_1_arith = pd.read_csv('YOUR PATH/fveg_CSP3_23_arith.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP3_11_1_arith = fveg_CSP3_11_1_arith['0']
fveg_CSP3_12_1_arith = fveg_CSP3_12_1_arith['0']
fveg_CSP3_13_1_arith = fveg_CSP3_13_1_arith['0']
fveg_CSP3_14_1_arith = fveg_CSP3_14_1_arith['0']
fveg_CSP3_15_1_arith = fveg_CSP3_15_1_arith['0']
fveg_CSP3_16_1_arith = fveg_CSP3_16_1_arith['0']
fveg_CSP3_17_1_arith = fveg_CSP3_17_1_arith['0']
fveg_CSP3_18_1_arith = fveg_CSP3_18_1_arith['0']
fveg_CSP3_19_1_arith = fveg_CSP3_19_1_arith['0']
fveg_CSP3_20_1_arith = fveg_CSP3_20_1_arith['0']
fveg_CSP3_21_1_arith = fveg_CSP3_21_1_arith['0']
fveg_CSP3_22_1_arith = fveg_CSP3_22_1_arith['0']
fveg_CSP3_23_1_arith = fveg_CSP3_23_1_arith['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2011_CSP3_arith = pd.read_csv('YOUR PATH/CSP32011BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2012_CSP3_arith = pd.read_csv('YOUR PATH/CSP32012BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2013_CSP3_arith = pd.read_csv('YOUR PATH/CSP32013BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2014_CSP3_arith = pd.read_csv('YOUR PATH/CSP32014BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2015_CSP3_arith = pd.read_csv('YOUR PATH/CSP32015BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2016_CSP3_arith = pd.read_csv('YOUR PATH/CSP32016BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2017_CSP3_arith = pd.read_csv('YOUR PATH/CSP32017BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2018_CSP3_arith = pd.read_csv('YOUR PATH/CSP32018BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2019_CSP3_arith = pd.read_csv('YOUR PATH/CSP32019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP3_arith = pd.read_csv('YOUR PATH/CSP32020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP3_arith = pd.read_csv('YOUR PATH/CSP32021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP3_arith = pd.read_csv('YOUR PATH/CSP32022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP3_arith = pd.read_csv('YOUR PATH/CSP32023BWE_plotdate.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE11_CSP3_arith = BWEcsv2011_CSP3_arith['BWE']
BWE12_CSP3_arith = BWEcsv2012_CSP3_arith['BWE']
BWE13_CSP3_arith = BWEcsv2013_CSP3_arith['BWE']
BWE14_CSP3_arith = BWEcsv2014_CSP3_arith['BWE']
BWE15_CSP3_arith = BWEcsv2015_CSP3_arith['BWE']
BWE16_CSP3_arith = BWEcsv2016_CSP3_arith['BWE']
BWE17_CSP3_arith = BWEcsv2017_CSP3_arith['BWE']
BWE18_CSP3_arith = BWEcsv2018_CSP3_arith['BWE']
BWE19_CSP3_arith = BWEcsv2019_CSP3_arith['BWE']
BWE20_CSP3_arith = BWEcsv2020_CSP3_arith['BWE']
BWE21_CSP3_arith = BWEcsv2021_CSP3_arith['BWE']
BWE22_CSP3_arith = BWEcsv2022_CSP3_arith['BWE']
BWE23_CSP3_arith = BWEcsv2023_CSP3_arith['BWE']

#Interpolate between dates
BWEinterp11_CSP3_arith = BWE11_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp12_CSP3_arith = BWE12_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp13_CSP3_arith = BWE13_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp14_CSP3_arith = BWE14_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp15_CSP3_arith = BWE15_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp16_CSP3_arith = BWE16_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp17_CSP3_arith = BWE17_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp18_CSP3_arith = BWE18_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp19_CSP3_arith = BWE19_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP3_arith = BWE20_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP3_arith = BWE21_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP3_arith = BWE22_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP3_arith = BWE23_CSP3_arith.interpolate(method = 'linear', limit_direction = 'forward')

BWEFillNaN11_CSP3_arith = BWEinterp11_CSP3_arith.fillna(int(BWEinterp11_CSP3_arith.mean()), inplace=True)
BWEFillNaN12_CSP3_arith = BWEinterp12_CSP3_arith.fillna(int(BWEinterp12_CSP3_arith.mean()), inplace=True)
BWEFillNaN13_CSP3_arith = BWEinterp13_CSP3_arith.fillna(int(BWEinterp13_CSP3_arith.mean()), inplace=True)
BWEFillNaN14_CSP3_arith = BWEinterp14_CSP3_arith.fillna(int(BWEinterp14_CSP3_arith.mean()), inplace=True)
BWEFillNaN15_CSP3_arith = BWEinterp15_CSP3_arith.fillna(int(BWEinterp15_CSP3_arith.mean()), inplace=True)
BWEFillNaN16_CSP3_arith = BWEinterp16_CSP3_arith.fillna(int(BWEinterp16_CSP3_arith.mean()), inplace=True)
BWEFillNaN17_CSP3_arith = BWEinterp17_CSP3_arith.fillna(int(BWEinterp17_CSP3_arith.mean()), inplace=True)
BWEFillNaN18_CSP3_arith = BWEinterp18_CSP3_arith.fillna(int(BWEinterp18_CSP3_arith.mean()), inplace=True)
BWEFillNaN19_CSP3_arith = BWEinterp19_CSP3_arith.fillna(int(BWEinterp19_CSP3_arith.mean()), inplace=True)
BWEFillNaN20_CSP3_arith = BWEinterp20_CSP3_arith.fillna(int(BWEinterp20_CSP3_arith.mean()), inplace=True)
BWEFillNaN21_CSP3_arith = BWEinterp21_CSP3_arith.fillna(int(BWEinterp21_CSP3_arith.mean()), inplace=True)
BWEFillNaN22_CSP3_arith = BWEinterp22_CSP3_arith.fillna(int(BWEinterp22_CSP3_arith.mean()), inplace=True)
BWEFillNaN23_CSP3_arith = BWEinterp23_CSP3_arith.fillna(int(BWEinterp23_CSP3_arith.mean()), inplace=True)

fveg_CSP3_11_1_arith = fveg_CSP3_11_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_12_1_arith = fveg_CSP3_12_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_13_1_arith = fveg_CSP3_13_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_14_1_arith = fveg_CSP3_14_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_15_1_arith = fveg_CSP3_15_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_16_1_arith = fveg_CSP3_16_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_17_1_arith = fveg_CSP3_17_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_18_1_arith = fveg_CSP3_18_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_19_1_arith = fveg_CSP3_19_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_20_1_arith = fveg_CSP3_20_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_21_1_arith = fveg_CSP3_21_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_22_1_arith = fveg_CSP3_22_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_23_1_arith = fveg_CSP3_23_1_arith.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x11_CSP3_arith = BWEinterp11_CSP3_arith
y11_CSP3_arith = 1/fveg_CSP3_11_1_arith
x12_CSP3_arith = BWEinterp12_CSP3_arith
y12_CSP3_arith = 1/fveg_CSP3_12_1_arith
x13_CSP3_arith = BWEinterp13_CSP3_arith
y13_CSP3_arith = 1/fveg_CSP3_13_1_arith
x14_CSP3_arith = BWEinterp14_CSP3_arith
y14_CSP3_arith = 1/fveg_CSP3_14_1_arith
x15_CSP3_arith = BWEinterp15_CSP3_arith
y15_CSP3_arith = 1/fveg_CSP3_15_1_arith
x16_CSP3_arith = BWEinterp16_CSP3_arith
y16_CSP3_arith = 1/fveg_CSP3_16_1_arith
x17_CSP3_arith = BWEinterp17_CSP3_arith
y17_CSP3_arith = 1/fveg_CSP3_17_1_arith
x18_CSP3_arith = BWEinterp18_CSP3_arith
y18_CSP3_arith = 1/fveg_CSP3_18_1_arith
x19_CSP3_arith = BWEinterp19_CSP3_arith
y19_CSP3_arith = 1/fveg_CSP3_19_1_arith
x20_CSP3_arith = BWEinterp20_CSP3_arith
y20_CSP3_arith = 1/fveg_CSP3_20_1_arith
x21_CSP3_arith = BWEinterp21_CSP3_arith
y21_CSP3_arith = 1/fveg_CSP3_21_1_arith
x22_CSP3_arith = BWEinterp22_CSP3_arith
y22_CSP3_arith = 1/fveg_CSP3_22_1_arith
x23_CSP3_arith = BWEinterp23_CSP3_arith
y23_CSP3_arith = 1/fveg_CSP3_23_1_arith

y15_CSP3_arith = fill_nan_with_avg(y15_CSP3_arith)
y20_CSP3_arith = fill_nan_with_avg(y20_CSP3_arith)
y22_CSP3_arith = fill_nan_with_avg(y22_CSP3_arith)
y23_CSP3_arith = fill_nan_with_avg(y23_CSP3_arith)

# Concatenate all x and y values
x_all_CSP3_arith = np.concatenate([x11_CSP3_arith, x12_CSP3_arith, x13_CSP3_arith, x14_CSP3_arith, x15_CSP3_arith, x16_CSP3_arith, x17_CSP3_arith, x18_CSP3_arith, x19_CSP3_arith, x20_CSP3_arith, x21_CSP3_arith, x22_CSP3_arith, x23_CSP3_arith])
y_all_CSP3_arith = np.concatenate([y11_CSP3_arith, y12_CSP3_arith, y13_CSP3_arith, y14_CSP3_arith, y15_CSP3_arith, y16_CSP3_arith, y17_CSP3_arith, y18_CSP3_arith, y19_CSP3_arith, y20_CSP3_arith, y21_CSP3_arith, y22_CSP3_arith, y23_CSP3_arith])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP3_arith = x_all_CSP3_arith.reshape(-1, 1)

# Perform linear regression
slope_CSP3_arith, intercept_CSP3_arith, r_value_CSP3_arith, p_value_CSP3_arith, std_err_CSP3_arith = stats.linregress(x_all_CSP3_arith, y_all_CSP3_arith)

# Calculate the t-value for a 95% confidence interval
t_CSP3_arith = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP3_arith) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP3_arith = t_CSP3_arith * std_err_CSP3_arith
intercept_conf_interval_CSP3_arith = t_CSP3_arith * std_err_CSP3_arith * np.sqrt((1/len(x_all_CSP3_arith)) + (np.mean(x_all_CSP3_arith)**2 / np.sum((x_all_CSP3_arith - np.mean(x_all_CSP3_arith))**2)))

# Calculate R-squared value
r_squared_CSP3_arith = r_value_CSP3_arith**2

shift_CSP3_arith = 1 - intercept_CSP3_arith

plt.scatter(x11_CSP3_arith, y11_CSP3_arith + shift_CSP3_arith, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.3)
plt.scatter(x12_CSP3_arith, y12_CSP3_arith + shift_CSP3_arith, color = 'indianred', marker = 'v', label = '2012', alpha=0.3)
plt.scatter(x13_CSP3_arith, y13_CSP3_arith + shift_CSP3_arith, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.3)
plt.scatter(x14_CSP3_arith, y14_CSP3_arith + shift_CSP3_arith, color = 'peru', marker = 's', label = '2014', alpha=0.3)
plt.scatter(x15_CSP3_arith, y15_CSP3_arith + shift_CSP3_arith, color = 'gold', marker = '<', label = '2015', alpha=0.3)
plt.scatter(x16_CSP3_arith, y16_CSP3_arith + shift_CSP3_arith, color = 'pink', marker = '>', label = '2016', alpha=0.3)
plt.scatter(x17_CSP3_arith, y17_CSP3_arith + shift_CSP3_arith, color = 'rosybrown', marker = '.', label = '2017', alpha=0.3)
plt.scatter(x18_CSP3_arith, y18_CSP3_arith + shift_CSP3_arith, color = 'fuchsia', marker = '1', label = '2018', alpha=0.3)
plt.scatter(x19_CSP3_arith, y19_CSP3_arith + shift_CSP3_arith, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP3_arith, y20_CSP3_arith + shift_CSP3_arith, color = 'aquamarine', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP3_arith, y21_CSP3_arith + shift_CSP3_arith, color = 'greenyellow', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP3_arith, y22_CSP3_arith + shift_CSP3_arith, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP3_arith, y23_CSP3_arith + shift_CSP3_arith, color = 'darkorange', marker = 'p', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP3_arith, b_CSP3_arith = np.polyfit(x_CSP3_arith,y_CSP3_arith,1)
c_CSP3_arith, d_CSP3_arith = np.polyfit(x_CSP3_arith,y_baatz_CSP3_arith,1)
plt.scatter(x_CSP3_arith, y_CSP3_arith, marker = 'o', color = 'red', label = r'$f_b$ Grav. 2023')
plt.plot(x_CSP3_arith, a_CSP3_arith*x_CSP3_arith+b_CSP3_arith, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP3_arith, 0.005*x_CSP3_arith + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP3_arith, slope_CSP3_arith*x_all_CSP3_arith + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 Arithmetic Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3_arith, round(slope_CSP3_arith,2)], [r2_CSP3_arith, round(r_squared_CSP3_arith, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_with_TDR_arithmetic.png', dpi=300, bbox_inches='tight')
plt.show()


# Normalize according to Baatz 2015
normal_CSP1_all_arith = (26.14/2893.9)*100
normal_CSP2_all_arith = (32.47/2949.04)*100
normal_CSP3_all_arith = (9.45/1621.86)*100


data_arith_1 = {
    '': ['Slope', 'Intercept', 'Baatz 2015\nSlope', 'Baatz 2015\nIntercept', 'Slope\nDifference', 'Slope\nRatio', 'Neutron\nIntensity\nReduction(%)'],
    'CSP1': [f"{r_mm_CSP1_arith}", f"{r_nn_CSP1_arith}", f"{r_mm_CSP1_baatz_arith}", f"{r_nn_CSP1_baatz_arith}", f"{round((r_mm_CSP1-r_mm_CSP1_baatz_arith),4)}", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith),4)}", f"{round((normal_CSP1_arith),2)}"],
    'CSP2': [f"{r_mm_CSP2_arith}", f"{r_nn_CSP2_arith}", f"{r_mm_CSP2_baatz_arith}", f"{r_nn_CSP2_baatz_arith}", f"{round((r_mm_CSP2-r_mm_CSP2_baatz_arith),4)}", f"{round((r_mm_CSP2_arith/r_mm_CSP2_baatz_arith),4)}", f"{round((normal_CSP2_arith),2)}"],
    'CSP3': [f"{r_mm_CSP3_arith}", f"{r_nn_CSP3_arith}", f"{r_mm_CSP3_baatz_arith}", f"{r_nn_CSP3_baatz_arith}", f"{round((r_mm_CSP3-r_mm_CSP3_baatz_arith),4)}", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith),4)}", f"{round((normal_CSP3_arith),2)}"],
    'CSP1\nTDR': [f"{round(val, 4)}" for val in [slope_CSP1_arith, intercept_CSP1_arith, r_mm_CSP1_baatz_arith, r_nn_CSP1_baatz_arith, slope_CSP1_arith-r_mm_CSP1_baatz_arith, slope_CSP1_arith/r_mm_CSP1_baatz_arith]] + [f"{round(normal_CSP1_all_arith, 2)}"],
    'CSP2\nTDR': [f"{round(val, 4)}" for val in [slope_CSP2_arith, intercept_CSP2_arith, r_mm_CSP2_baatz_arith, r_nn_CSP2_baatz_arith, slope_CSP2_arith-r_mm_CSP2_baatz_arith, slope_CSP2_arith/r_mm_CSP2_baatz_arith]] + [f"{round(normal_CSP2_all_arith, 2)}"],
    'CSP3\nTDR': [f"{round(val, 4)}" for val in [slope_CSP3_arith, intercept_CSP3_arith, r_mm_CSP3_baatz_arith, r_nn_CSP3_baatz_arith, slope_CSP3_arith-r_mm_CSP3_baatz_arith, slope_CSP3_arith/r_mm_CSP3_baatz_arith]] + [f"{round(normal_CSP3_all_arith, 2)}"],
    'CRS-2000/B\nAverage': [f"{round(val, 4)}" for val in [(r_mm_CSP1_arith+r_mm_CSP2_arith+slope_CSP1_arith+slope_CSP2_arith)/4, (r_nn_CSP1_arith+r_nn_CSP2_arith+intercept_CSP1_arith+intercept_CSP2_arith)/4, (r_mm_CSP1_baatz_arith+r_mm_CSP2_baatz_arith)/2, (r_nn_CSP1_baatz_arith+r_nn_CSP2_baatz_arith)/2, ((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith-r_mm_CSP2_baatz_arith)+(slope_CSP1_arith-r_mm_CSP1_baatz_arith)+(slope_CSP2_arith-r_mm_CSP2_baatz_arith))/4, ((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith/r_mm_CSP2_baatz_arith)+(slope_CSP1_arith/r_mm_CSP1_baatz_arith)+(slope_CSP2_arith/r_mm_CSP2_baatz_arith))/4]] + [f"{round((normal_CSP1_arith+normal_CSP2_arith+normal_CSP1_all_arith+normal_CSP2_all_arith)/4, 2)}"],
    'CRS-1000/B\nAverage': [f"{round(val, 4)}" for val in [(slope_CSP3_arith+r_mm_CSP3_arith)/2, (intercept_CSP3_arith+r_nn_CSP3_arith)/2, r_mm_CSP3_baatz_arith, r_nn_CSP3_baatz_arith, ((slope_CSP3_arith-r_mm_CSP3_baatz_arith)+(r_mm_CSP3_arith-r_mm_CSP3_baatz_arith))/2, ((slope_CSP3_arith/r_mm_CSP3_baatz_arith)+(r_mm_CSP3_arith/r_mm_CSP3_baatz_arith))/2]] + [f"{round((normal_CSP3_all_arith+normal_CSP3_arith)/2, 2)}"],
    'Gravimetric\nAverage': [f"{round(val, 4)}" for val in [(r_mm_CSP1_arith+r_mm_CSP2_arith+r_mm_CSP3_arith)/3, (r_nn_CSP1_arith+r_nn_CSP2_arith+r_nn_CSP3_arith)/3, (r_mm_CSP1_baatz_arith+r_mm_CSP2_baatz_arith+r_mm_CSP3_baatz_arith)/3, (r_nn_CSP1_baatz_arith+r_nn_CSP2_baatz_arith+r_nn_CSP3_baatz_arith)/3, ((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith-r_mm_CSP2_baatz_arith)+(r_mm_CSP3_arith-r_mm_CSP3_baatz_arith))/3, ((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith/r_mm_CSP2_baatz_arith)+(r_mm_CSP3_arith/r_mm_CSP3_baatz_arith))/3]] + [f"{round((normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith)/3, 2)}"], 
    'TDR\nAverage': [f"{round(val, 4)}" for val in [(slope_CSP1_arith+slope_CSP2_arith+slope_CSP3_arith)/3, (intercept_CSP1_arith+intercept_CSP2_arith+intercept_CSP3_arith)/3, (r_mm_CSP1_baatz_arith+r_mm_CSP2_baatz_arith+r_mm_CSP3_baatz_arith)/3, (r_nn_CSP1_baatz_arith+r_nn_CSP2_baatz_arith+r_nn_CSP3_baatz_arith)/3, ((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith-r_mm_CSP2_baatz_arith)+(r_mm_CSP3_arith-r_mm_CSP3_baatz_arith))/3, ((slope_CSP1_arith/r_mm_CSP1_baatz_arith)+(slope_CSP2_arith/r_mm_CSP2_baatz_arith)+(slope_CSP3_arith/r_mm_CSP3_baatz_arith))/3]] + [f"{round((normal_CSP1_all_arith+normal_CSP2_all_arith+normal_CSP3_all_arith)/3, 2)}"],
    'Overall\nAverage': [f"{round((val1 + val2) / 2, 4)}" for val1, val2 in zip(
    [(r_mm_CSP1_arith+r_mm_CSP2_arith+r_mm_CSP3_arith)/3, (r_nn_CSP1_arith+r_nn_CSP2_arith+r_nn_CSP3_arith)/3, (r_mm_CSP1_baatz_arith+r_mm_CSP2_baatz_arith+r_mm_CSP3_baatz_arith)/3, (r_nn_CSP1_baatz_arith+r_nn_CSP2_baatz_arith+r_nn_CSP3_baatz_arith)/3, ((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith-r_mm_CSP2_baatz_arith)+(r_mm_CSP3_arith-r_mm_CSP3_baatz_arith))/3, ((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith/r_mm_CSP2_baatz_arith)+(r_mm_CSP3_arith/r_mm_CSP3_baatz_arith))/3, (normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith)/3],
    [(slope_CSP1_arith+slope_CSP2_arith+slope_CSP3_arith)/3, (intercept_CSP1_arith+intercept_CSP2_arith+intercept_CSP3_arith)/3, (r_mm_CSP1_baatz_arith+r_mm_CSP2_baatz_arith+r_mm_CSP3_baatz_arith)/3, (r_nn_CSP1_baatz_arith+r_nn_CSP2_baatz_arith+r_nn_CSP3_baatz_arith)/3, ((r_mm_CSP1_arith-r_mm_CSP1_baatz_arith)+(r_mm_CSP2_arith-r_mm_CSP2_baatz_arith)+(r_mm_CSP3_arith-r_mm_CSP3_baatz_arith))/3, ((slope_CSP1_arith/r_mm_CSP1_baatz_arith)+(slope_CSP2_arith/r_mm_CSP2_baatz_arith)+(slope_CSP3_arith/r_mm_CSP3_baatz_arith))/3, (normal_CSP1_all_arith+normal_CSP2_all_arith+normal_CSP3_all_arith)/3]
)]
}
# Convert the dictionary to a pandas DataFrame
df_arith_1 = pd.DataFrame(data_arith_1)

# Function to render DataFrame as a table
def render_mpl_table(data_arith_1, col_width=6.0, row_height=0.7, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_arith_1.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(22,8))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_arith_1.values, bbox=bbox, colLabels=data_arith_1.columns, **kwargs)

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

render_mpl_table(df_arith_1, header_columns=0, col_width=6.0)

# Save the table as an image
plt.savefig('YOUR PATH/Summary_Table_with_TDR_arith.png', dpi=300, bbox_inches='tight')
plt.show()

data_1_arith = {
    '': ['CSP1', 'CSP2', 'CSP3', 'CSP1\nTDR', 'CSP2\nTDR', 'CSP3\nTDR', 'CRS-2000/B\nAverage', 'CRS-1000/B\nAverage', 'Gravimetric\nAverage', 'TDR\nAverage', 'Overall\nAverage'],
    'Neutron Intensity\nReduction\n(% per mm BWE)': [f"{round((normal_CSP1_arith),3)}", f"{round((normal_CSP2_arith),3)}", f"{round((normal_CSP3_arith),3)}", f"{round((normal_CSP1_all),3)}", f"{round(normal_CSP2_all, 3)}", f"{round(normal_CSP3_all, 3)}", f"{round((normal_CSP1_arith+normal_CSP2_arith+normal_CSP1_all+normal_CSP2_all)/4, 3)} +/- 0.05", f"{round((normal_CSP3_all+normal_CSP3_arith)/2, 3)} +/- 0.28", f"{round((normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith)/3, 3)} +/- 0.52", f"{round((normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/3, 3)} +/- 0.24", f"{round((normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith+normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/6, 3)} +/- 0.30"],
    'Slope\nRatio': [f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith),3)}", f"{round((r_mm_CSP2_arith/r_mm_CSP2_baatz_arith),3)}", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith),3)}", f"{round(slope_CSP1/r_mm_CSP1_baatz, 3)}", f"{round(slope_CSP2/r_mm_CSP2_baatz, 3)}", f"{round(slope_CSP3/r_mm_CSP3_baatz, 3)}", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz)/4, 3)} +/- 0.06", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+slope_CSP3/r_mm_CSP3_baatz)/2,3)} +/- 0.16", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP3_arith/r_mm_CSP3_baatz_arith)/3, 3)} +/- 0.19", f"{round((slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/3, 3)} +/- 0.05", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/6, 3)} +/- 0.11"],}

df_1_arith = pd.DataFrame(data_1_arith)

def render_mpl_table(data_1_arith, col_width=2.8, row_height=0.6, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_1_arith.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(8,12))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_1_arith.values, bbox=bbox, colLabels=data_1_arith.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            cell.set_height(0.6)  # Set the height of header row to 0.6
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.set_height(0.5)  # Set the height of non-header rows to 0.2
    ax.set_title('Arithmetic Weighted Neutron Reduction')
    return ax

render_mpl_table(df_1_arith, header_columns=0, col_width=2.8)

# Save the table as an image
plt.savefig('YOUR PATH/Update_Summary_Table_with_TDR_arith.png', dpi=300, bbox_inches='tight')
plt.show()



############################################
############################################
############################################
############################################
############################################
#10cm 25 cm DEPTHS ONLY
############################################
############################################
###################2023#####################
############################################
############################################
############################################
############################################
###################2023#####################
############################################
############################################

############################################
################## CSP1 ####################
############################################
#import spreadsheet
GWC_CSP1_weighted_tdrdepth = pd.read_csv('YOUR PATH/2023_CSP1_SWC_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP1_Upper_tdrdepth = GWC_CSP1_weighted_tdrdepth['Upper']
SWC_CSP1_Lower_tdrdepth = GWC_CSP1_weighted_tdrdepth['Lower']
SWC_CSP1_Weighted_tdrdepth = ((SWC_CSP1_Upper_tdrdepth * 0.75) + (SWC_CSP1_Lower_tdrdepth * 0.25))
a_CSP1_weighted_tdrdepth = SWC_CSP1_Weighted_tdrdepth

pwc_CSP1_tdrdepth = a_CSP1_weighted_tdrdepth
lwc = 0.058
soc = 0.0092
N0_CSP1_weighted_tdrdepth = 2858.27

#use parameters to calculate N0
N_CSP1_tdrdepth = N0_CSP1_weighted_tdrdepth*(((0.0808)/((pwc_CSP1_tdrdepth+lwc+soc)+0.115))+0.372)
N_CSP1_tdrdepth.to_csv('YOUR PATH/N_CSP1_topdepths.csv')
N_CSP1_23_tdrdepth = pd.read_csv('YOUR PATH/N_CSP1_topdepths.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP1_2023_tdrdepth = N_CSP1_23_tdrdepth['0']

NPIV_CSP1_weighted_tdrdepth = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_weighted_tdrdepth = NPIV_CSP1_weighted_tdrdepth['NPIV']

#Calculate forward
fveg_CSP1_tdrdepth = NPIVCount_CSP1_weighted_tdrdepth / N_CSP1_2023_tdrdepth
fveg_CSP1_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP1_topdepths.csv')
fveg_CSP1_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP1_topdepths.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_1_tdrdepth = fveg_CSP1_1_tdrdepth['0']

#Import BWE for plotting
BWEcsv_CSP1_tdrdepth = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_tdrdepth = BWEcsv_CSP1_tdrdepth['BWE']

#Set up plotting for both foward and inverse
x_CSP1_tdrdepth = BWE_CSP1_tdrdepth
y_CSP1_tdrdepth = 1/fveg_CSP1_1_tdrdepth

#Least square regression
A_CSP1_tdrdepth = np.vstack([x_CSP1_tdrdepth, np.ones(len(x_CSP1_tdrdepth))]).T
y_CSP1_tdrdepth = np.array(y_CSP1_tdrdepth) 
y_CSP1_tdrdepth = y_CSP1_tdrdepth[:, np.newaxis]
alpha_CSP1_tdrdepth = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_tdrdepth.T,A_CSP1_tdrdepth)),A_CSP1_tdrdepth.T)),y_CSP1_tdrdepth)
m_CSP1_tdrdepth = alpha_CSP1_tdrdepth[0]
n_CSP1_tdrdepth = alpha_CSP1_tdrdepth[1]
mm_CSP1_tdrdepth = float(m_CSP1_tdrdepth)
mmm_CSP1_tdrdepth = [float(m_CSP1_tdrdepth)]
nn_CSP1_tdrdepth = float(n_CSP1_tdrdepth)
r_mm_CSP1_tdrdepth = round(mm_CSP1_tdrdepth,4)
r_nn_CSP1_tdrdepth = round(nn_CSP1_tdrdepth,4)

#Calculate r-squared
x_CSP1_r2_tdrdepth = sm.add_constant(x_CSP1_tdrdepth)
model_tdrdepth = sm.OLS(y_CSP1_tdrdepth, x_CSP1_r2_tdrdepth)
results_tdrdepth = model_tdrdepth.fit()
r2_CSP1_tdrdepth = results_tdrdepth.rsquared
r2_CSP1_tdrdepth = round(r2_CSP1_tdrdepth,2)

#Plot data with best fit line, slope, and r-squared
a_CSP1_tdrdepth, b_CSP1_tdrdepth = np.polyfit(x_CSP1_tdrdepth,y_CSP1_tdrdepth,1)
plt.scatter(x_CSP1_tdrdepth, y_CSP1_tdrdepth, marker = 'o', color = 'red')
plt.plot(x_CSP1_tdrdepth, a_CSP1_tdrdepth*x_CSP1_tdrdepth+b_CSP1_tdrdepth, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP1_tdrdepth,r_nn_CSP1_tdrdepth, r2_CSP1_tdrdepth]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


#import spreadsheet
GWC_CSP2_weighted_tdrdepth = pd.read_csv('YOUR PATH/2023_CSP2_SWC_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP2_Upper_tdrdepth = GWC_CSP2_weighted_tdrdepth['Upper']
SWC_CSP2_Lower_tdrdepth = GWC_CSP2_weighted_tdrdepth['Lower']
SWC_CSP2_Weighted_tdrdepth = ((SWC_CSP2_Upper_tdrdepth * 0.75) + (SWC_CSP2_Lower_tdrdepth * 0.25))
a_CSP2_weighted_tdrdepth = SWC_CSP2_Weighted_tdrdepth

pwc_CSP2_tdrdepth = a_CSP2_weighted_tdrdepth
lwc = 0.058
soc = 0.0092
N0_CSP2_weighted_tdrdepth = 2946.61

#use parameters to calculate N0
N_CSP2_tdrdepth = N0_CSP2_weighted_tdrdepth*(((0.0808)/((pwc_CSP2_tdrdepth+lwc+soc)+0.115))+0.372)
N_CSP2_tdrdepth.to_csv('YOUR PATH/N_CSP2_topdepths.csv')
N_CSP2_23_tdrdepth = pd.read_csv('YOUR PATH/N_CSP2_topdepths.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP2_2023_tdrdepth = N_CSP2_23_tdrdepth['0']

NPIV_CSP2_weighted_tdrdepth = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_weighted_tdrdepth = NPIV_CSP2_weighted_tdrdepth['NPIV']

#Calculate forward
fveg_CSP2_tdrdepth = NPIVCount_CSP2_weighted_tdrdepth / N_CSP2_2023_tdrdepth
fveg_CSP2_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP2_topdepths.csv')
fveg_CSP2_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP2_topdepths.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_1_tdrdepth = fveg_CSP2_1_tdrdepth['0']

#Import BWE for plotting
BWEcsv_CSP2_tdrdepth = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_tdrdepth = BWEcsv_CSP2_tdrdepth['BWE']

#Set up plotting for both forward and inverse
x_CSP2_tdrdepth = BWE_CSP2_tdrdepth
y_CSP2_tdrdepth = 1/fveg_CSP2_1_tdrdepth

#Least square regression
A_CSP2_tdrdepth = np.vstack([x_CSP2_tdrdepth, np.ones(len(x_CSP2_tdrdepth))]).T
y_CSP2_tdrdepth = np.array(y_CSP2_tdrdepth) 
y_CSP2_tdrdepth = y_CSP2_tdrdepth[:, np.newaxis]
alpha_CSP2_tdrdepth = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_tdrdepth.T,A_CSP2_tdrdepth)),A_CSP2_tdrdepth.T)),y_CSP2_tdrdepth)
m_CSP2_tdrdepth = alpha_CSP2_tdrdepth[0]
n_CSP2_tdrdepth = alpha_CSP2_tdrdepth[1]
mm_CSP2_tdrdepth = float(m_CSP2_tdrdepth)
mmm_CSP2_tdrdepth = [float(m_CSP2_tdrdepth)]
nn_CSP2_tdrdepth = float(n_CSP2_tdrdepth)
r_mm_CSP2_tdrdepth = round(mm_CSP2_tdrdepth,4)
r_nn_CSP2_tdrdepth = round(nn_CSP2_tdrdepth,4)

#Calculate r-squared
x_CSP2_r2_tdrdepth = sm.add_constant(x_CSP2_tdrdepth)
model_tdrdepth = sm.OLS(y_CSP2_tdrdepth, x_CSP2_r2_tdrdepth)
results_tdrdepth = model_tdrdepth.fit()
r2_CSP2_tdrdepth = results_tdrdepth.rsquared
r2_CSP2_tdrdepth = round(r2_CSP2_tdrdepth,2)

#Plot data with best fit line, slope, and r-squared
a_CSP2_tdrdepth, b_CSP2_tdrdepth = np.polyfit(x_CSP2_tdrdepth,y_CSP2_tdrdepth,1)
plt.scatter(x_CSP2_tdrdepth, y_CSP2_tdrdepth, marker = 'o', color = 'red')
plt.plot(x_CSP2_tdrdepth, a_CSP2_tdrdepth*x_CSP2_tdrdepth+b_CSP2_tdrdepth, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP2_tdrdepth,r_nn_CSP2_tdrdepth, r2_CSP2_tdrdepth]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2023_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


#import spreadsheet
GWC_CSP3_weighted_CRNSDistance_tdrdepth = pd.read_csv('YOUR PATH/2023_CSP3_SWC_depth_weighted_CRNS_only.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP3_Upper_CRNSDistance_tdrdepth = GWC_CSP3_weighted_CRNSDistance_tdrdepth['Upper']
SWC_CSP3_Lower_CRNSDistance_tdrdepth = GWC_CSP3_weighted_CRNSDistance_tdrdepth['Lower']
SWC_CSP3_Weighted_CRNSDistance_tdrdepth = ((SWC_CSP3_Upper_CRNSDistance_tdrdepth * 0.75) + (SWC_CSP3_Lower_CRNSDistance_tdrdepth * 0.25))
a_CSP3_weighted_CRNSDistance_tdrdepth = SWC_CSP3_Weighted_CRNSDistance_tdrdepth

pwc_CSP3_tdrdepth = a_CSP3_weighted_CRNSDistance_tdrdepth
lwc = 0.058
soc = 0.0092
N0_CSP3_weighted_tdrdepth = 1602.54

#use parameters to calculate N0
N_CSP3_tdrdepth = N0_CSP3_weighted_tdrdepth*(((0.0808)/((pwc_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N_CSP3_tdrdepth.to_csv('YOUR PATH/N_CSP3_topdepths.csv')
N_CSP3_23_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_topdepths.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
N_CSP3_2023_tdrdepth = N_CSP3_23_tdrdepth['0']

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_weighted_CRNSDistance_tdrdepth = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_weighted_CRNSDistance_tdrdepth = NPIV_CSP3_weighted_CRNSDistance_tdrdepth['NPIV']

#Calculate forward
fveg_CSP3_tdrdepth = NPIVCount_CSP3_weighted_CRNSDistance_tdrdepth / N_CSP3_2023_tdrdepth
fveg_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_topdepths.csv')
fveg_CSP3_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_topdepths.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_1_tdrdepth = fveg_CSP3_1_tdrdepth['0']

#Import BWE for plotting
BWEcsv_CSP3_tdrdepth = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_tdrdepth = BWEcsv_CSP3_tdrdepth['BWE']

#Set up plotting for both forward and inverse
x_CSP3_tdrdepth = BWE_CSP3_tdrdepth
y_CSP3_tdrdepth = 1/fveg_CSP3_1_tdrdepth

#Least square regression
A_CSP3_tdrdepth = np.vstack([x_CSP3_tdrdepth, np.ones(len(x_CSP3_tdrdepth))]).T
y_CSP3_tdrdepth = np.array(y_CSP3_tdrdepth) 
y_CSP3_tdrdepth = y_CSP3_tdrdepth[:, np.newaxis]
alpha_CSP3_tdrdepth = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_tdrdepth.T,A_CSP3_tdrdepth)),A_CSP3_tdrdepth.T)),y_CSP3_tdrdepth)
m_CSP3_tdrdepth = alpha_CSP3_tdrdepth[0]
n_CSP3_tdrdepth = alpha_CSP3_tdrdepth[1]
mm_CSP3_tdrdepth = float(m_CSP3_tdrdepth)
mmm_CSP3_tdrdepth = [float(m_CSP3_tdrdepth)]
nn_CSP3_tdrdepth = float(n_CSP3_tdrdepth)
r_mm_CSP3_tdrdepth = round(mm_CSP3_tdrdepth,4)
r_nn_CSP3_tdrdepth = round(nn_CSP3_tdrdepth,4)

#Calculate r-squared
x_CSP3_r2_tdrdepth = sm.add_constant(x_CSP3_tdrdepth)
model_tdrdepth = sm.OLS(y_CSP3_tdrdepth, x_CSP3_r2_tdrdepth)
results_tdrdepth = model_tdrdepth.fit()
r2_CSP3_tdrdepth = results_tdrdepth.rsquared
r2_CSP3_tdrdepth = round(r2_CSP3_tdrdepth,2)

#Plot data with best fit line, slope, and r-squared
a_CSP3_tdrdepth, b_CSP3_tdrdepth = np.polyfit(x_CSP3_tdrdepth,y_CSP3_tdrdepth,1)
plt.scatter(x_CSP3_tdrdepth, y_CSP3_tdrdepth, marker = 'o', color = 'red')
plt.plot(x_CSP3_tdrdepth, a_CSP3_tdrdepth*x_CSP3_tdrdepth+b_CSP3_tdrdepth, linestyle = '-', color = 'blue')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25,7.8)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept', 'r\u00b2']
table_vals = [[r_mm_CSP3_tdrdepth,r_nn_CSP3_tdrdepth, r2_CSP3_tdrdepth]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_2023_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


#Baatz 2015 function 

############################################
################## CSP1 ####################
############################################

fvegbaatz_CSP1_tdrdepth = 1/((1-((-29.32/2858.27)*BWE_CSP1_tdrdepth)))
y_baatz_CSP1_tdrdepth = 1/fvegbaatz_CSP1_tdrdepth

#Least square regression
A_CSP1_baatz_tdrdepth = np.vstack([x_CSP1_tdrdepth, np.ones(len(x_CSP1_tdrdepth))]).T
y_CSP1_baatz_tdrdepth = np.array(y_baatz_CSP1_tdrdepth)  
y_CSP1_baatz_tdrdepth = y_baatz_CSP1_tdrdepth.values[:, np.newaxis]
alpha_CSP1_baatz_tdrdepth = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_baatz_tdrdepth.T,A_CSP1_baatz_tdrdepth)),A_CSP1_baatz_tdrdepth.T)),y_CSP1_baatz_tdrdepth)
m_CSP1_baatz_tdrdepth = alpha_CSP1_baatz_tdrdepth[0]
n_CSP1_baatz_tdrdepth = alpha_CSP1_baatz_tdrdepth[1]
mm_CSP1_baatz_tdrdepth = float(m_CSP1_baatz_tdrdepth)
mmm_CSP1_baatz_tdrdepth = [float(m_CSP1_baatz_tdrdepth)]
nn_CSP1_baatz_tdrdepth = float(n_CSP1_baatz_tdrdepth)
r_mm_CSP1_baatz_tdrdepth = round(mm_CSP1_baatz_tdrdepth,4)
r_nn_CSP1_baatz_tdrdepth = round(nn_CSP1_baatz_tdrdepth,4)

#Calculate r-squared   
x_CSP1_baatz_tdrdepth = sm.add_constant(x_CSP1_tdrdepth)
model_tdrdepth = sm.OLS(y_CSP1_baatz_tdrdepth, x_CSP1_baatz_tdrdepth)
results_tdrdepth = model_tdrdepth.fit()
r2_CSP1_baatz_tdrdepth = results_tdrdepth.rsquared
r2_CSP1_baatz_tdrdepth = round(r2_CSP1_baatz_tdrdepth,2)

#Plot data with best fit line, slope, and r-squared
a_CSP1_tdrdepth, b_CSP1_tdrdepth = np.polyfit(x_CSP1_tdrdepth,y_CSP1_tdrdepth,1)
c_CSP1_tdrdepth, d_CSP1_tdrdepth = np.polyfit(x_CSP1_tdrdepth,y_baatz_CSP1_tdrdepth,1)
plt.scatter(x_CSP1_tdrdepth, y_CSP1_tdrdepth, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
#plt.scatter(x_CSP1, y_baatz_CSP1, marker = 'o', color = 'blue', label = r'$f_b$ by\nBaatz et al. (2015)')
plt.plot(x_CSP1_tdrdepth, a_CSP1_tdrdepth*x_CSP1_tdrdepth+b_CSP1_tdrdepth, linestyle = '-', color = 'red')
plt.plot(x_CSP1_tdrdepth, 0.005*x_CSP1_tdrdepth + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP1_tdrdepth], [r2_CSP1_tdrdepth]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP1_fveg_with_baatz_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
################## CSP2 ####################
############################################

fvegbaatz_CSP2_tdrdepth = 1/((1-((-50.99/2946.61)*BWE_CSP2_tdrdepth)))
y_baatz_CSP2_tdrdepth = 1/fvegbaatz_CSP2_tdrdepth

#Least square regression
A_CSP2_baatz_tdrdepth = np.vstack([x_CSP2_tdrdepth, np.ones(len(x_CSP2_tdrdepth))]).T
y_CSP2_baatz_tdrdepth = np.array(y_baatz_CSP2_tdrdepth)  
y_CSP2_baatz_tdrdepth = y_baatz_CSP2_tdrdepth.values[:, np.newaxis]
alpha_CSP2_baatz_tdrdepth = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_baatz_tdrdepth.T,A_CSP2_baatz_tdrdepth)),A_CSP2_baatz_tdrdepth.T)),y_CSP2_baatz_tdrdepth)
m_CSP2_baatz_tdrdepth = alpha_CSP2_baatz_tdrdepth[0]
n_CSP2_baatz_tdrdepth = alpha_CSP2_baatz_tdrdepth[1]
mm_CSP2_baatz_tdrdepth = float(m_CSP2_baatz_tdrdepth)
mmm_CSP2_baatz_tdrdepth = [float(m_CSP2_baatz_tdrdepth)]
nn_CSP2_baatz_tdrdepth = float(n_CSP2_baatz_tdrdepth)
r_mm_CSP2_baatz_tdrdepth = round(mm_CSP2_baatz_tdrdepth,4)
r_nn_CSP2_baatz_tdrdepth = round(nn_CSP2_baatz_tdrdepth,4)

#Calculate r-squared
x_CSP2_baatz_tdrdepth = sm.add_constant(x_CSP2_tdrdepth)
model_tdrdepth = sm.OLS(y_CSP2_baatz_tdrdepth, x_CSP2_baatz_tdrdepth)
results_tdrdepth = model_tdrdepth.fit()
r2_CSP2_baatz_tdrdepth = results_tdrdepth.rsquared
r2_CSP2_baatz_tdrdepth = round(r2_CSP2_baatz_tdrdepth,2)

#Plot data with best fit line, slope, and r-squared
a_CSP2_tdrdepth, b_CSP2_tdrdepth = np.polyfit(x_CSP2_tdrdepth,y_CSP2_tdrdepth,1)
c_CSP2_tdrdepth, d_CSP2_tdrdepth = np.polyfit(x_CSP2_tdrdepth,y_baatz_CSP2_tdrdepth,1)
plt.scatter(x_CSP2_tdrdepth, y_CSP2_tdrdepth, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
#plt.scatter(x_CSP2, y_baatz_CSP2, marker = 'o', color = 'blue', label = r'$f_b$ by\nBaatz et al. (2015)')
plt.plot(x_CSP2_tdrdepth, a_CSP2_tdrdepth*x_CSP2_tdrdepth+b_CSP2_tdrdepth, linestyle = '-', color = 'red')
plt.plot(x_CSP2_tdrdepth, 0.005*x_CSP2_tdrdepth + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP2_tdrdepth], [r2_CSP2_tdrdepth]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP2_fveg_with_Baatz_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
################## CSP3 ####################
############################################

fvegbaatz_CSP3_tdrdepth = 1/((1-((-8.88/1602.54)*BWE_CSP3_tdrdepth)))
y_baatz_CSP3_tdrdepth = 1/fvegbaatz_CSP3_tdrdepth

#Least square regression
A_CSP3_baatz_tdrdepth = np.vstack([x_CSP3_tdrdepth, np.ones(len(x_CSP3_tdrdepth))]).T
y_CSP3_baatz_tdrdepth = np.array(y_baatz_CSP3_tdrdepth)  
y_CSP3_baatz_tdrdepth = y_baatz_CSP3_tdrdepth.values[:, np.newaxis]
alpha_CSP3_baatz_tdrdepth = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_baatz_tdrdepth.T,A_CSP3_baatz_tdrdepth)),A_CSP3_baatz_tdrdepth.T)),y_CSP3_baatz_tdrdepth)
m_CSP3_baatz_tdrdepth = alpha_CSP3_baatz_tdrdepth[0]
n_CSP3_baatz_tdrdepth = alpha_CSP3_baatz_tdrdepth[1]
mm_CSP3_baatz_tdrdepth = float(m_CSP3_baatz_tdrdepth)
mmm_CSP3_baatz_tdrdepth = [float(m_CSP3_baatz_tdrdepth)]
nn_CSP3_baatz_tdrdepth = float(n_CSP3_baatz_tdrdepth)
r_mm_CSP3_baatz_tdrdepth = round(mm_CSP3_baatz_tdrdepth,4)
r_nn_CSP3_baatz_tdrdepth = round(nn_CSP3_baatz_tdrdepth,4)

#Calculate r-squared
x_CSP3_baatz_tdrdepth = sm.add_constant(x_CSP3_tdrdepth)
model_tdrdepth = sm.OLS(y_CSP3_baatz_tdrdepth, x_CSP3_baatz_tdrdepth)
results_tdrdepth = model_tdrdepth.fit()
r2_CSP3_baatz_tdrdepth = results_tdrdepth.rsquared
r2_CSP3_baatz_tdrdepth = round(r2_CSP3_baatz_tdrdepth,2)

#Plot data with best fit line, slope, and r-squared
a_CSP3_tdrdepth, b_CSP3_tdrdepth = np.polyfit(x_CSP3_tdrdepth,y_CSP3_tdrdepth,1)
c_CSP3_tdrdepth, d_CSP3_tdrdepth = np.polyfit(x_CSP3_tdrdepth,y_baatz_CSP3_tdrdepth,1)
plt.scatter(x_CSP3_tdrdepth, y_CSP3_tdrdepth, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
#plt.scatter(x_CSP3, y_baatz_CSP3, marker = 'o', color = 'blue', label = r'$f_b$ by\nBaatz et al. (2015)')
plt.plot(x_CSP3_tdrdepth, a_CSP3_tdrdepth*x_CSP3_tdrdepth+b_CSP3_tdrdepth, linestyle = '-', color = 'red')
plt.plot(x_CSP3_tdrdepth, 0.005*x_CSP3_tdrdepth + 1, linestyle = '-', color = 'blue', label = r'$f_b$ by Baatz et al. (2015)')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.8)

#Add confidence table to the plot
col_labels = ['Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3_tdrdepth], [r2_CSP3_tdrdepth]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.1] * 2,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_fveg_with_Baatz_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()

# Normalize according to Baatz 2015
normal_CSP1_tdrdepth = (29.32/2858.27)*100
normal_CSP2_tdrdepth = (50.99/2946.61)*100
normal_CSP3_tdrdepth = (8.88/1602.54)*100

##############################################
############ Table of Values #################
##############################################

data_tdrdepth = {
    '': ['Slope', 'Intercept', 'Baatz 2015\nSlope', 'Baatz 2015\nIntercept', 'Slope\nDifference', 'Slope\nRatio', 'Neutron\nIntensity\nReduction(%)'],
    'CSP1': [f"{r_mm_CSP1_tdrdepth}", f"{r_nn_CSP1_tdrdepth}", f"{r_mm_CSP1_baatz_tdrdepth}", f"{r_nn_CSP1_baatz_tdrdepth}", f"{round((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth),4)}", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth),4)}", f"{round((normal_CSP1_tdrdepth),2)}"],
    'CSP2': [f"{r_mm_CSP2_tdrdepth}", f"{r_nn_CSP2_tdrdepth}", f"{r_mm_CSP2_baatz_tdrdepth}", f"{r_nn_CSP2_baatz_tdrdepth}", f"{round((r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth),4)}", f"{round((r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth),4)}", f"{round((normal_CSP2_tdrdepth),2)}"],
    'CSP3': [f"{r_mm_CSP3_tdrdepth}", f"{r_nn_CSP3_tdrdepth}", f"{r_mm_CSP3_baatz_tdrdepth}", f"{r_nn_CSP3_baatz_tdrdepth}", f"{round((r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth),4)}", f"{round((r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth),4)}", f"{round((normal_CSP3_tdrdepth),2)}"],
    'Average': [f"{round((r_mm_CSP1_tdrdepth+r_mm_CSP2_tdrdepth+r_mm_CSP3_tdrdepth)/3,4)}", f"{round((r_nn_CSP1_tdrdepth+r_nn_CSP2_tdrdepth+r_nn_CSP3_tdrdepth)/3,4)}", f"{round((r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_baatz_tdrdepth)/3, 4)}", f"{round((r_nn_CSP1_baatz_tdrdepth+r_nn_CSP2_baatz_tdrdepth+r_nn_CSP3_baatz_tdrdepth)/3,4)}", f"{round((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth)/3,4)}", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth)/3,4)}", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth)/3,2)}"]
}

# Convert the dictionary to a pandas DataFrame
df_tdrdepth = pd.DataFrame(data_tdrdepth)

# Function to render DataFrame as a table
def render_mpl_table(data_tdrdepth, col_width=3.0, row_height=0.7, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_tdrdepth.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(12,7))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_tdrdepth.values, bbox=bbox, colLabels=data_tdrdepth.columns, **kwargs)

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

render_mpl_table(df_tdrdepth, header_columns=0, col_width=2.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Summary_Table_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP1 ################################
##################################################################
##################################################################
##################################################################


### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc19_CSP1_tdrdepth = dailywavgSWC2019_CSP1
pwc20_CSP1_tdrdepth = dailywavgSWC2020_CSP1
pwc21_CSP1_tdrdepth = dailywavgSWC2021_CSP1
pwc22_CSP1_tdrdepth = dailywavgSWC2022_CSP1
pwc23_CSP1_tdrdepth = dailywavgSWC2023_CSP1

lwc = 0.058
soc = 0.0092
N0_for_calc_tdrdepth = 2858.27

N19_CSP1_tdrdepth = dailyNPIV2019_CSP1
N20_CSP1_tdrdepth = dailyNPIV2020_CSP1
N21_CSP1_tdrdepth = dailyNPIV2021_CSP1
N22_CSP1_tdrdepth = dailyNPIV2022_CSP1
N23_CSP1_tdrdepth = dailyNPIV2023_CSP1

#Calculate N
N0_CSP1_19_tdrdepth = N0_for_calc_tdrdepth*(((0.0808)/((pwc19_CSP1_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP1_20_tdrdepth = N0_for_calc_tdrdepth*(((0.0808)/((pwc20_CSP1_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP1_21_tdrdepth = N0_for_calc_tdrdepth*(((0.0808)/((pwc21_CSP1_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP1_22_tdrdepth = N0_for_calc_tdrdepth*(((0.0808)/((pwc22_CSP1_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP1_23_tdrdepth = N0_for_calc_tdrdepth*(((0.0808)/((pwc23_CSP1_tdrdepth+lwc+soc)+0.115))+0.372)

NCSP1_19_tdrdepth= N0_CSP1_19_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP1_19_top.csv')
NCSP1_20_tdrdepth= N0_CSP1_20_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP1_20_top.csv')
NCSP1_21_tdrdepth= N0_CSP1_21_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP1_21_top.csv')
NCSP1_22_tdrdepth= N0_CSP1_22_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP1_22_top.csv')
NCSP1_23_tdrdepth= N0_CSP1_23_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP1_23_top.csv')

CSP1_19_tdrdepth = pd.read_csv('YOUR PATH/N_CSP1_19_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_20_tdrdepth = pd.read_csv('YOUR PATH/N_CSP1_20_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_21_tdrdepth = pd.read_csv('YOUR PATH/N_CSP1_21_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_22_tdrdepth = pd.read_csv('YOUR PATH/N_CSP1_22_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP1_23_tdrdepth = pd.read_csv('YOUR PATH/N_CSP1_23_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP1_19_1_tdrdepth = CSP1_19_tdrdepth['0']
N_CSP1_20_1_tdrdepth = CSP1_20_tdrdepth['0']
N_CSP1_21_1_tdrdepth = CSP1_21_tdrdepth['0']
N_CSP1_22_1_tdrdepth = CSP1_22_tdrdepth['0']
N_CSP1_23_1_tdrdepth = CSP1_23_tdrdepth['0']

#Calculate the daily filtered neutron counts
dailyNPIV2019_CSP1_tdrdepth = NPIVCount2019_CSP1.resample('D').mean()
dailyNPIV2020_CSP1_tdrdepth = NPIVCount2020_CSP1.resample('D').mean()
dailyNPIV2021_CSP1_tdrdepth = NPIVCount2021_CSP1.resample('D').mean()
dailyNPIV2022_CSP1_tdrdepth = NPIVCount2022_CSP1.resample('D').mean()
dailyNPIV2023_CSP1_tdrdepth = NPIVCount2023_CSP1.resample('D').mean()

fveg_CSP1_19_tdrdepth = dailyNPIV2019_CSP1_tdrdepth / N0_CSP1_19_tdrdepth
fveg_CSP1_20_tdrdepth = dailyNPIV2020_CSP1_tdrdepth / N0_CSP1_20_tdrdepth
fveg_CSP1_21_tdrdepth = dailyNPIV2021_CSP1_tdrdepth / N0_CSP1_21_tdrdepth
fveg_CSP1_22_tdrdepth = dailyNPIV2022_CSP1_tdrdepth / N0_CSP1_22_tdrdepth
fveg_CSP1_23_tdrdepth = dailyNPIV2023_CSP1_tdrdepth / N0_CSP1_23_tdrdepth

fveg_CSP1_19_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP1_19_top.csv')
fveg_CSP1_20_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP1_20_top.csv')
fveg_CSP1_21_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP1_21_top.csv')
fveg_CSP1_22_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP1_22_top.csv')
fveg_CSP1_23_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP1_23_top.csv')

fveg_CSP1_19_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP1_19_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_20_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP1_20_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_21_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP1_21_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_22_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP1_22_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP1_23_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP1_23_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP1_19_1_tdrdepth = fveg_CSP1_19_1_tdrdepth['0']
fveg_CSP1_20_1_tdrdepth = fveg_CSP1_20_1_tdrdepth['0']
fveg_CSP1_21_1_tdrdepth = fveg_CSP1_21_1_tdrdepth['0']
fveg_CSP1_22_1_tdrdepth = fveg_CSP1_22_1_tdrdepth['0']
fveg_CSP1_23_1_tdrdepth = fveg_CSP1_23_1_tdrdepth['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWE19_CSP1_tdrdepth = BWEcsv2019_CSP1['BWE']
BWE20_CSP1_tdrdepth = BWEcsv2020_CSP1['BWE']
BWE21_CSP1_tdrdepth = BWEcsv2021_CSP1['BWE']
BWE22_CSP1_tdrdepth = BWEcsv2022_CSP1['BWE']
BWE23_CSP1_tdrdepth = BWEcsv2023_CSP1['BWE']

#Interpolate between dates
BWEinterp19_CSP1_tdrdepth = BWE19_CSP1_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP1_tdrdepth = BWE20_CSP1_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP1_tdrdepth = BWE21_CSP1_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP1_tdrdepth = BWE22_CSP1_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP1_tdrdepth = BWE23_CSP1_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP1_tdrdepth = BWEinterp19_CSP1_tdrdepth
y19_CSP1_tdrdepth = 1/fveg_CSP1_19_1_tdrdepth
x20_CSP1_tdrdepth = BWEinterp20_CSP1_tdrdepth
y20_CSP1_tdrdepth = 1/fveg_CSP1_20_1_tdrdepth
x21_CSP1_tdrdepth = BWEinterp21_CSP1_tdrdepth
y21_CSP1_tdrdepth = 1/fveg_CSP1_21_1_tdrdepth
x22_CSP1_tdrdepth = BWEinterp22_CSP1_tdrdepth
y22_CSP1_tdrdepth = 1/fveg_CSP1_22_1_tdrdepth
x23_CSP1_tdrdepth = BWEinterp23_CSP1_tdrdepth
y23_CSP1_tdrdepth = 1/fveg_CSP1_23_1_tdrdepth

# Concatenate all x and y values
x_all_CSP1_tdrdepth = np.concatenate([x19_CSP1_tdrdepth, x20_CSP1_tdrdepth, x21_CSP1_tdrdepth, x22_CSP1_tdrdepth, x23_CSP1_tdrdepth])
y_all_CSP1_tdrdepth = np.concatenate([y19_CSP1_tdrdepth, y20_CSP1_tdrdepth, y21_CSP1_tdrdepth, y22_CSP1_tdrdepth, y23_CSP1_tdrdepth])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP1_tdrdepth = x_all_CSP1_tdrdepth.reshape(-1, 1)

# Perform linear regression
slope_CSP1_tdrdepth, intercept_CSP1_tdrdepth, r_value_CSP1_tdrdepth, p_value_CSP1_tdrdepth, std_err_CSP1_tdrdepth = stats.linregress(x_all_CSP1_tdrdepth, y_all_CSP1_tdrdepth)

# Calculate the t-value for a 95% confidence interval
t_CSP1_tdrdepth = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP1_tdrdepth) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP1_tdrdepth = t_CSP1_tdrdepth * std_err_CSP1_tdrdepth
intercept_conf_interval_CSP1_tdrdepth = t_CSP1_tdrdepth * std_err_CSP1_tdrdepth * np.sqrt((1/len(x_all_CSP1_tdrdepth)) + (np.mean(x_all_CSP1_tdrdepth)**2 / np.sum((x_all_CSP1_tdrdepth - np.mean(x_all_CSP1_tdrdepth))**2)))

# Calculate R-squared value
r_squared_CSP1_tdrdepth = r_value_CSP1_tdrdepth**2

shift_tdrdepth = 1 - intercept_CSP1_tdrdepth

plt.scatter(x19_CSP1_tdrdepth, y19_CSP1_tdrdepth + shift_tdrdepth, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP1_tdrdepth, y20_CSP1_tdrdepth + shift_tdrdepth, color = 'cyan', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP1_tdrdepth, y21_CSP1_tdrdepth + shift_tdrdepth, color = 'darkgreen', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP1_tdrdepth, y22_CSP1_tdrdepth + shift_tdrdepth, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP1_tdrdepth, y23_CSP1_tdrdepth + shift_tdrdepth, color = 'darkorange', marker = 'D', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP1_tdrdepth, b_CSP1_tdrdepth = np.polyfit(x_CSP1_tdrdepth,y_CSP1_tdrdepth,1)
c_CSP1_tdrdepth, d_CSP1_tdrdepth = np.polyfit(x_CSP1_tdrdepth,y_baatz_CSP1_tdrdepth,1)
plt.scatter(x_CSP1_tdrdepth, y_CSP1_tdrdepth, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP1_tdrdepth, 0.005*x_CSP1_tdrdepth + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP1_tdrdepth, slope_CSP1_tdrdepth*x_all_CSP1_tdrdepth + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP1_tdrdepth, a_CSP1_tdrdepth*x_CSP1_tdrdepth+b_CSP1_tdrdepth, linestyle = '-', color = 'red', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 10cm, 25cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP1_tdrdepth, round(slope_CSP1_tdrdepth,2)], [r2_CSP1_tdrdepth, round(r_squared_CSP1_tdrdepth, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP1_with_TDR_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP2 ################################
##################################################################
##################################################################
##################################################################

#Define the data for ease in calculations later
NPIVCount2019_CSP2_tdrdepth = NPIV2019_CSP2['NPIV']
NPIVCount2020_CSP2_tdrdepth = NPIV2020_CSP2['NPIV']
NPIVCount2021_CSP2_tdrdepth = NPIV2021_CSP2['NPIV']
NPIVCount2022_CSP2_tdrdepth = NPIV2022_CSP2['NPIV']
NPIVCount2023_CSP2_tdrdepth = NPIV2023_CSP2['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2019_CSP2_tdrdepth = NPIVCount2019_CSP2_tdrdepth.resample('D').mean()
dailyNPIV2020_CSP2_tdrdepth = NPIVCount2020_CSP2_tdrdepth.resample('D').mean()
dailyNPIV2021_CSP2_tdrdepth = NPIVCount2021_CSP2_tdrdepth.resample('D').mean()
dailyNPIV2022_CSP2_tdrdepth = NPIVCount2022_CSP2_tdrdepth.resample('D').mean()
dailyNPIV2023_CSP2_tdrdepth = NPIVCount2023_CSP2_tdrdepth.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc19_CSP2_tdrdepth = dailywavgSWC2019_CSP2
pwc20_CSP2_tdrdepth = dailywavgSWC2020_CSP2
pwc21_CSP2_tdrdepth = dailywavgSWC2021_CSP2
pwc22_CSP2_tdrdepth = dailywavgSWC2022_CSP2
pwc23_CSP2_tdrdepth = dailywavgSWC2023_CSP2

N19_CSP2_tdrdepth = dailyNPIV2019_CSP2_tdrdepth
N20_CSP2_tdrdepth = dailyNPIV2020_CSP2_tdrdepth
N21_CSP2_tdrdepth = dailyNPIV2021_CSP2_tdrdepth
N22_CSP2_tdrdepth = dailyNPIV2022_CSP2_tdrdepth
N23_CSP2_tdrdepth = dailyNPIV2023_CSP2_tdrdepth

N0_for_calc_2_tdrdepth = 2946.61
lwc = 0.058
soc = 0.0092

#Calculate N
N0_CSP2_19_tdrdepth = N0_for_calc_2_tdrdepth*(((0.0808)/((pwc19_CSP2_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP2_20_tdrdepth = N0_for_calc_2_tdrdepth*(((0.0808)/((pwc20_CSP2_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP2_21_tdrdepth = N0_for_calc_2_tdrdepth*(((0.0808)/((pwc21_CSP2_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP2_22_tdrdepth = N0_for_calc_2_tdrdepth*(((0.0808)/((pwc22_CSP2_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP2_23_tdrdepth = N0_for_calc_2_tdrdepth*(((0.0808)/((pwc23_CSP2_tdrdepth+lwc+soc)+0.115))+0.372)

N0interp19_CSP2_tdrdepth = N0_CSP2_19_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP2_tdrdepth = N0_CSP2_20_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP2_tdrdepth = N0_CSP2_21_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP2_tdrdepth = N0_CSP2_22_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP2_tdrdepth = N0_CSP2_23_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')

NCSP2_19_tdrdepth= N0interp19_CSP2_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP2_19_top.csv')
NCSP2_20_tdrdepth= N0interp20_CSP2_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP2_20_top.csv')
NCSP2_21_tdrdepth= N0interp21_CSP2_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP2_21_top.csv')
NCSP2_22_tdrdepth= N0interp22_CSP2_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP2_22_top.csv')
NCSP2_23_tdrdepth= N0interp23_CSP2_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP2_23_top.csv')

CSP2_19_tdrdepth = pd.read_csv('YOUR PATH/N_CSP2_19_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_20_tdrdepth = pd.read_csv('YOUR PATH/N_CSP2_20_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_21_tdrdepth = pd.read_csv('YOUR PATH/N_CSP2_21_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_22_tdrdepth = pd.read_csv('YOUR PATH/N_CSP2_22_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP2_23_tdrdepth = pd.read_csv('YOUR PATH/N_CSP2_23_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP2_19_1_tdrdepth = CSP2_19_tdrdepth['0']
N_CSP2_20_1_tdrdepth = CSP2_20_tdrdepth['0']
N_CSP2_21_1_tdrdepth = CSP2_21_tdrdepth['0']
N_CSP2_22_1_tdrdepth = CSP2_22_tdrdepth['0']
N_CSP2_23_1_tdrdepth = CSP2_23_tdrdepth['0']

fveg_CSP2_19_tdrdepth = dailyNPIV2019_CSP2_tdrdepth / N0_CSP2_19_tdrdepth
fveg_CSP2_20_tdrdepth = dailyNPIV2020_CSP2_tdrdepth / N0_CSP2_20_tdrdepth
fveg_CSP2_21_tdrdepth = dailyNPIV2021_CSP2_tdrdepth / N0_CSP2_21_tdrdepth
fveg_CSP2_22_tdrdepth = dailyNPIV2022_CSP2_tdrdepth / N0_CSP2_22_tdrdepth
fveg_CSP2_23_11_tdrdepth = dailyNPIV2023_CSP2_tdrdepth / N0_CSP2_23_tdrdepth

fveg_CSP2_19_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP2_19_top.csv')
fveg_CSP2_20_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP2_20_top.csv')
fveg_CSP2_21_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP2_21_top.csv')
fveg_CSP2_22_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP2_22_top.csv')
fveg_CSP2_23_11_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP2_23_top.csv')

fveg_CSP2_19_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP2_19_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_20_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP2_20_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_21_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP2_21_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_22_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP2_22_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP2_23_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP2_23_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP2_19_1_tdrdepth = fveg_CSP2_19_1_tdrdepth['0']
fveg_CSP2_20_1_tdrdepth = fveg_CSP2_20_1_tdrdepth['0']
fveg_CSP2_21_1_tdrdepth = fveg_CSP2_21_1_tdrdepth['0']
fveg_CSP2_22_1_tdrdepth = fveg_CSP2_22_1_tdrdepth['0']
fveg_CSP2_23_1_tdrdepth = fveg_CSP2_23_1_tdrdepth['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2019_CSP2_tdrdepth = pd.read_csv('YOUR PATH/CSP22019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP2_tdrdepth = pd.read_csv('YOUR PATH/CSP22020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP2_tdrdepth = pd.read_csv('YOUR PATH/CSP22021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP2_tdrdepth = pd.read_csv('YOUR PATH/CSP22022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP2_tdrdepth = pd.read_csv('YOUR PATH/CSP22023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE19_CSP2_tdrdepth = BWEcsv2019_CSP2_tdrdepth['BWE']
BWE20_CSP2_tdrdepth = BWEcsv2020_CSP2_tdrdepth['BWE']
BWE21_CSP2_tdrdepth = BWEcsv2021_CSP2_tdrdepth['BWE']
BWE22_CSP2_tdrdepth = BWEcsv2022_CSP2_tdrdepth['BWE']
BWE23_CSP2_tdrdepth = BWEcsv2023_CSP2_tdrdepth['BWE']

#Interpolate between dates
BWEinterp19_CSP2_tdrdepth = BWE19_CSP2_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP2_tdrdepth = BWE20_CSP2_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP2_tdrdepth = BWE21_CSP2_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP2_tdrdepth = BWE22_CSP2_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP2_tdrdepth = BWE23_CSP2_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')

fveg_CSP2_19_1_tdrdepth = fveg_CSP2_19_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_20_1_tdrdepth = fveg_CSP2_20_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_21_1_tdrdepth = fveg_CSP2_21_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_22_1_tdrdepth = fveg_CSP2_22_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP2_23_1_tdrdepth = fveg_CSP2_23_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan)

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP2_tdrdepth = BWEinterp19_CSP2_tdrdepth
y19_CSP2_tdrdepth = 1/fveg_CSP2_19_1_tdrdepth
x20_CSP2_tdrdepth = BWEinterp20_CSP2_tdrdepth
y20_CSP2_tdrdepth = 1/fveg_CSP2_20_1_tdrdepth
x21_CSP2_tdrdepth = BWEinterp21_CSP2_tdrdepth
y21_CSP2_tdrdepth = 1/fveg_CSP2_21_1_tdrdepth
x22_CSP2_tdrdepth = BWEinterp22_CSP2_tdrdepth
y22_CSP2_tdrdepth = 1/fveg_CSP2_22_1_tdrdepth
x23_CSP2_tdrdepth = BWEinterp23_CSP2_tdrdepth
y23_CSP2_tdrdepth = 1/fveg_CSP2_23_1_tdrdepth

def fill_nan_with_avg(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr
y23_CSP2_tdrdepth = fill_nan_with_avg(y23_CSP2_tdrdepth)
# Concatenate all x and y values
x_all_CSP2_tdrdepth = np.concatenate([x19_CSP2_tdrdepth, x20_CSP2_tdrdepth, x21_CSP2_tdrdepth, x22_CSP2_tdrdepth, x23_CSP2_tdrdepth])
y_all_CSP2_tdrdepth = np.concatenate([y19_CSP2_tdrdepth, y20_CSP2_tdrdepth, y21_CSP2_tdrdepth, y22_CSP2_tdrdepth, y23_CSP2_tdrdepth])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP2_tdrdepth = x_all_CSP2_tdrdepth.reshape(-1, 1)

# Perform linear regression
slope_CSP2_tdrdepth, intercept_CSP2_tdrdepth, r_value_CSP2_tdrdepth, p_value_CSP2_tdrdepth, std_err_CSP2_tdrdepth = stats.linregress(x_all_CSP2_tdrdepth, y_all_CSP2_tdrdepth)

# Calculate the t-value for a 95% confidence interval
t_CSP2_tdrdepth = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP2_tdrdepth) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP2_tdrdepth = t_CSP2_tdrdepth * std_err_CSP2_tdrdepth
intercept_conf_interval_CSP2_tdrdepth = t_CSP2_tdrdepth * std_err_CSP2_tdrdepth * np.sqrt((1/len(x_all_CSP2_tdrdepth)) + (np.mean(x_all_CSP2_tdrdepth)**2 / np.sum((x_all_CSP2_tdrdepth - np.mean(x_all_CSP2_tdrdepth))**2)))

# Calculate R-squared value
r_squared_CSP2_tdrdepth = r_value_CSP2_tdrdepth**2

shift_CSP2_tdrdepth = 1 - intercept_CSP2_tdrdepth

plt.scatter(x19_CSP2_tdrdepth, y19_CSP2_tdrdepth + shift_CSP2_tdrdepth, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP2_tdrdepth, y20_CSP2_tdrdepth + shift_CSP2_tdrdepth, color = 'cyan', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP2_tdrdepth, y21_CSP2_tdrdepth + shift_CSP2_tdrdepth, color = 'darkgreen', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP2_tdrdepth, y22_CSP2_tdrdepth + shift_CSP2_tdrdepth, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP2_tdrdepth, y23_CSP2_tdrdepth + shift_CSP2_tdrdepth, color = 'darkorange', marker = 'D', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP2_tdrdepth, b_CSP2_tdrdepth = np.polyfit(x_CSP2_tdrdepth,y_CSP2_tdrdepth,1)
c_CSP2_tdrdepth, d_CSP2_tdrdepth = np.polyfit(x_CSP2_tdrdepth,y_baatz_CSP2_tdrdepth,1)
plt.scatter(x_CSP2_tdrdepth, y_CSP2_tdrdepth, marker = 'o', color = 'red', label = r'$f_b$ Gravimetric')
plt.plot(x_CSP2_tdrdepth, 0.005*x_CSP2_tdrdepth + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP2_tdrdepth, slope_CSP2_tdrdepth*x_all_CSP2_tdrdepth + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP2_tdrdepth, a_CSP2_tdrdepth*x_CSP2_tdrdepth+b_CSP2_tdrdepth, linestyle = '-', color = 'red', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 10cm, 25 cm Weighted '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP2_tdrdepth, round(slope_CSP2_tdrdepth,2)], [r2_CSP2_tdrdepth, round(r_squared_CSP2_tdrdepth, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP2_with_TDR_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()



##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
##################################################################
##################################################################
##################################################################

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc11_CSP3_tdrdepth = dailywavgSWC2011_CSP3
pwc12_CSP3_tdrdepth = dailywavgSWC2012_CSP3
pwc13_CSP3_tdrdepth = dailywavgSWC2013_CSP3
pwc14_CSP3_tdrdepth = dailywavgSWC2014_CSP3
pwc15_CSP3_tdrdepth = dailywavgSWC2015_CSP3
pwc16_CSP3_tdrdepth = dailywavgSWC2016_CSP3
pwc17_CSP3_tdrdepth = dailywavgSWC2017_CSP3
pwc18_CSP3_tdrdepth = dailywavgSWC2018_CSP3
pwc19_CSP3_tdrdepth = dailywavgSWC2019_CSP3
pwc20_CSP3_tdrdepth = dailywavgSWC2020_CSP3
pwc21_CSP3_tdrdepth = dailywavgSWC2021_CSP3
pwc22_CSP3_tdrdepth = dailywavgSWC2022_CSP3
pwc23_CSP3_tdrdepth = dailywavgSWC2023_CSP3

N11_CSP3_tdrdepth = dailyNPIV2011_CSP3
N12_CSP3_tdrdepth = dailyNPIV2012_CSP3
N13_CSP3_tdrdepth = dailyNPIV2013_CSP3
N14_CSP3_tdrdepth = dailyNPIV2014_CSP3
N15_CSP3_tdrdepth = dailyNPIV2015_CSP3
N16_CSP3_tdrdepth = dailyNPIV2016_CSP3
N17_CSP3_tdrdepth = dailyNPIV2017_CSP3
N18_CSP3_tdrdepth = dailyNPIV2018_CSP3
N19_CSP3_tdrdepth = dailyNPIV2019_CSP3
N20_CSP3_tdrdepth = dailyNPIV2020_CSP3
N21_CSP3_tdrdepth = dailyNPIV2021_CSP3
N22_CSP3_tdrdepth = dailyNPIV2022_CSP3
N23_CSP3_tdrdepth = dailyNPIV2023_CSP3

N0_for_calc_3_tdrdepth = 1602.54
lwc = 0.058
soc = 0.0092

#Calculate N
N0_CSP3_11_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc11_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_12_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc12_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_13_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc13_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_14_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc14_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_15_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc15_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_16_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc16_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_17_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc17_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_18_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc18_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_19_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc19_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_20_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc20_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_21_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc21_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_22_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc22_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)
N0_CSP3_23_tdrdepth = N0_for_calc_3_tdrdepth*(((0.0808)/((pwc23_CSP3_tdrdepth+lwc+soc)+0.115))+0.372)

N0interp11_CSP3_tdrdepth = N0_CSP3_11_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp12_CSP3_tdrdepth = N0_CSP3_12_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp13_CSP3_tdrdepth = N0_CSP3_13_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp14_CSP3_tdrdepth = N0_CSP3_14_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp15_CSP3_tdrdepth = N0_CSP3_15_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp16_CSP3_tdrdepth = N0_CSP3_16_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp17_CSP3_tdrdepth = N0_CSP3_17_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp18_CSP3_tdrdepth = N0_CSP3_18_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp19_CSP3_tdrdepth = N0_CSP3_19_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP3_tdrdepth = N0_CSP3_20_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP3_tdrdepth = N0_CSP3_21_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP3_tdrdepth = N0_CSP3_22_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP3_tdrdepth = N0_CSP3_23_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')

NCSP3_11_tdrdepth= N0interp11_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_11_top.csv')
NCSP3_12_tdrdepth= N0interp12_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_12_top.csv')
NCSP3_13_tdrdepth= N0interp13_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_13_top.csv')
NCSP3_14_tdrdepth= N0interp14_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_14_top.csv')
NCSP3_15_tdrdepth= N0interp15_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_15_top.csv')
NCSP3_16_tdrdepth= N0interp16_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_16_top.csv')
NCSP3_17_tdrdepth= N0interp17_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_17_top.csv')
NCSP3_18_tdrdepth= N0interp18_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_18_top.csv')
NCSP3_19_tdrdepth= N0interp19_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_19_top.csv')
NCSP3_20_tdrdepth= N0interp20_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_20_top.csv')
NCSP3_21_tdrdepth= N0interp21_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_21_top.csv')
NCSP3_22_tdrdepth= N0interp22_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_22_top.csv')
NCSP3_23_tdrdepth= N0interp23_CSP3_tdrdepth.to_csv('YOUR PATH TO MAKE/N_CSP3_23_top.csv')

CSP3_11_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_11_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_12_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_12_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_13_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_13_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_14_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_14_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_15_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_15_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_16_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_16_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_17_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_17_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_18_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_18_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_19_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_19_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_20_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_20_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_21_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_21_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_22_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_22_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
CSP3_23_tdrdepth = pd.read_csv('YOUR PATH/N_CSP3_23_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

N_CSP3_11_1_tdrdepth = CSP3_11_tdrdepth['0']
N_CSP3_12_1_tdrdepth = CSP3_12_tdrdepth['0']
N_CSP3_13_1_tdrdepth = CSP3_13_tdrdepth['0']
N_CSP3_14_1_tdrdepth = CSP3_14_tdrdepth['0']
N_CSP3_15_1_tdrdepth = CSP3_15_tdrdepth['0']
N_CSP3_16_1_tdrdepth = CSP3_16_tdrdepth['0']
N_CSP3_17_1_tdrdepth = CSP3_17_tdrdepth['0']
N_CSP3_18_1_tdrdepth = CSP3_18_tdrdepth['0']
N_CSP3_19_1_tdrdepth = CSP3_19_tdrdepth['0']
N_CSP3_20_1_tdrdepth = CSP3_20_tdrdepth['0']
N_CSP3_21_1_tdrdepth = CSP3_21_tdrdepth['0']
N_CSP3_22_1_tdrdepth = CSP3_22_tdrdepth['0']
N_CSP3_23_1_tdrdepth = CSP3_23_tdrdepth['0']

#Calculate the daily filtered neutron counts
dailyNPIV2011_CSP3_tdrdepth = NPIVCount2011_CSP3.resample('D').mean()
dailyNPIV2012_CSP3_tdrdepth = NPIVCount2012_CSP3.resample('D').mean()
dailyNPIV2013_CSP3_tdrdepth = NPIVCount2013_CSP3.resample('D').mean()
dailyNPIV2014_CSP3_tdrdepth = NPIVCount2014_CSP3.resample('D').mean()
dailyNPIV2015_CSP3_tdrdepth = NPIVCount2015_CSP3.resample('D').mean()
dailyNPIV2016_CSP3_tdrdepth = NPIVCount2016_CSP3.resample('D').mean()
dailyNPIV2017_CSP3_tdrdepth = NPIVCount2017_CSP3.resample('D').mean()
dailyNPIV2018_CSP3_tdrdepth = NPIVCount2018_CSP3.resample('D').mean()
dailyNPIV2019_CSP3_tdrdepth = NPIVCount2019_CSP3.resample('D').mean()
dailyNPIV2020_CSP3_tdrdepth = NPIVCount2020_CSP3.resample('D').mean()
dailyNPIV2021_CSP3_tdrdepth = NPIVCount2021_CSP3.resample('D').mean()
dailyNPIV2022_CSP3_tdrdepth = NPIVCount2022_CSP3.resample('D').mean()
dailyNPIV2023_CSP3_tdrdepth = NPIVCount2023_CSP3.resample('D').mean()

fveg_CSP3_11_tdrdepth = dailyNPIV2011_CSP3_tdrdepth / N0_CSP3_11_tdrdepth
fveg_CSP3_12_tdrdepth = dailyNPIV2012_CSP3_tdrdepth / N0_CSP3_12_tdrdepth
fveg_CSP3_13_tdrdepth = dailyNPIV2013_CSP3_tdrdepth / N0_CSP3_13_tdrdepth
fveg_CSP3_14_tdrdepth = dailyNPIV2014_CSP3_tdrdepth / N0_CSP3_14_tdrdepth
fveg_CSP3_15_1_tdrdepth = dailyNPIV2015_CSP3_tdrdepth / N0_CSP3_15_tdrdepth
fveg_CSP3_16_tdrdepth = dailyNPIV2016_CSP3_tdrdepth / N0_CSP3_16_tdrdepth
fveg_CSP3_17_tdrdepth = dailyNPIV2017_CSP3_tdrdepth / N0_CSP3_17_tdrdepth
fveg_CSP3_18_tdrdepth = dailyNPIV2018_CSP3_tdrdepth / N0_CSP3_18_tdrdepth
fveg_CSP3_19_tdrdepth = dailyNPIV2019_CSP3_tdrdepth / N0_CSP3_19_tdrdepth
fveg_CSP3_20_1_tdrdepth = dailyNPIV2020_CSP3_tdrdepth / N0_CSP3_20_tdrdepth
fveg_CSP3_21_tdrdepth = dailyNPIV2021_CSP3_tdrdepth / N0_CSP3_21_tdrdepth
fveg_CSP3_22_1_tdrdepth = dailyNPIV2022_CSP3_tdrdepth / N0_CSP3_22_tdrdepth
fveg_CSP3_23_1_tdrdepth = dailyNPIV2023_CSP3_tdrdepth / N0_CSP3_23_tdrdepth

fveg_CSP3_11_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_11_top.csv')
fveg_CSP3_12_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_12_top.csv')
fveg_CSP3_13_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_13_top.csv')
fveg_CSP3_14_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_14_top.csv')
fveg_CSP3_15_1_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_15_top_1.csv')
fveg_CSP3_16_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_16_top.csv')
fveg_CSP3_17_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_17_top.csv')
fveg_CSP3_18_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_18_top.csv')
fveg_CSP3_19_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_19_top.csv')
fveg_CSP3_20_1_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_20_top_1.csv')
fveg_CSP3_21_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_21_top.csv')
fveg_CSP3_22_1_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_22_top_1.csv')
fveg_CSP3_23_1_tdrdepth.to_csv('YOUR PATH TO MAKE/fveg_CSP3_23_top_1.csv')

fveg_CSP3_11_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_11_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_12_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_12_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_13_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_13_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_14_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_14_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_15_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_15_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_16_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_16_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_17_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_17_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_18_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_18_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_19_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_19_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_20_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_20_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_21_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_21_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_22_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_22_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
fveg_CSP3_23_1_tdrdepth = pd.read_csv('YOUR PATH/fveg_CSP3_23_top.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

fveg_CSP3_11_1_tdrdepth = fveg_CSP3_11_1_tdrdepth['0']
fveg_CSP3_12_1_tdrdepth = fveg_CSP3_12_1_tdrdepth['0']
fveg_CSP3_13_1_tdrdepth = fveg_CSP3_13_1_tdrdepth['0']
fveg_CSP3_14_1_tdrdepth = fveg_CSP3_14_1_tdrdepth['0']
fveg_CSP3_15_1_tdrdepth = fveg_CSP3_15_1_tdrdepth['0']
fveg_CSP3_16_1_tdrdepth = fveg_CSP3_16_1_tdrdepth['0']
fveg_CSP3_17_1_tdrdepth = fveg_CSP3_17_1_tdrdepth['0']
fveg_CSP3_18_1_tdrdepth = fveg_CSP3_18_1_tdrdepth['0']
fveg_CSP3_19_1_tdrdepth = fveg_CSP3_19_1_tdrdepth['0']
fveg_CSP3_20_1_tdrdepth = fveg_CSP3_20_1_tdrdepth['0']
fveg_CSP3_21_1_tdrdepth = fveg_CSP3_21_1_tdrdepth['0']
fveg_CSP3_22_1_tdrdepth = fveg_CSP3_22_1_tdrdepth['0']
fveg_CSP3_23_1_tdrdepth = fveg_CSP3_23_1_tdrdepth['0']

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2011_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32011BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2012_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32012BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2013_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32013BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2014_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32014BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2015_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32015BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2016_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32016BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2017_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32017BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2018_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32018BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2019_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP3_tdrdepth = pd.read_csv('YOUR PATH/CSP32023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE11_CSP3_tdrdepth = BWEcsv2011_CSP3_tdrdepth['BWE']
BWE12_CSP3_tdrdepth = BWEcsv2012_CSP3_tdrdepth['BWE']
BWE13_CSP3_tdrdepth = BWEcsv2013_CSP3_tdrdepth['BWE']
BWE14_CSP3_tdrdepth = BWEcsv2014_CSP3_tdrdepth['BWE']
BWE15_CSP3_tdrdepth = BWEcsv2015_CSP3_tdrdepth['BWE']
BWE16_CSP3_tdrdepth = BWEcsv2016_CSP3_tdrdepth['BWE']
BWE17_CSP3_tdrdepth = BWEcsv2017_CSP3_tdrdepth['BWE']
BWE18_CSP3_tdrdepth = BWEcsv2018_CSP3_tdrdepth['BWE']
BWE19_CSP3_tdrdepth = BWEcsv2019_CSP3_tdrdepth['BWE']
BWE20_CSP3_tdrdepth = BWEcsv2020_CSP3_tdrdepth['BWE']
BWE21_CSP3_tdrdepth = BWEcsv2021_CSP3_tdrdepth['BWE']
BWE22_CSP3_tdrdepth = BWEcsv2022_CSP3_tdrdepth['BWE']
BWE23_CSP3_tdrdepth = BWEcsv2023_CSP3_tdrdepth['BWE']

#Interpolate between dates
BWEinterp11_CSP3_tdrdepth = BWE11_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp12_CSP3_tdrdepth = BWE12_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp13_CSP3_tdrdepth = BWE13_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp14_CSP3_tdrdepth = BWE14_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp15_CSP3_tdrdepth = BWE15_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp16_CSP3_tdrdepth = BWE16_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp17_CSP3_tdrdepth = BWE17_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp18_CSP3_tdrdepth = BWE18_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp19_CSP3_tdrdepth = BWE19_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp20_CSP3_tdrdepth = BWE20_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP3_tdrdepth = BWE21_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP3_tdrdepth = BWE22_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP3_tdrdepth = BWE23_CSP3_tdrdepth.interpolate(method = 'linear', limit_direction = 'forward')

BWEFillNaN11_CSP3_tdrdepth = BWEinterp11_CSP3_tdrdepth.fillna(int(BWEinterp11_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN12_CSP3_tdrdepth = BWEinterp12_CSP3_tdrdepth.fillna(int(BWEinterp12_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN13_CSP3_tdrdepth = BWEinterp13_CSP3_tdrdepth.fillna(int(BWEinterp13_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN14_CSP3_tdrdepth = BWEinterp14_CSP3_tdrdepth.fillna(int(BWEinterp14_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN15_CSP3_tdrdepth = BWEinterp15_CSP3_tdrdepth.fillna(int(BWEinterp15_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN16_CSP3_tdrdepth = BWEinterp16_CSP3_tdrdepth.fillna(int(BWEinterp16_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN17_CSP3_tdrdepth = BWEinterp17_CSP3_tdrdepth.fillna(int(BWEinterp17_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN18_CSP3_tdrdepth = BWEinterp18_CSP3_tdrdepth.fillna(int(BWEinterp18_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN19_CSP3_tdrdepth = BWEinterp19_CSP3_tdrdepth.fillna(int(BWEinterp19_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN20_CSP3_tdrdepth = BWEinterp20_CSP3_tdrdepth.fillna(int(BWEinterp20_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN21_CSP3_tdrdepth = BWEinterp21_CSP3_tdrdepth.fillna(int(BWEinterp21_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN22_CSP3_tdrdepth = BWEinterp22_CSP3_tdrdepth.fillna(int(BWEinterp22_CSP3_tdrdepth.mean()), inplace=True)
BWEFillNaN23_CSP3_tdrdepth = BWEinterp23_CSP3_tdrdepth.fillna(int(BWEinterp23_CSP3_tdrdepth.mean()), inplace=True)

fveg_CSP3_11_1_tdrdepth = fveg_CSP3_11_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_12_1_tdrdepth = fveg_CSP3_12_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_13_1_tdrdepth = fveg_CSP3_13_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_14_1_tdrdepth = fveg_CSP3_14_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_15_1_tdrdepth = fveg_CSP3_15_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_16_1_tdrdepth = fveg_CSP3_16_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_17_1_tdrdepth = fveg_CSP3_17_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_18_1_tdrdepth = fveg_CSP3_18_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_19_1_tdrdepth = fveg_CSP3_19_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_20_1_tdrdepth = fveg_CSP3_20_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_21_1_tdrdepth = fveg_CSP3_21_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_22_1_tdrdepth = fveg_CSP3_22_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
fveg_CSP3_23_1_tdrdepth = fveg_CSP3_23_1_tdrdepth.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x11_CSP3_tdrdepth = BWEinterp11_CSP3_tdrdepth
y11_CSP3_tdrdepth = 1/fveg_CSP3_11_1_tdrdepth
x12_CSP3_tdrdepth = BWEinterp12_CSP3_tdrdepth
y12_CSP3_tdrdepth = 1/fveg_CSP3_12_1_tdrdepth
x13_CSP3_tdrdepth = BWEinterp13_CSP3_tdrdepth
y13_CSP3_tdrdepth = 1/fveg_CSP3_13_1_tdrdepth
x14_CSP3_tdrdepth = BWEinterp14_CSP3_tdrdepth
y14_CSP3_tdrdepth = 1/fveg_CSP3_14_1_tdrdepth
x15_CSP3_tdrdepth = BWEinterp15_CSP3_tdrdepth
y15_CSP3_tdrdepth = 1/fveg_CSP3_15_1_tdrdepth
x16_CSP3_tdrdepth = BWEinterp16_CSP3_tdrdepth
y16_CSP3_tdrdepth = 1/fveg_CSP3_16_1_tdrdepth
x17_CSP3_tdrdepth = BWEinterp17_CSP3_tdrdepth
y17_CSP3_tdrdepth = 1/fveg_CSP3_17_1_tdrdepth
x18_CSP3_tdrdepth = BWEinterp18_CSP3_tdrdepth
y18_CSP3_tdrdepth = 1/fveg_CSP3_18_1_tdrdepth
x19_CSP3_tdrdepth = BWEinterp19_CSP3_tdrdepth
y19_CSP3_tdrdepth = 1/fveg_CSP3_19_1_tdrdepth
x20_CSP3_tdrdepth = BWEinterp20_CSP3_tdrdepth
y20_CSP3_tdrdepth = 1/fveg_CSP3_20_1_tdrdepth
x21_CSP3_tdrdepth = BWEinterp21_CSP3_tdrdepth
y21_CSP3_tdrdepth = 1/fveg_CSP3_21_1_tdrdepth
x22_CSP3_tdrdepth = BWEinterp22_CSP3_tdrdepth
y22_CSP3_tdrdepth = 1/fveg_CSP3_22_1_tdrdepth
x23_CSP3_tdrdepth = BWEinterp23_CSP3_tdrdepth
y23_CSP3_tdrdepth = 1/fveg_CSP3_23_1_tdrdepth

y15_CSP3_tdrdepth = fill_nan_with_avg(y15_CSP3_tdrdepth)
y20_CSP3_tdrdepth = fill_nan_with_avg(y20_CSP3_tdrdepth)
y22_CSP3_tdrdepth = fill_nan_with_avg(y22_CSP3_tdrdepth)
y23_CSP3_tdrdepth = fill_nan_with_avg(y23_CSP3_tdrdepth)

# Concatenate all x and y values
x_all_CSP3_tdrdepth = np.concatenate([x11_CSP3_tdrdepth, x12_CSP3_tdrdepth, x13_CSP3_tdrdepth, x14_CSP3_tdrdepth, x15_CSP3_tdrdepth, x16_CSP3_tdrdepth, x17_CSP3_tdrdepth, x18_CSP3_tdrdepth, x19_CSP3_tdrdepth, x20_CSP3_tdrdepth, x21_CSP3_tdrdepth, x22_CSP3_tdrdepth, x23_CSP3_tdrdepth])
y_all_CSP3_tdrdepth = np.concatenate([y11_CSP3_tdrdepth, y12_CSP3_tdrdepth, y13_CSP3_tdrdepth, y14_CSP3_tdrdepth, y15_CSP3_tdrdepth, y16_CSP3_tdrdepth, y17_CSP3_tdrdepth, y18_CSP3_tdrdepth, y19_CSP3_tdrdepth, y20_CSP3_tdrdepth, y21_CSP3_tdrdepth, y22_CSP3_tdrdepth, y23_CSP3_tdrdepth])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP3_tdrdepth = x_all_CSP3_tdrdepth.reshape(-1, 1)

# Perform linear regression
slope_CSP3_tdrdepth, intercept_CSP3_tdrdepth, r_value_CSP3_tdrdepth, p_value_CSP3_tdrdepth, std_err_CSP3_tdrdepth = stats.linregress(x_all_CSP3_tdrdepth, y_all_CSP3_tdrdepth)

# Calculate the t-value for a 95% confidence interval
t_CSP3_tdrdepth = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP3_tdrdepth) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP3_tdrdepth = t_CSP3_tdrdepth * std_err_CSP3_tdrdepth
intercept_conf_interval_CSP3_tdrdepth = t_CSP3_tdrdepth * std_err_CSP3_tdrdepth * np.sqrt((1/len(x_all_CSP3_tdrdepth)) + (np.mean(x_all_CSP3_tdrdepth)**2 / np.sum((x_all_CSP3_tdrdepth - np.mean(x_all_CSP3_tdrdepth))**2)))

# Calculate R-squared value
r_squared_CSP3_tdrdepth = r_value_CSP3_tdrdepth**2


shift_CSP3_tdrdepth = 1 - intercept_CSP3_tdrdepth

plt.scatter(x11_CSP3_tdrdepth, y11_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.3)
plt.scatter(x12_CSP3_tdrdepth, y12_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'indianred', marker = 'v', label = '2012', alpha=0.3)
plt.scatter(x13_CSP3_tdrdepth, y13_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.3)
plt.scatter(x14_CSP3_tdrdepth, y14_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'peru', marker = 's', label = '2014', alpha=0.3)
plt.scatter(x15_CSP3_tdrdepth, y15_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'gold', marker = '<', label = '2015', alpha=0.3)
plt.scatter(x16_CSP3_tdrdepth, y16_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'pink', marker = '>', label = '2016', alpha=0.3)
plt.scatter(x17_CSP3_tdrdepth, y17_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'rosybrown', marker = '.', label = '2017', alpha=0.3)
plt.scatter(x18_CSP3_tdrdepth, y18_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'fuchsia', marker = '1', label = '2018', alpha=0.3)
plt.scatter(x19_CSP3_tdrdepth, y19_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.3)
plt.scatter(x20_CSP3_tdrdepth, y20_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'aquamarine', marker = 'v', label = '2020', alpha=0.3)
plt.scatter(x21_CSP3_tdrdepth, y21_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'greenyellow', marker = '^', label = '2021', alpha=0.3)
plt.scatter(x22_CSP3_tdrdepth, y22_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'violet', marker = 's', label = '2022', alpha=0.3)
plt.scatter(x23_CSP3_tdrdepth, y23_CSP3_tdrdepth + shift_CSP3_tdrdepth, color = 'darkorange', marker = 'p', label = '2023', alpha=0.3)

#Plot data with best fit line, slope, and r-squared
a_CSP3_tdrdepth, b_CSP3_tdrdepth = np.polyfit(x_CSP3_tdrdepth,y_CSP3_tdrdepth,1)
c_CSP3_tdrdepth, d_CSP3_tdrdepth = np.polyfit(x_CSP3_tdrdepth,y_baatz_CSP3_tdrdepth,1)
plt.scatter(x_CSP3_tdrdepth, y_CSP3_tdrdepth, marker = 'o', color = 'red', label = r'$f_b$ Grav. 2023')
plt.plot(x_CSP3_tdrdepth, a_CSP3_tdrdepth*x_CSP3_tdrdepth+b_CSP3_tdrdepth, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP3_tdrdepth, 0.005*x_CSP3_tdrdepth + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
# Plot the line of best fit
plt.plot(x_all_CSP3_tdrdepth, slope_CSP3_tdrdepth*x_all_CSP3_tdrdepth + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 10cm, 25cm Weighting '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.85, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['Grav. Slope', 'TDR Slope']
row_labels = ['Value', 'r\u00b2']
table_vals = [[r_mm_CSP3_tdrdepth, round(slope_CSP3_tdrdepth,2)], [r2_CSP3_tdrdepth, round(r_squared_CSP3_tdrdepth, 2)]]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 3,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig('YOUR PATH TO SAVE/CSP3_with_TDR_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


# Normalize according to Baatz 2015
normal_CSP1_all_tdrdepth = (26.14/2893.9)*100
normal_CSP2_all_tdrdepth = (32.47/2949.04)*100
normal_CSP3_all_tdrdepth = (9.45/1621.86)*100


data_tdrdepth_1 = {
    '': ['Slope', 'Intercept', 'Baatz 2015\nSlope', 'Baatz 2015\nIntercept', 'Slope\nDifference', 'Slope\nRatio', 'Neutron\nIntensity\nReduction(%)'],
    'CSP1': [f"{r_mm_CSP1_tdrdepth}", f"{r_nn_CSP1_tdrdepth}", f"{r_mm_CSP1_baatz_tdrdepth}", f"{r_nn_CSP1_baatz_tdrdepth}", f"{round((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth),4)}", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth),4)}", f"{round((normal_CSP1_tdrdepth),2)}"],
    'CSP2': [f"{r_mm_CSP2_tdrdepth}", f"{r_nn_CSP2_tdrdepth}", f"{r_mm_CSP2_baatz_tdrdepth}", f"{r_nn_CSP2_baatz_tdrdepth}", f"{round((r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth),4)}", f"{round((r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth),4)}", f"{round((normal_CSP2_tdrdepth),2)}"],
    'CSP3': [f"{r_mm_CSP3_tdrdepth}", f"{r_nn_CSP3_tdrdepth}", f"{r_mm_CSP3_baatz_tdrdepth}", f"{r_nn_CSP3_baatz_tdrdepth}", f"{round((r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth),4)}", f"{round((r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth),4)}", f"{round((normal_CSP3_tdrdepth),2)}"],
    'CSP1\nTDR': [f"{round(val, 4)}" for val in [slope_CSP1_tdrdepth, intercept_CSP1_tdrdepth, r_mm_CSP1_baatz_tdrdepth, r_nn_CSP1_baatz_tdrdepth, slope_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth, slope_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth]] + [f"{round(normal_CSP1_all_tdrdepth, 2)}"],
    'CSP2\nTDR': [f"{round(val, 4)}" for val in [slope_CSP2_tdrdepth, intercept_CSP2_tdrdepth, r_mm_CSP2_baatz_tdrdepth, r_nn_CSP2_baatz_tdrdepth, slope_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth, slope_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth]] + [f"{round(normal_CSP2_all_tdrdepth, 2)}"],
    'CSP3\nTDR': [f"{round(val, 4)}" for val in [slope_CSP3_tdrdepth, intercept_CSP3_tdrdepth, r_mm_CSP3_baatz_tdrdepth, r_nn_CSP3_baatz_tdrdepth, slope_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth, slope_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth]] + [f"{round(normal_CSP3_all_tdrdepth, 2)}"],
    'CRS-2000/B\nAverage': [f"{round(val, 4)}" for val in [(r_mm_CSP1_tdrdepth+r_mm_CSP2_tdrdepth+slope_CSP1_tdrdepth+slope_CSP2_tdrdepth)/4, (r_nn_CSP1_tdrdepth+r_nn_CSP2_tdrdepth+intercept_CSP1_tdrdepth+intercept_CSP2_tdrdepth)/4, (r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_baatz_tdrdepth)/2, (r_nn_CSP1_baatz_tdrdepth+r_nn_CSP2_baatz_tdrdepth)/2, ((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth)+(slope_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth)+(slope_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth))/4, ((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth)+(slope_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth)+(slope_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth))/4]] + [f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP1_all_tdrdepth+normal_CSP2_all_tdrdepth)/4, 2)}"],
    'CRS-1000/B\nAverage': [f"{round(val, 4)}" for val in [(slope_CSP3_tdrdepth+r_mm_CSP3_tdrdepth)/2, (intercept_CSP3_tdrdepth+r_nn_CSP3_tdrdepth)/2, r_mm_CSP3_baatz_tdrdepth, r_nn_CSP3_baatz_tdrdepth, ((slope_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth))/2, ((slope_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth))/2]] + [f"{round((normal_CSP3_all_tdrdepth+normal_CSP3_tdrdepth)/2, 2)}"],
    'Gravimetric\nAverage': [f"{round(val, 4)}" for val in [(r_mm_CSP1_tdrdepth+r_mm_CSP2_tdrdepth+r_mm_CSP3_tdrdepth)/3, (r_nn_CSP1_tdrdepth+r_nn_CSP2_tdrdepth+r_nn_CSP3_tdrdepth)/3, (r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_baatz_tdrdepth)/3, (r_nn_CSP1_baatz_tdrdepth+r_nn_CSP2_baatz_tdrdepth+r_nn_CSP3_baatz_tdrdepth)/3, ((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth))/3, ((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth))/3]] + [f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth)/3, 2)}"], 
    'TDR\nAverage': [f"{round(val, 4)}" for val in [(slope_CSP1_tdrdepth+slope_CSP2_tdrdepth+slope_CSP3_tdrdepth)/3, (intercept_CSP1_tdrdepth+intercept_CSP2_tdrdepth+intercept_CSP3_tdrdepth)/3, (r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_baatz_tdrdepth)/3, (r_nn_CSP1_baatz_tdrdepth+r_nn_CSP2_baatz_tdrdepth+r_nn_CSP3_baatz_tdrdepth)/3, ((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth))/3, ((slope_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth)+(slope_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth)+(slope_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth))/3]] + [f"{round((normal_CSP1_all_tdrdepth+normal_CSP2_all_tdrdepth+normal_CSP3_all_tdrdepth)/3, 2)}"],
    'Overall\nAverage': [f"{round((val1 + val2) / 2, 4)}" for val1, val2 in zip(
    [(r_mm_CSP1_tdrdepth+r_mm_CSP2_tdrdepth+r_mm_CSP3_tdrdepth)/3, (r_nn_CSP1_tdrdepth+r_nn_CSP2_tdrdepth+r_nn_CSP3_tdrdepth)/3, (r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_baatz_tdrdepth)/3, (r_nn_CSP1_baatz_tdrdepth+r_nn_CSP2_baatz_tdrdepth+r_nn_CSP3_baatz_tdrdepth)/3, ((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth))/3, ((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth))/3, (normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth)/3],
    [(slope_CSP1_tdrdepth+slope_CSP2_tdrdepth+slope_CSP3_tdrdepth)/3, (intercept_CSP1_tdrdepth+intercept_CSP2_tdrdepth+intercept_CSP3_tdrdepth)/3, (r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_baatz_tdrdepth)/3, (r_nn_CSP1_baatz_tdrdepth+r_nn_CSP2_baatz_tdrdepth+r_nn_CSP3_baatz_tdrdepth)/3, ((r_mm_CSP1_tdrdepth-r_mm_CSP1_baatz_tdrdepth)+(r_mm_CSP2_tdrdepth-r_mm_CSP2_baatz_tdrdepth)+(r_mm_CSP3_tdrdepth-r_mm_CSP3_baatz_tdrdepth))/3, ((slope_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth)+(slope_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth)+(slope_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth))/3, (normal_CSP1_all_tdrdepth+normal_CSP2_all_tdrdepth+normal_CSP3_all_tdrdepth)/3]
)]
}
# Convert the dictionary to a pandas DataFrame
df_tdrdepth_1 = pd.DataFrame(data_tdrdepth_1)

# Function to render DataFrame as a table
def render_mpl_table(data_tdrdepth_1, col_width=6.0, row_height=0.7, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_tdrdepth_1.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(22,8))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_tdrdepth_1.values, bbox=bbox, colLabels=data_tdrdepth_1.columns, **kwargs)

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

render_mpl_table(df_tdrdepth_1, header_columns=0, col_width=6.0)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Summary_Table_with_TDR_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()

data_1_tdrdepth_1 = {
    '': ['CSP1', 'CSP2', 'CSP3', 'CSP1\nTDR', 'CSP2\nTDR', 'CSP3\nTDR', 'CRS-2000/B\nAverage', 'CRS-1000/B\nAverage', 'Gravimetric\nAverage', 'TDR\nAverage', 'Overall\nAverage'],
    'Neutron Intensity\nReduction\n(% per mm BWE)': [f"{round((normal_CSP1_tdrdepth),3)}", f"{round((normal_CSP2_tdrdepth),3)}", f"{round((normal_CSP3_tdrdepth),3)}", f"{round((normal_CSP1_all),3)}", f"{round(normal_CSP2_all, 3)}", f"{round(normal_CSP3_all, 3)}", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP1_all+normal_CSP2_all)/4, 3)} +/- 0.36", f"{round((normal_CSP3_all+normal_CSP3_tdrdepth)/2, 3)} +/- 0.03", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth)/3, 3)} +/- 0.67", f"{round((normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/3, 3)} +/- 0.30", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth+normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/6, 3)} +/- 0.34"],
    'Slope\nRatio': [f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth),3)}", f"{round((r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth),3)}", f"{round((r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth),3)}", f"{round(slope_CSP1/r_mm_CSP1_baatz, 3)}", f"{round(slope_CSP2/r_mm_CSP2_baatz, 3)}", f"{round(slope_CSP3/r_mm_CSP3_baatz, 3)}", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz)/4, 3)} +/- 0.23", f"{round((r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth+slope_CSP3/r_mm_CSP3_baatz)/2,3)} +/- 0.25", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth)/3, 3)} +/- 0.05", f"{round((slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/3, 3)} +/- 0.07", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/6, 3)} +/- 0.16"],
}

df_1_tdrdepth_1 = pd.DataFrame(data_1_tdrdepth_1)

def render_mpl_table(data_1_tdrdepth_1, col_width=2.8, row_height=0.6, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_1_tdrdepth_1.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(8,12))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_1_tdrdepth_1.values, bbox=bbox, colLabels=data_1_tdrdepth_1.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            cell.set_height(0.6)  # Set the height of header row to 0.6
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.set_height(0.5)  # Set the height of non-header rows to 0.2
    ax.set_title('10cm, 25cm Weighted Neutron Reduction')
    return ax

render_mpl_table(df_1_tdrdepth_1, header_columns=0, col_width=2.8)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Update_Summary_Table_with_TDR_twoweights.png', dpi=300, bbox_inches='tight')
plt.show()


#######Combine into one plot and table
#Plot data with best fit line, slope, and r-squared
a_CSP1, b_CSP1 = np.polyfit(x_CSP1,y_CSP1,1)
c_CSP1, d_CSP1 = np.polyfit(x_CSP1,y_baatz_CSP1,1)
a_CSP1_arith, b_CSP1_arith = np.polyfit(x_CSP1_arith,y_CSP1_arith,1)
a_CSP1_tdrdepth, b_CSP1_tdrdepth = np.polyfit(x_CSP1_tdrdepth,y_CSP1_tdrdepth,1)
plt.scatter(x_CSP1, y_CSP1, marker = 'o', color = 'red', label = r'$f_b$ Schrön Weight')
plt.scatter(x_CSP1_arith, y_CSP1_arith, marker = 'o', color = 'green', label = r'$f_b$ Arithmetic Weight')
plt.scatter(x_CSP1_tdrdepth, y_CSP1_tdrdepth, marker = 'o', color = 'gold', label = r'$f_b$ 10, 25cm Weight')
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP1, 0.005*x_CSP1 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
plt.plot(x_CSP1, a_CSP1*x_CSP1+b_CSP1, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP1_arith, a_CSP1_arith*x_CSP1_arith+b_CSP1_arith, linestyle = '-', color = 'green', linewidth=2.0)
plt.plot(x_CSP1_tdrdepth, a_CSP1_tdrdepth*x_CSP1_tdrdepth+b_CSP1_tdrdepth, linestyle = '-', color = 'gold')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP1 '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.95, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['TDR', 'Schrön', 'Arithmetic', '10, 25cm']
row_labels = ['Slope', 'r\u00b2']
table_vals = [
    [round(slope_CSP1,2), r_mm_CSP1, r_mm_CSP1_arith, r_mm_CSP1_tdrdepth], 
    [round(r_squared_CSP1, 2), r2_CSP1, r2_CSP1_arith, r2_CSP1_tdrdepth]  # r^2
]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 4,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Compare_All_Grav_Methods_CSP1.png', dpi=300, bbox_inches='tight')
plt.show()


#Plot data with best fit line, slope, and r-squared
a_CSP2, b_CSP2 = np.polyfit(x_CSP2,y_CSP2,1)
c_CSP2, d_CSP2 = np.polyfit(x_CSP2,y_baatz_CSP2,1)
a_CSP2_arith, b_CSP2_arith = np.polyfit(x_CSP2_arith,y_CSP2_arith,1)
a_CSP2_tdrdepth, b_CSP2_tdrdepth = np.polyfit(x_CSP2_tdrdepth,y_CSP2_tdrdepth,1)
plt.scatter(x_CSP2, y_CSP2, marker = 'o', color = 'red', label = r'$f_b$ Schrön Weight')
plt.scatter(x_CSP2_arith, y_CSP2_arith, marker = 'o', color = 'green', label = r'$f_b$ Arithmetic Weight')
plt.scatter(x_CSP2_tdrdepth, y_CSP2_tdrdepth, marker = 'o', color = 'gold', label = r'$f_b$ 10, 25cm Weight')
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP2, 0.005*x_CSP2 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
plt.plot(x_CSP2, a_CSP2*x_CSP2+b_CSP2, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP2_arith, a_CSP2_arith*x_CSP2_arith+b_CSP2_arith, linestyle = '-', color = 'green', linewidth=2.0)
plt.plot(x_CSP2_tdrdepth, a_CSP2_tdrdepth*x_CSP2_tdrdepth+b_CSP2_tdrdepth, linestyle = '-', color = 'gold')
plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP2 '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.95, 1.2)
plt.xlim(-0.25, 7.5)

#Add confidence table to the plot
col_labels = ['TDR', 'Schrön', 'Arithmetic', '10, 25cm']
row_labels = ['Slope', 'r\u00b2']
table_vals = [
    [round(slope_CSP2,2), r_mm_CSP2, r_mm_CSP2_arith, r_mm_CSP2_tdrdepth], 
    [round(r_squared_CSP2, 2), r2_CSP2, r2_CSP2_arith, r2_CSP2_tdrdepth]  # r^2
]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 4,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Compare_All_Grav_Methods_CSP2.png', dpi=300, bbox_inches='tight')
plt.show()


#Plot data with best fit line, slope, and r-squared
a_CSP3, b_CSP3 = np.polyfit(x_CSP3,y_CSP3,1)
c_CSP3, d_CSP3 = np.polyfit(x_CSP3,y_baatz_CSP3,1)
a_CSP3_arith, b_CSP3_arith = np.polyfit(x_CSP3_arith,y_CSP3_arith,1)
a_CSP3_tdrdepth, b_CSP3_tdrdepth = np.polyfit(x_CSP3_tdrdepth,y_CSP3_tdrdepth,1)
plt.scatter(x_CSP3, y_CSP3, marker = 'o', color = 'red', label = r'$f_b$ Schrön Weight')
plt.scatter(x_CSP3_arith, y_CSP3_arith, marker = 'o', color = 'green', label = r'$f_b$ Arithmetic Weight')
plt.scatter(x_CSP3_tdrdepth, y_CSP3_tdrdepth, marker = 'o', color = 'gold', label = r'$f_b$ 10, 25cm Weight')
plt.plot(x_all_CSP3, slope_CSP3*x_all_CSP3 + 1, color='black', label='TDR Best fit', linewidth=2.0)
plt.plot(x_CSP3, 0.005*x_CSP3 + 1, linestyle = '-', color = 'blue', linewidth=2.0,label = r'$f_b$ by Baatz et al. (2015)')
plt.plot(x_CSP3, a_CSP3*x_CSP3+b_CSP3, linestyle = '-', color = 'red', linewidth=2.0)
plt.plot(x_CSP3_arith, a_CSP3_arith*x_CSP3_arith+b_CSP3_arith, linestyle = '-', color = 'green', linewidth=2.0)
plt.plot(x_CSP3_tdrdepth, a_CSP3_tdrdepth*x_CSP3_tdrdepth+b_CSP3_tdrdepth, linestyle = '-', color = 'gold')

plt.xlabel('BWE (mm)')
plt.ylabel(r'$f_b$')
plt.title('CSP3 '+ r'$f_b$')
plt.grid(which='major', zorder=-2, alpha=0.5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.95, 1.2)
plt.xlim(-0.25, 7.5)

col_labels = ['TDR', 'Schrön', 'Arithmetic', '10, 25cm']
row_labels = ['Slope', 'r\u00b2']
table_vals = [
    [round(slope_CSP3,2), r_mm_CSP3, r_mm_CSP3_arith, r_mm_CSP3_tdrdepth], 
    [round(r_squared_CSP3, 2), r2_CSP3, r2_CSP3_arith, r2_CSP3_tdrdepth]  # r^2
]
table = plt.table(cellText=table_vals,
                  colWidths=[0.12] * 4,
                  colLabels=col_labels,
                  rowLabels=row_labels,
                  loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Compare_All_Grav_Methods_CSP3.png', dpi=300, bbox_inches='tight')
plt.show()


data_1_tdrdepth_1 = {
    '': ['CSP1 Schrön', 'CSP2 Schrön', 'CSP3 Schrön', 'CSP1 Arithmetic', 'CSP2 Arithmetic', 'CSP3 Arithmetic', 'CSP1 10, 25cm', 'CSP2 10, 25cm', 'CSP3 10, 25cm', 'CRS-2000/B\nAverage', 'CRS-1000/B\nAverage', 'Gravimetric\nAverage'],
    'Neutron Intensity\nReduction\n(% per mm BWE)': [f"{round((normal_CSP1),3)}", f"{round((normal_CSP2),3)}", f"{round((normal_CSP3),3)}", f"{round((normal_CSP1_arith),3)}", f"{round(normal_CSP2_arith, 3)}", f"{round(normal_CSP3_arith, 3)}", f"{round((normal_CSP1_tdrdepth),3)}", f"{round((normal_CSP2_tdrdepth),3)}", f"{round((normal_CSP3_tdrdepth),3)}", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP1+normal_CSP2+normal_CSP1_arith+normal_CSP2_arith)/6, 3)} +/- 0.28", f"{round((normal_CSP3_tdrdepth+normal_CSP3_arith+normal_CSP3)/3, 3)} +/- 0.15", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth+normal_CSP1+normal_CSP2+normal_CSP3+normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith)/9, 3)} +/- 0.32"],
    'Slope\nRatio': [f"{round((r_mm_CSP1/r_mm_CSP1_baatz),3)}", f"{round((r_mm_CSP2/r_mm_CSP2_baatz),3)}", f"{round((r_mm_CSP3/r_mm_CSP3_baatz),3)}", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith),3)}", f"{round((r_mm_CSP2_arith/r_mm_CSP2_baatz_arith),3)}", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith),3)}", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth),3)}", f"{round((r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth),3)}", f"{round((r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth),3)}", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth)/6,3)} +/- 0.10", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+r_mm_CSP3/r_mm_CSP3_baatz+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth)/3,3)} +/- 0.21", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP3/r_mm_CSP3_baatz+r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth)/9,3)} +/- 0.10"],
}

df_1_tdrdepth_1 = pd.DataFrame(data_1_tdrdepth_1)

def render_mpl_table(data_1_tdrdepth_1, col_width=2.8, row_height=0.6, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_1_tdrdepth_1.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(8,12))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_1_tdrdepth_1.values, bbox=bbox, colLabels=data_1_tdrdepth_1.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            cell.set_height(0.6)  # Set the height of header row to 0.6
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.set_height(0.5)  # Set the height of non-header rows to 0.2
    ax.set_title('Neutron Reduction by All Gravimetric Methods')
    return ax

render_mpl_table(df_1_tdrdepth_1, header_columns=0, col_width=2.8)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Compare_All_Grav_Methods_Table.png', dpi=300, bbox_inches='tight')
plt.show()



data_1_tdrdepth_1 = {
    '': ['CSP1 Schrön', 'CSP2 Schrön', 'CSP3 Schrön', 'CSP1 Arithmetic', 'CSP2 Arithmetic', 'CSP3 Arithmetic', 'CSP1 10, 25cm', 'CSP2 10, 25cm', 'CSP3 10, 25cm', 'CSP1 TDR', 'CSP2 TDR', 'CSP3 TDR','CRS-2000/B\nAverage', 'CRS-1000/B\nAverage', 'Gravimetric\nAverage', 'TDR Average', 'Overall\nAverage'],
    'Neutron Intensity\nReduction\n(% per mm BWE)': [f"{round((normal_CSP1),3)}", f"{round((normal_CSP2),3)}", f"{round((normal_CSP3),3)}", f"{round((normal_CSP1_arith),3)}", f"{round(normal_CSP2_arith, 3)}", f"{round(normal_CSP3_arith, 3)}", f"{round((normal_CSP1_tdrdepth),3)}", f"{round((normal_CSP2_tdrdepth),3)}", f"{round((normal_CSP3_tdrdepth),3)}", f"{round((normal_CSP1_all),3)}", f"{round(normal_CSP2_all, 3)}", f"{round(normal_CSP3_all, 3)}", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP1+normal_CSP2+normal_CSP1_arith+normal_CSP2_arith+normal_CSP1_all+normal_CSP2_all)/8, 3)} +/- 0.24", f"{round((normal_CSP3_tdrdepth+normal_CSP3_arith+normal_CSP3+normal_CSP3_all)/4, 3)} +/- 0.11", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth+normal_CSP1+normal_CSP2+normal_CSP3+normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith)/9, 3)} +/- 0.32", f"{round((normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/3, 3)} +/- 0.30", f"{round((normal_CSP1_tdrdepth+normal_CSP2_tdrdepth+normal_CSP3_tdrdepth+normal_CSP1+normal_CSP2+normal_CSP3+normal_CSP1_arith+normal_CSP2_arith+normal_CSP3_arith+normal_CSP1_all+normal_CSP2_all+normal_CSP3_all)/12, 3)} +/- 0.25"],
    'Slope\nRatio': [f"{round((r_mm_CSP1/r_mm_CSP1_baatz),3)}", f"{round((r_mm_CSP2/r_mm_CSP2_baatz),3)}", f"{round((r_mm_CSP3/r_mm_CSP3_baatz),3)}", f"{round((r_mm_CSP1_arith/r_mm_CSP1_baatz_arith),3)}", f"{round((r_mm_CSP2_arith/r_mm_CSP2_baatz_arith),3)}", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith),3)}", f"{round((r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth),3)}", f"{round((r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth),3)}", f"{round((r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth),3)}", f"{round(slope_CSP1/r_mm_CSP1_baatz, 3)}", f"{round(slope_CSP2/r_mm_CSP2_baatz, 3)}", f"{round(slope_CSP3/r_mm_CSP3_baatz, 3)}", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz)/8,3)} +/- 0.14", f"{round((r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+r_mm_CSP3/r_mm_CSP3_baatz+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth+slope_CSP3/r_mm_CSP3_baatz)/4,3)} +/- 0.16", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP3/r_mm_CSP3_baatz+r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth)/9,3)} +/- 0.10", f"{round((slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/3,3)} +/- 0.07", f"{round((r_mm_CSP1/r_mm_CSP1_baatz+r_mm_CSP2/r_mm_CSP2_baatz+r_mm_CSP3/r_mm_CSP3_baatz+r_mm_CSP1_arith/r_mm_CSP1_baatz_arith+r_mm_CSP2_arith/r_mm_CSP2_baatz_arith+r_mm_CSP3_arith/r_mm_CSP3_baatz_arith+r_mm_CSP1_tdrdepth/r_mm_CSP1_baatz_tdrdepth+r_mm_CSP2_tdrdepth/r_mm_CSP2_baatz_tdrdepth+r_mm_CSP3_tdrdepth/r_mm_CSP3_baatz_tdrdepth+slope_CSP1/r_mm_CSP1_baatz+slope_CSP2/r_mm_CSP2_baatz+slope_CSP3/r_mm_CSP3_baatz)/12,3)} +/- 0.11"],
}

df_1_tdrdepth_1 = pd.DataFrame(data_1_tdrdepth_1)

def render_mpl_table(data_1_tdrdepth_1, col_width=2.8, row_height=0.6, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data_1_tdrdepth_1.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(8,14))
        ax.axis('off')

    mpl_table = ax.table(cellText=data_1_tdrdepth_1.values, bbox=bbox, colLabels=data_1_tdrdepth_1.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            cell.set_height(0.6)  # Set the height of header row to 0.6
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.set_height(0.5)  # Set the height of non-header rows to 0.2
    ax.set_title('Neutron Reduction by All Gravimetric Methods with TDR')
    return ax

render_mpl_table(df_1_tdrdepth_1, header_columns=0, col_width=2.8)

# Save the table as an image
plt.savefig('YOUR PATH TO SAVE/Compare_All_Grav_TDR_Methods_Table.png', dpi=300, bbox_inches='tight')
plt.show()