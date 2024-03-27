#Plotting all years

#import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

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

N19_CSP1 = dailyNPIV2019_CSP1
N20_CSP1 = dailyNPIV2020_CSP1
N21_CSP1 = dailyNPIV2021_CSP1
N22_CSP1 = dailyNPIV2022_CSP1
N23_CSP1 = dailyNPIV2023_CSP1

#use parameters to calculate N0
N019_CSP1 = N19_CSP1 / ((0.0808 / ((pwc19_CSP1 + lwc + soc) + 0.115)) + 0.372)
N020_CSP1 = N20_CSP1 / ((0.0808 / ((pwc20_CSP1 + lwc + soc) + 0.115)) + 0.372)
N021_CSP1 = N21_CSP1 / ((0.0808 / ((pwc21_CSP1 + lwc + soc) + 0.115)) + 0.372)
N022_CSP1 = N22_CSP1 / ((0.0808 / ((pwc22_CSP1 + lwc + soc) + 0.115)) + 0.372)
N023_CSP1 = N23_CSP1 / ((0.0808 / ((pwc23_CSP1 + lwc + soc) + 0.115)) + 0.372)

N0interp19_CSP1 = N019_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP1 = N020_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP1 = N021_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP1 = N022_CSP1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP1 = N023_CSP1.interpolate(method = 'linear', limit_direction = 'forward')

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2019_CSP1 = pd.read_csv('YOUR PATH/CSP12019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2020_CSP1 = pd.read_csv('YOUR PATH/CSP12020BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP1 = pd.read_csv('YOUR PATH/CSP12021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP1 = pd.read_csv('YOUR PATH/CSP12022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP1 = pd.read_csv('YOUR PATH/CSP12023BWE_plotdate.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

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

#BWEplotdata = BWEinterp.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata19_CSP1 = N019_CSP1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata20_CSP1 = N020_CSP1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata21_CSP1 = N021_CSP1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata22_CSP1 = N022_CSP1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata23_CSP1 = N023_CSP1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP1 = BWEinterp19_CSP1
y19_CSP1 = N0interp19_CSP1
x20_CSP1 = BWEinterp20_CSP1
y20_CSP1 = N0interp20_CSP1
x21_CSP1 = BWEinterp21_CSP1
y21_CSP1 = N0interp21_CSP1
x22_CSP1 = BWEinterp22_CSP1
y22_CSP1 = N0interp22_CSP1
x23_CSP1 = BWEinterp23_CSP1
y23_CSP1 = N0interp23_CSP1

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

plt.scatter(x19_CSP1, y19_CSP1, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP1, y20_CSP1, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP1, y21_CSP1, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP1, y22_CSP1, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP1, y23_CSP1, color = 'darkorange', marker = 'D', label = '2023', alpha=0.5)

# Plot the line of best fit
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + intercept_CSP1, color='black', label='Best fit', linewidth=2.5)

plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.text(-0.2, 2570, u"Best fit line: r\u00b2 = {:0.2f}".format(r_squared_CSP1), size=9)
plt.text(1, 2540, "p < 0.001", size=9)
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2500, 3375)
plt.xlim(-0.3,7.2)

#Round values for the table
r_slope_CSP1 = round(slope_CSP1, 2)
r_intercept_CSP1 = round(intercept_CSP1, 2)
r_slope_conf_interval_CSP1 = round(slope_conf_interval_CSP1, 2)
r_intercept_conf_interval_CSP1 = round(intercept_conf_interval_CSP1, 2)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_slope_CSP1,r_intercept_CSP1], [r_slope_conf_interval_CSP1,r_intercept_conf_interval_CSP1]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_no_gravimetric.png', dpi=300, bbox_inches='tight')
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

#use parameters to calculate N0
N019_CSP2 = N19_CSP2 / ((0.0808 / ((pwc19_CSP2 + lwc + soc) + 0.115)) + 0.372)
N020_CSP2 = N20_CSP2 / ((0.0808 / ((pwc20_CSP2 + lwc + soc) + 0.115)) + 0.372)
N021_CSP2 = N21_CSP2 / ((0.0808 / ((pwc21_CSP2 + lwc + soc) + 0.115)) + 0.372)
N022_CSP2 = N22_CSP2 / ((0.0808 / ((pwc22_CSP2 + lwc + soc) + 0.115)) + 0.372)
N023_CSP2 = N23_CSP2 / ((0.0808 / ((pwc23_CSP2 + lwc + soc) + 0.115)) + 0.372)

N0interp19_CSP2 = N019_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP2 = N020_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP2 = N021_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP2 = N022_CSP2.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP2 = N023_CSP2.interpolate(method = 'linear', limit_direction = 'forward')

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

#BWEplotdata = BWEinterp.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata19_CSP2 = N019_CSP2.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata20_CSP2 = N020_CSP2.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata21_CSP2 = N021_CSP2.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata22_CSP2 = N022_CSP2.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata23_CSP2 = N023_CSP2.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x19_CSP2 = BWEinterp19_CSP2
y19_CSP2 = N0interp19_CSP2
x20_CSP2 = BWEinterp20_CSP2
y20_CSP2 = N0interp20_CSP2
x21_CSP2 = BWEinterp21_CSP2
y21_CSP2 = N0interp21_CSP2
x22_CSP2 = BWEinterp22_CSP2
y22_CSP2 = N0interp22_CSP2
x23_CSP2 = BWEinterp23_CSP2
y23_CSP2 = N0interp23_CSP2

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

plt.scatter(x19_CSP2, y19_CSP2, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP2, y20_CSP2, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP2, y21_CSP2, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP2, y22_CSP2, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP2, y23_CSP2, color = 'darkorange', marker = 'D', label = '2023', alpha=0.5)

# Plot the line of best fit
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + intercept_CSP2, color='black', label='Best fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.text(-0.2, 2570, u"Best fit line: r\u00b2 = {:0.2f}".format(r_squared_CSP2), size=9)
plt.text(1, 2540, f"p < 0.001", size=9)
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2500, 3375)
plt.xlim(-0.3,7.2)

#Round values for the table
r_slope_CSP2 = round(slope_CSP2, 2)
r_intercept_CSP2 = round(intercept_CSP2, 2)
r_slope_conf_interval_CSP2 = round(slope_conf_interval_CSP2, 2)
r_intercept_conf_interval_CSP2 = round(intercept_conf_interval_CSP2, 2)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_slope_CSP2,r_intercept_CSP2], [r_slope_conf_interval_CSP2,r_intercept_conf_interval_CSP2]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_no_grav.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
########################2020 Included#############################
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
GWC2022_CSP3 = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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

#use parameters to calculate N0
N011_CSP3 = N11_CSP3 / ((0.0808 / ((pwc11_CSP3 + lwc + soc) + 0.115)) + 0.372)
N012_CSP3 = N12_CSP3 / ((0.0808 / ((pwc12_CSP3 + lwc + soc) + 0.115)) + 0.372)
N013_CSP3 = N13_CSP3 / ((0.0808 / ((pwc13_CSP3 + lwc + soc) + 0.115)) + 0.372)
N014_CSP3 = N14_CSP3 / ((0.0808 / ((pwc14_CSP3 + lwc + soc) + 0.115)) + 0.372)
N015_CSP3 = N15_CSP3 / ((0.0808 / ((pwc15_CSP3 + lwc + soc) + 0.115)) + 0.372)
N016_CSP3 = N16_CSP3 / ((0.0808 / ((pwc16_CSP3 + lwc + soc) + 0.115)) + 0.372)
N017_CSP3 = N17_CSP3 / ((0.0808 / ((pwc17_CSP3 + lwc + soc) + 0.115)) + 0.372)
N018_CSP3 = N18_CSP3 / ((0.0808 / ((pwc18_CSP3 + lwc + soc) + 0.115)) + 0.372)
N019_CSP3 = N19_CSP3 / ((0.0808 / ((pwc19_CSP3 + lwc + soc) + 0.115)) + 0.372)
N020_CSP3 = N20_CSP3 / ((0.0808 / ((pwc20_CSP3 + lwc + soc) + 0.115)) + 0.372)
N021_CSP3 = N21_CSP3 / ((0.0808 / ((pwc21_CSP3 + lwc + soc) + 0.115)) + 0.372)
N022_CSP3 = N22_CSP3 / ((0.0808 / ((pwc22_CSP3 + lwc + soc) + 0.115)) + 0.372)
N023_CSP3 = N23_CSP3 / ((0.0808 / ((pwc23_CSP3 + lwc + soc) + 0.115)) + 0.372)

N0interp11_CSP3 = N011_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp12_CSP3 = N012_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp13_CSP3 = N013_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp14_CSP3 = N014_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp15_CSP3 = N015_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp16_CSP3 = N016_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp17_CSP3 = N017_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp18_CSP3 = N018_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp19_CSP3 = N019_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp20_CSP3 = N020_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP3 = N021_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP3 = N022_CSP3.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP3 = N023_CSP3.interpolate(method = 'linear', limit_direction = 'forward')

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
BWEcsv2023_CSP3 = pd.read_csv('YOUR PATH/CSP32023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

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

N0plotdata11_CSP3 = N011_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata12_CSP3 = N012_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata13_CSP3 = N013_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata14_CSP3 = N014_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata15_CSP3 = N015_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata16_CSP3 = N016_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata17_CSP3 = N017_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata18_CSP3 = N018_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata19_CSP3 = N019_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata20_CSP3 = N020_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata21_CSP3 = N021_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata22_CSP3 = N022_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata23_CSP3 = N023_CSP3.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x11_CSP3 = BWEinterp11_CSP3
y11_CSP3 = N0interp11_CSP3
x12_CSP3 = BWEinterp12_CSP3
y12_CSP3 = N0interp12_CSP3
x13_CSP3 = BWEinterp13_CSP3
y13_CSP3 = N0interp13_CSP3
x14_CSP3 = BWEinterp14_CSP3
y14_CSP3 = N0interp14_CSP3
x15_CSP3 = BWEinterp15_CSP3
y15_CSP3 = N0interp15_CSP3
x16_CSP3 = BWEinterp16_CSP3
y16_CSP3 = N0interp16_CSP3
x17_CSP3 = BWEinterp17_CSP3
y17_CSP3 = N0interp17_CSP3
x18_CSP3 = BWEinterp18_CSP3
y18_CSP3 = N0interp18_CSP3
x19_CSP3 = BWEinterp19_CSP3
y19_CSP3 = N0interp19_CSP3
x20_CSP3 = BWEinterp20_CSP3
y20_CSP3 = N0interp20_CSP3
x21_CSP3 = BWEinterp21_CSP3
y21_CSP3 = N0interp21_CSP3
x22_CSP3 = BWEinterp22_CSP3
y22_CSP3 = N0interp22_CSP3
x23_CSP3 = BWEinterp23_CSP3
y23_CSP3 = N0interp23_CSP3

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

#plot the data
plt.scatter(x11_CSP3, y11_CSP3, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.5)
plt.scatter(x12_CSP3, y12_CSP3, color = 'indianred', marker = 'v', label = '2012', alpha=0.5)
plt.scatter(x13_CSP3, y13_CSP3, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.5)
plt.scatter(x14_CSP3, y14_CSP3, color = 'peru', marker = 's', label = '2014', alpha=0.5)
plt.scatter(x15_CSP3, y15_CSP3, color = 'gold', marker = '<', label = '2015', alpha=0.5)
plt.scatter(x16_CSP3, y16_CSP3, color = 'olivedrab', marker = '>', label = '2016', alpha=0.5)
plt.scatter(x17_CSP3, y17_CSP3, color = 'lawngreen', marker = '.', label = '2017', alpha=0.5)
plt.scatter(x18_CSP3, y18_CSP3, color = 'fuchsia', marker = '1', label = '2018', alpha=0.5)
plt.scatter(x19_CSP3, y19_CSP3, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP3, y20_CSP3, color = 'aquamarine', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP3, y21_CSP3, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP3, y22_CSP3, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP3, y23_CSP3, color = 'darkorange', marker = 'p', label = '2023', alpha=0.5)

# Plot the line of best fit
plt.plot(x_all_CSP3, slope_CSP3*x_all_CSP3 + intercept_CSP3, color='black', label='Best fit', linewidth=2.5)
plt.gca().set_axisbelow(True)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE')
plt.text(-0.2, 1420, u"Best fit line: r\u00b2 = {:0.2f}".format(r_squared_CSP3), size=9)
plt.text(0.85, 1400, "p < 0.001", size=9)
plt.grid(which='major', zorder=-1, alpha=0.5)

#plt.rcParams['figure.figsize'] = [10,6]
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')

#Round values for the table
r_slope_CSP3 = round(slope_CSP3, 2)
r_intercept_CSP3 = round(intercept_CSP3, 2)
r_slope_conf_interval_CSP3 = round(slope_conf_interval_CSP3, 2)
r_intercept_conf_interval_CSP3 = round(intercept_conf_interval_CSP3, 2)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_slope_CSP3,r_intercept_CSP3], [r_slope_conf_interval_CSP3,r_intercept_conf_interval_CSP3]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_no_grav')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
#######################2020 NOT Included##########################
##################################################################
##################################################################

### DATA AND CALCULATIONS FOR SWC ###
#import spreadsheet
GWC2011_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2012_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2013_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2014_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2015_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2016_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2017_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2018_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2019_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2021_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2022_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
GWC2023_CSP3_1 = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#Bulk Density=1.42g/cm^3
SWC11_PI_F_1_1_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC11_PI_F_1_2_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC11_PI_F_1_3_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC11_PI_F_1_4_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_5_1']/1.42
SWC11_PI_F_2_1_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC11_PI_F_2_2_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC11_PI_F_2_3_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC11_PI_F_2_4_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_5_1']/1.42
SWC11_PI_F_3_1_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC11_PI_F_3_2_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC11_PI_F_3_3_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC11_PI_F_3_4_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_3_4_1']/1.42
SWC11_PI_F_1_5_1_CSP3_1 = GWC2011_CSP3_1['SWC_PI_F_1_5_1']/1.42

SWC12_PI_F_1_1_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC12_PI_F_1_2_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC12_PI_F_1_3_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC12_PI_F_1_4_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC12_PI_F_2_1_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC12_PI_F_2_2_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC12_PI_F_2_3_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC12_PI_F_2_4_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC12_PI_F_3_1_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC12_PI_F_3_2_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC12_PI_F_3_3_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC12_PI_F_3_4_1_CSP3_1 = GWC2012_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC13_PI_F_1_1_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC13_PI_F_1_2_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC13_PI_F_1_3_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC13_PI_F_1_4_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC13_PI_F_2_1_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC13_PI_F_2_2_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC13_PI_F_2_3_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC13_PI_F_2_4_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC13_PI_F_3_1_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC13_PI_F_3_2_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC13_PI_F_3_3_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC13_PI_F_3_4_1_CSP3_1 = GWC2013_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC14_PI_F_1_1_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC14_PI_F_1_2_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC14_PI_F_1_3_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC14_PI_F_1_4_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC14_PI_F_2_1_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC14_PI_F_2_2_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC14_PI_F_2_3_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC14_PI_F_2_4_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC14_PI_F_3_1_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC14_PI_F_3_2_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC14_PI_F_3_3_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC14_PI_F_3_4_1_CSP3_1 = GWC2014_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC15_PI_F_1_1_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC15_PI_F_1_2_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC15_PI_F_1_3_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC15_PI_F_1_4_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC15_PI_F_2_1_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC15_PI_F_2_2_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC15_PI_F_2_3_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC15_PI_F_2_4_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC15_PI_F_3_1_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC15_PI_F_3_2_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC15_PI_F_3_3_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC15_PI_F_3_4_1_CSP3_1 = GWC2015_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC16_PI_F_1_1_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC16_PI_F_1_2_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC16_PI_F_1_3_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC16_PI_F_1_4_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC16_PI_F_2_1_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC16_PI_F_2_2_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC16_PI_F_2_3_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC16_PI_F_2_4_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC16_PI_F_3_1_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC16_PI_F_3_2_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC16_PI_F_3_3_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC16_PI_F_3_4_1_CSP3_1 = GWC2016_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC17_PI_F_1_1_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC17_PI_F_1_2_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC17_PI_F_1_3_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC17_PI_F_1_4_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC17_PI_F_2_1_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC17_PI_F_2_2_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC17_PI_F_2_3_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC17_PI_F_2_4_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC17_PI_F_3_1_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC17_PI_F_3_2_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC17_PI_F_3_3_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC17_PI_F_3_4_1_CSP3_1 = GWC2017_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC18_PI_F_1_1_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC18_PI_F_1_2_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC18_PI_F_1_3_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC18_PI_F_1_4_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC18_PI_F_2_1_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC18_PI_F_2_2_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC18_PI_F_2_3_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC18_PI_F_2_4_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC18_PI_F_3_1_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC18_PI_F_3_2_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC18_PI_F_3_3_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC18_PI_F_3_4_1_CSP3_1 = GWC2018_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC19_PI_F_1_1_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC19_PI_F_1_2_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC19_PI_F_1_3_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC19_PI_F_1_4_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC19_PI_F_2_1_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC19_PI_F_2_2_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC19_PI_F_2_3_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC19_PI_F_2_4_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC19_PI_F_3_1_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC19_PI_F_3_2_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC19_PI_F_3_3_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC19_PI_F_3_4_1_CSP3_1 = GWC2019_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC21_PI_F_1_1_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_1_1_1']/1.42
SWC21_PI_F_1_2_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_1_2_1']/1.42
SWC21_PI_F_1_3_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_1_3_1']/1.42
SWC21_PI_F_1_4_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_1_4_1']/1.42
SWC21_PI_F_2_1_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_2_1_1']/1.42
SWC21_PI_F_2_2_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_2_2_1']/1.42
SWC21_PI_F_2_3_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_2_3_1']/1.42
SWC21_PI_F_2_4_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_2_4_1']/1.42
SWC21_PI_F_3_1_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_3_1_1']/1.42
SWC21_PI_F_3_2_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_3_2_1']/1.42
SWC21_PI_F_3_3_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_3_3_1']/1.42
SWC21_PI_F_3_4_1_CSP3_1 = GWC2021_CSP3_1['SWC_PI_F_3_4_1']/1.42

SWC22_PI_F_1_1_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_1_1_1']/1.42
SWC22_PI_F_1_2_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_1_2_1']/1.42
SWC22_PI_F_1_3_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_1_3_1']/1.42
SWC22_PI_F_1_4_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_1_4_1']/1.42
SWC22_PI_F_2_1_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_2_1_1']/1.42
SWC22_PI_F_2_2_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_2_2_1']/1.42
SWC22_PI_F_2_3_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_2_3_1']/1.42
SWC22_PI_F_2_4_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_2_4_1']/1.42
SWC22_PI_F_3_1_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_3_1_1']/1.42
SWC22_PI_F_3_2_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_3_2_1']/1.42
SWC22_PI_F_3_3_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_3_3_1']/1.42
SWC22_PI_F_3_4_1_CSP3_1 = GWC2022_CSP3_1['SWC_F_3_4_1']/1.42

SWC23_PI_F_1_1_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_1_1_1']/1.42
SWC23_PI_F_1_2_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_1_2_1']/1.42
SWC23_PI_F_1_3_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_1_3_1']/1.42
SWC23_PI_F_1_4_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_1_4_1']/1.42
SWC23_PI_F_2_1_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_2_1_1']/1.42
SWC23_PI_F_2_2_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_2_2_1']/1.42
SWC23_PI_F_2_3_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_2_3_1']/1.42
SWC23_PI_F_2_4_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_2_4_1']/1.42
SWC23_PI_F_3_1_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_3_1_1']/1.42
SWC23_PI_F_3_2_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_3_2_1']/1.42
SWC23_PI_F_3_3_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_3_3_1']/1.42
SWC23_PI_F_3_4_1_CSP3_1 = GWC2023_CSP3_1['SWC_F_3_4_1']/1.42

twosenwavgall2011_CSP3_1 = ((((SWC11_PI_F_1_1_1_CSP3_1 + SWC11_PI_F_2_1_1_CSP3_1 + SWC11_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC11_PI_F_1_2_1_CSP3_1 + SWC11_PI_F_2_2_1_CSP3_1 + SWC11_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2012_CSP3_1 = ((((SWC12_PI_F_1_1_1_CSP3_1 + SWC12_PI_F_2_1_1_CSP3_1 + SWC12_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC12_PI_F_1_2_1_CSP3_1 + SWC12_PI_F_2_2_1_CSP3_1 + SWC12_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2013_CSP3_1 = ((((SWC13_PI_F_1_1_1_CSP3_1 + SWC13_PI_F_2_1_1_CSP3_1 + SWC13_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC13_PI_F_1_2_1_CSP3_1 + SWC13_PI_F_2_2_1_CSP3_1 + SWC13_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2014_CSP3_1 = ((((SWC14_PI_F_1_1_1_CSP3_1 + SWC14_PI_F_2_1_1_CSP3_1 + SWC14_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC14_PI_F_1_2_1_CSP3_1 + SWC14_PI_F_2_2_1_CSP3_1 + SWC14_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2015_CSP3_1 = ((((SWC15_PI_F_1_1_1_CSP3_1 + SWC15_PI_F_2_1_1_CSP3_1 + SWC15_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC15_PI_F_1_2_1_CSP3_1 + SWC15_PI_F_2_2_1_CSP3_1 + SWC15_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2016_CSP3_1 = ((((SWC16_PI_F_1_1_1_CSP3_1 + SWC16_PI_F_2_1_1_CSP3_1 + SWC16_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC16_PI_F_1_2_1_CSP3_1 + SWC16_PI_F_2_2_1_CSP3_1 + SWC16_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2017_CSP3_1 = ((((SWC17_PI_F_1_1_1_CSP3_1 + SWC17_PI_F_2_1_1_CSP3_1 + SWC17_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC17_PI_F_1_2_1_CSP3_1 + SWC17_PI_F_2_2_1_CSP3_1 + SWC17_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2018_CSP3_1 = ((((SWC18_PI_F_1_1_1_CSP3_1 + SWC18_PI_F_2_1_1_CSP3_1 + SWC18_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC18_PI_F_1_2_1_CSP3_1 + SWC18_PI_F_2_2_1_CSP3_1 + SWC18_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2019_CSP3_1 = ((((SWC19_PI_F_1_1_1_CSP3_1 + SWC19_PI_F_2_1_1_CSP3_1 + SWC19_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC19_PI_F_1_2_1_CSP3_1 + SWC19_PI_F_2_2_1_CSP3_1 + SWC19_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2021_CSP3_1 = ((((SWC21_PI_F_1_1_1_CSP3_1 + SWC21_PI_F_2_1_1_CSP3_1 + SWC21_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC21_PI_F_1_2_1_CSP3_1 + SWC21_PI_F_2_2_1_CSP3_1 + SWC21_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2022_CSP3_1 = ((((SWC22_PI_F_1_1_1_CSP3_1 + SWC22_PI_F_2_1_1_CSP3_1 + SWC22_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC22_PI_F_1_2_1_CSP3_1 + SWC22_PI_F_2_2_1_CSP3_1 + SWC22_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100
twosenwavgall2023_CSP3_1 = ((((SWC23_PI_F_1_1_1_CSP3_1 + SWC23_PI_F_2_1_1_CSP3_1 + SWC23_PI_F_3_1_1_CSP3_1)/3)*0.75) + (((SWC23_PI_F_1_2_1_CSP3_1 + SWC23_PI_F_2_2_1_CSP3_1 + SWC23_PI_F_3_2_1_CSP3_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above 
dailywavgSWC2011_CSP3_1 = twosenwavgall2011_CSP3_1.resample('D').mean() 
dailywavgSWC2012_CSP3_1 = twosenwavgall2012_CSP3_1.resample('D').mean() 
dailywavgSWC2013_CSP3_1 = twosenwavgall2013_CSP3_1.resample('D').mean() 
dailywavgSWC2014_CSP3_1 = twosenwavgall2014_CSP3_1.resample('D').mean()
dailywavgSWC2015_CSP3_1 = twosenwavgall2015_CSP3_1.resample('D').mean() 
dailywavgSWC2016_CSP3_1 = twosenwavgall2016_CSP3_1.resample('D').mean() 
dailywavgSWC2017_CSP3_1 = twosenwavgall2017_CSP3_1.resample('D').mean() 
dailywavgSWC2018_CSP3_1 = twosenwavgall2018_CSP3_1.resample('D').mean()
dailywavgSWC2019_CSP3_1 = twosenwavgall2019_CSP3_1.resample('D').mean()  
dailywavgSWC2021_CSP3_1 = twosenwavgall2021_CSP3_1.resample('D').mean() 
dailywavgSWC2022_CSP3_1 = twosenwavgall2022_CSP3_1.resample('D').mean() 
dailywavgSWC2023_CSP3_1 = twosenwavgall2023_CSP3_1.resample('D').mean()

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV2011_CSP3_1 = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2012_CSP3_1 = pd.read_csv('YOUR PATH/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2013_CSP3_1 = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2014_CSP3_1 = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2015_CSP3_1 = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2016_CSP3_1 = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2017_CSP3_1 = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2018_CSP3_1 = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2019_CSP3_1 = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2021_CSP3_1 = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2022_CSP3_1 = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
NPIV2023_CSP3_1 = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount2011_CSP3_1 = NPIV2011_CSP3_1['NPIV']
NPIVCount2012_CSP3_1 = NPIV2012_CSP3_1['NPIV']
NPIVCount2013_CSP3_1 = NPIV2013_CSP3_1['NPIV']
NPIVCount2014_CSP3_1 = NPIV2014_CSP3_1['NPIV']
NPIVCount2015_CSP3_1 = NPIV2015_CSP3_1['NPIV']
NPIVCount2016_CSP3_1 = NPIV2016_CSP3_1['NPIV']
NPIVCount2017_CSP3_1 = NPIV2017_CSP3_1['NPIV']
NPIVCount2018_CSP3_1 = NPIV2018_CSP3_1['NPIV']
NPIVCount2019_CSP3_1 = NPIV2019_CSP3_1['NPIV']
NPIVCount2021_CSP3_1 = NPIV2021_CSP3_1['NPIV']
NPIVCount2022_CSP3_1 = NPIV2022_CSP3_1['NPIV']
NPIVCount2023_CSP3_1 = NPIV2023_CSP3_1['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV2011_CSP3_1 = NPIVCount2011_CSP3_1.resample('D').mean()
dailyNPIV2012_CSP3_1 = NPIVCount2012_CSP3_1.resample('D').mean()
dailyNPIV2013_CSP3_1 = NPIVCount2013_CSP3_1.resample('D').mean()
dailyNPIV2014_CSP3_1 = NPIVCount2014_CSP3_1.resample('D').mean()
dailyNPIV2015_CSP3_1 = NPIVCount2015_CSP3_1.resample('D').mean()
dailyNPIV2016_CSP3_1 = NPIVCount2016_CSP3_1.resample('D').mean()
dailyNPIV2017_CSP3_1 = NPIVCount2017_CSP3_1.resample('D').mean()
dailyNPIV2018_CSP3_1 = NPIVCount2018_CSP3_1.resample('D').mean()
dailyNPIV2019_CSP3_1 = NPIVCount2019_CSP3_1.resample('D').mean()
dailyNPIV2021_CSP3_1 = NPIVCount2021_CSP3_1.resample('D').mean()
dailyNPIV2022_CSP3_1 = NPIVCount2022_CSP3_1.resample('D').mean()
dailyNPIV2023_CSP3_1 = NPIVCount2023_CSP3_1.resample('D').mean()

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc11_CSP3_1 = dailywavgSWC2011_CSP3_1
pwc12_CSP3_1 = dailywavgSWC2012_CSP3_1
pwc13_CSP3_1 = dailywavgSWC2013_CSP3_1
pwc14_CSP3_1 = dailywavgSWC2014_CSP3_1
pwc15_CSP3_1 = dailywavgSWC2015_CSP3_1
pwc16_CSP3_1 = dailywavgSWC2016_CSP3_1
pwc17_CSP3_1 = dailywavgSWC2017_CSP3_1
pwc18_CSP3_1 = dailywavgSWC2018_CSP3_1
pwc19_CSP3_1 = dailywavgSWC2019_CSP3_1
pwc21_CSP3_1 = dailywavgSWC2021_CSP3_1
pwc22_CSP3_1 = dailywavgSWC2022_CSP3_1
pwc23_CSP3_1 = dailywavgSWC2023_CSP3_1

N11_CSP3_1 = dailyNPIV2011_CSP3_1
N12_CSP3_1 = dailyNPIV2012_CSP3_1
N13_CSP3_1 = dailyNPIV2013_CSP3_1
N14_CSP3_1 = dailyNPIV2014_CSP3_1
N15_CSP3_1 = dailyNPIV2015_CSP3_1
N16_CSP3_1 = dailyNPIV2016_CSP3_1
N17_CSP3_1 = dailyNPIV2017_CSP3_1
N18_CSP3_1 = dailyNPIV2018_CSP3_1
N19_CSP3_1 = dailyNPIV2019_CSP3_1
N21_CSP3_1 = dailyNPIV2021_CSP3_1
N22_CSP3_1 = dailyNPIV2022_CSP3_1
N23_CSP3_1 = dailyNPIV2023_CSP3_1

#use parameters to calculate N0
N011_CSP3_1 = N11_CSP3_1 / ((0.0808 / ((pwc11_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N012_CSP3_1 = N12_CSP3_1 / ((0.0808 / ((pwc12_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N013_CSP3_1 = N13_CSP3_1 / ((0.0808 / ((pwc13_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N014_CSP3_1 = N14_CSP3_1 / ((0.0808 / ((pwc14_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N015_CSP3_1 = N15_CSP3_1 / ((0.0808 / ((pwc15_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N016_CSP3_1 = N16_CSP3_1 / ((0.0808 / ((pwc16_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N017_CSP3_1 = N17_CSP3_1 / ((0.0808 / ((pwc17_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N018_CSP3_1 = N18_CSP3_1 / ((0.0808 / ((pwc18_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N019_CSP3_1 = N19_CSP3_1 / ((0.0808 / ((pwc19_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N021_CSP3_1 = N21_CSP3_1 / ((0.0808 / ((pwc21_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N022_CSP3_1 = N22_CSP3_1 / ((0.0808 / ((pwc22_CSP3_1 + lwc + soc) + 0.115)) + 0.372)
N023_CSP3_1 = N23_CSP3_1 / ((0.0808 / ((pwc23_CSP3_1 + lwc + soc) + 0.115)) + 0.372)

N0interp11_CSP3_1 = N011_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp12_CSP3_1 = N012_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp13_CSP3_1 = N013_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp14_CSP3_1 = N014_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp15_CSP3_1 = N015_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp16_CSP3_1 = N016_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp17_CSP3_1 = N017_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp18_CSP3_1 = N018_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp19_CSP3_1 = N019_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp21_CSP3_1 = N021_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp22_CSP3_1 = N022_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
N0interp23_CSP3_1 = N023_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv2011_CSP3_1 = pd.read_csv('YOUR PATH/CSP32011BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2012_CSP3_1 = pd.read_csv('YOUR PATH/CSP32012BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2013_CSP3_1 = pd.read_csv('YOUR PATH/CSP32013BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2014_CSP3_1 = pd.read_csv('YOUR PATH/CSP32014BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2015_CSP3_1 = pd.read_csv('YOUR PATH/CSP32015BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2016_CSP3_1 = pd.read_csv('YOUR PATH/CSP32016BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2017_CSP3_1 = pd.read_csv('YOUR PATH/CSP32017BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2018_CSP3_1 = pd.read_csv('YOUR PATH/CSP32018BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2019_CSP3_1 = pd.read_csv('YOUR PATH/CSP32019BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2021_CSP3_1 = pd.read_csv('YOUR PATH/CSP32021BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2022_CSP3_1 = pd.read_csv('YOUR PATH/CSP32022BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWEcsv2023_CSP3_1 = pd.read_csv('YOUR PATH/CSP32023BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

BWE11_CSP3_1 = BWEcsv2011_CSP3_1['BWE']
BWE12_CSP3_1 = BWEcsv2012_CSP3_1['BWE']
BWE13_CSP3_1 = BWEcsv2013_CSP3_1['BWE']
BWE14_CSP3_1 = BWEcsv2014_CSP3_1['BWE']
BWE15_CSP3_1 = BWEcsv2015_CSP3_1['BWE']
BWE16_CSP3_1 = BWEcsv2016_CSP3_1['BWE']
BWE17_CSP3_1 = BWEcsv2017_CSP3_1['BWE']
BWE18_CSP3_1 = BWEcsv2018_CSP3_1['BWE']
BWE19_CSP3_1 = BWEcsv2019_CSP3_1['BWE']
BWE21_CSP3_1 = BWEcsv2021_CSP3_1['BWE']
BWE22_CSP3_1 = BWEcsv2022_CSP3_1['BWE']
BWE23_CSP3_1 = BWEcsv2023_CSP3_1['BWE']

#Interpolate between dates
BWEinterp11_CSP3_1 = BWE11_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp12_CSP3_1 = BWE12_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp13_CSP3_1 = BWE13_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp14_CSP3_1 = BWE14_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp15_CSP3_1 = BWE15_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp16_CSP3_1 = BWE16_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp17_CSP3_1 = BWE17_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp18_CSP3_1 = BWE18_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp19_CSP3_1 = BWE19_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp21_CSP3_1 = BWE21_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp22_CSP3_1 = BWE22_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp23_CSP3_1 = BWE23_CSP3_1.interpolate(method = 'linear', limit_direction = 'forward')

BWEFillNaN11_CSP3_1 = BWEinterp11_CSP3_1.fillna(int(BWEinterp11_CSP3.mean()), inplace=True)
BWEFillNaN12_CSP3_1 = BWEinterp12_CSP3_1.fillna(int(BWEinterp12_CSP3.mean()), inplace=True)
BWEFillNaN13_CSP3_1 = BWEinterp13_CSP3_1.fillna(int(BWEinterp13_CSP3.mean()), inplace=True)
BWEFillNaN14_CSP3_1 = BWEinterp14_CSP3_1.fillna(int(BWEinterp14_CSP3.mean()), inplace=True)
BWEFillNaN15_CSP3_1 = BWEinterp15_CSP3_1.fillna(int(BWEinterp15_CSP3.mean()), inplace=True)
BWEFillNaN16_CSP3_1 = BWEinterp16_CSP3_1.fillna(int(BWEinterp16_CSP3.mean()), inplace=True)
BWEFillNaN17_CSP3_1 = BWEinterp17_CSP3_1.fillna(int(BWEinterp17_CSP3.mean()), inplace=True)
BWEFillNaN18_CSP3_1 = BWEinterp18_CSP3_1.fillna(int(BWEinterp18_CSP3.mean()), inplace=True)
BWEFillNaN19_CSP3_1 = BWEinterp19_CSP3_1.fillna(int(BWEinterp19_CSP3.mean()), inplace=True)
BWEFillNaN21_CSP3_1 = BWEinterp21_CSP3_1.fillna(int(BWEinterp21_CSP3.mean()), inplace=True)
BWEFillNaN22_CSP3_1 = BWEinterp22_CSP3_1.fillna(int(BWEinterp22_CSP3.mean()), inplace=True)
BWEFillNaN23_CSP3_1 = BWEinterp23_CSP3_1.fillna(int(BWEinterp23_CSP3.mean()), inplace=True)

N0plotdata11_CSP3_1 = N011_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata12_CSP3_1 = N012_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata13_CSP3_1 = N013_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata14_CSP3_1 = N014_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata15_CSP3_1 = N015_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata16_CSP3_1 = N016_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata17_CSP3_1 = N017_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata18_CSP3_1 = N018_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata19_CSP3_1 = N019_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata21_CSP3_1 = N021_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata22_CSP3_1 = N022_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
N0plotdata23_CSP3_1 = N023_CSP3_1.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

### PLOT BWE VS N0 ###
#define plot variables
x11_CSP3_1 = BWEinterp11_CSP3_1
y11_CSP3_1 = N0interp11_CSP3_1
x12_CSP3_1 = BWEinterp12_CSP3_1
y12_CSP3_1 = N0interp12_CSP3_1
x13_CSP3_1 = BWEinterp13_CSP3_1
y13_CSP3_1 = N0interp13_CSP3_1
x14_CSP3_1 = BWEinterp14_CSP3_1
y14_CSP3_1 = N0interp14_CSP3_1
x15_CSP3_1 = BWEinterp15_CSP3_1
y15_CSP3_1 = N0interp15_CSP3_1
x16_CSP3_1 = BWEinterp16_CSP3_1
y16_CSP3_1 = N0interp16_CSP3_1
x17_CSP3_1 = BWEinterp17_CSP3_1
y17_CSP3_1 = N0interp17_CSP3_1
x18_CSP3_1 = BWEinterp18_CSP3_1
y18_CSP3_1 = N0interp18_CSP3_1
x19_CSP3_1 = BWEinterp19_CSP3_1
y19_CSP3_1 = N0interp19_CSP3_1
x21_CSP3_1 = BWEinterp21_CSP3_1
y21_CSP3_1 = N0interp21_CSP3_1
x22_CSP3_1 = BWEinterp22_CSP3_1
y22_CSP3_1 = N0interp22_CSP3_1
x23_CSP3_1 = BWEinterp23_CSP3_1
y23_CSP3_1 = N0interp23_CSP3_1

# Concatenate all x and y values
x_all_CSP3_1 = np.concatenate([x11_CSP3_1, x12_CSP3_1, x13_CSP3_1, x14_CSP3_1, x15_CSP3_1, x16_CSP3_1, x17_CSP3_1, x18_CSP3_1, x19_CSP3_1, x21_CSP3_1, x22_CSP3_1, x23_CSP3_1])
y_all_CSP3_1 = np.concatenate([y11_CSP3_1, y12_CSP3_1, y13_CSP3_1, y14_CSP3_1, y15_CSP3_1, y16_CSP3_1, y17_CSP3_1, y18_CSP3_1, y19_CSP3_1, y21_CSP3_1, y22_CSP3_1, y23_CSP3_1])

# Reshape x_all to 2D array for LinearRegression
x_all_2D_CSP3_1 = x_all_CSP3_1.reshape(-1, 1)

# Perform linear regression
slope_CSP3_1, intercept_CSP3_1, r_value_CSP3_1, p_value_CSP3_1, std_err_CSP3_1 = stats.linregress(x_all_CSP3_1, y_all_CSP3_1)

# Calculate the t-value for a 95% confidence interval
t_CSP3_1 = stats.t.ppf((1 + 0.95) / 2., len(x_all_CSP3_1) - 2)

# Calculate the confidence intervals for the slope and intercept
slope_conf_interval_CSP3_1 = t_CSP3_1 * std_err_CSP3_1
intercept_conf_interval_CSP3_1 = t_CSP3_1 * std_err_CSP3_1 * np.sqrt((1/len(x_all_CSP3_1)) + (np.mean(x_all_CSP3_1)**2 / np.sum((x_all_CSP3_1 - np.mean(x_all_CSP3_1))**2)))

# Calculate R-squared value
r_squared_CSP3_1 = r_value_CSP3_1**2

#plot the data
plt.scatter(x11_CSP3_1, y11_CSP3_1, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.5)
plt.scatter(x12_CSP3_1, y12_CSP3_1, color = 'indianred', marker = 'v', label = '2012', alpha=0.5)
plt.scatter(x13_CSP3_1, y13_CSP3_1, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.5)
plt.scatter(x14_CSP3_1, y14_CSP3_1, color = 'peru', marker = 's', label = '2014', alpha=0.5)
plt.scatter(x15_CSP3_1, y15_CSP3_1, color = 'gold', marker = '<', label = '2015', alpha=0.5)
plt.scatter(x16_CSP3_1, y16_CSP3_1, color = 'olivedrab', marker = '>', label = '2016', alpha=0.5)
plt.scatter(x17_CSP3_1, y17_CSP3_1, color = 'lawngreen', marker = '.', label = '2017', alpha=0.5)
plt.scatter(x18_CSP3_1, y18_CSP3_1, color = 'fuchsia', marker = '1', label = '2018', alpha=0.5)
plt.scatter(x19_CSP3_1, y19_CSP3_1, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x21_CSP3_1, y21_CSP3_1, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP3_1, y22_CSP3_1, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP3_1, y23_CSP3_1, color = 'darkorange', marker = 'p', label = '2023', alpha=0.5)

# Plot the line of best fit
plt.plot(x_all_CSP3_1, slope_CSP3_1*x_all_CSP3_1 + intercept_CSP3_1, color='black', label='Best fit', linewidth=2.5)
plt.gca().set_axisbelow(True)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE without 2020')
plt.text(-0.2, 1420, u"Best fit line: r\u00b2 = {:0.2f}".format(r_squared_CSP3_1), size=9)
plt.text(0.85, 1400, "p < 0.001", size=9)
plt.grid(which='major', zorder=-1, alpha=0.5)

#plt.rcParams['figure.figsize'] = [10,6]
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')

#Round values for the table
r_slope_CSP3_1 = round(slope_CSP3_1, 2)
r_intercept_CSP3_1 = round(intercept_CSP3_1, 2)
r_slope_conf_interval_CSP3_1 = round(slope_conf_interval_CSP3_1, 2)
r_intercept_conf_interval_CSP3_1 = round(intercept_conf_interval_CSP3_1, 2)

#Add confidence table to the plot
col_labels = ['Slope', 'Intercept']
row_labels = ['Value', '95% CI']
table_vals = [[r_slope_CSP3_1,r_intercept_CSP3_1], [r_slope_conf_interval_CSP3_1,r_intercept_conf_interval_CSP3_1]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_no_grav_no_2020.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP1 ################################
#####################with 2023 Comparison#########################
##################################################################
##################################################################

#########2023 Data##########
#import spreadsheet
GWC_CSP1_23 = pd.read_csv('YOUR PATH/2023_CSP1_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP1_23 = GWC_CSP1_23['SWC_1_1_1']
a_CSP1_23 = SWC_CSP1_23

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP1_23 = pd.read_csv('YOUR PATH/Npvi_CSP1_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP1_23 = NPIV_CSP1_23['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP1_23 = a_CSP1_23
lwc = 0.058
soc = 0.0092
N_CSP1_23 = NPIVCount_CSP1_23

#use parameters to calculate N0
N0_CSP1_23 = N_CSP1_23 / ((0.0808 / ((pwc_CSP1_23 + lwc + soc) + 0.115)) + 0.372)

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP1_23 = pd.read_csv('YOUR PATH/2023_CSP1_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP1_23 = BWEcsv_CSP1_23['BWE']

#define plot variables
x_CSP1_23 = BWE_CSP1_23
y_CSP1_23 = N0_CSP1_23

#calculate r-squared
corr_matrix_CSP1_23 = np.corrcoef(x_CSP1_23,y_CSP1_23)
corr_CSP1_23 = corr_matrix_CSP1_23[0,1]
R_sq_CSP1_23 = corr_CSP1_23**2

#Least square regression
A_CSP1_23 = np.vstack([x_CSP1_23, np.ones(len(x_CSP1_23))]).T
y_CSP1_23 = np.array(y_CSP1_23)  
y_CSP1_23 = np.array(y_CSP1_23)[:, np.newaxis]
alpha_CSP1 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_23.T,A_CSP1_23)),A_CSP1_23.T)),y_CSP1_23)
m_CSP1 = alpha_CSP1[0]
n_CSP1 = alpha_CSP1[1]
mm_CSP1 = float(m_CSP1)
mmm_CSP1 = [float(m_CSP1)]
nn_CSP1 = float(n_CSP1)
r_mm_CSP1 = round(mm_CSP1,2)
r_nn_CSP1 = round(nn_CSP1,2)

marker_size = 70
a_CSP1_1, b_CSP1_1 = np.polyfit(x_CSP1_23,y_CSP1_23,1)
plt.scatter(x19_CSP1, y19_CSP1, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP1, y20_CSP1, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP1, y21_CSP1, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP1, y22_CSP1, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP1, y23_CSP1, color = 'darkorange', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP1_23, y_CSP1_23, marker='o', color='red', s=marker_size, label = '2023 Gravimetric\nArithmetic Average')

# Plot the line of best fit
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + intercept_CSP1, color='black', label='Best fit', linewidth=2.5)
plt.plot(x_CSP1_23, a_CSP1_1*x_CSP1_23+b_CSP1_1, linestyle = '-', color = 'red', label='2023 Arithmetic\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE comparison')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP1 = round(r_squared_CSP1, 2)
r_R_sq_CSP1_23 = round(R_sq_CSP1_23, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2023', '2023']
table_vals = [[r_slope_CSP1,r_intercept_CSP1,r_r_squared_CSP1], [r_mm_CSP1,r_nn_CSP1,r_R_sq_CSP1_23]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023_comparison.png', dpi=300, bbox_inches='tight') 
plt.show()

#########################################
# Schron et al. Weight 
#########################################

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
r_R_sq_CSP1_weighted_schron = round(R_sq_CSP1_weighted_schron, 2)

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


marker_size = 70
a_CSP1_1_weighted_schron, b_CSP1_1_weighted_schron = np.polyfit(x_CSP1_weighted_schron,y_CSP1_weighted_schron,1)
plt.scatter(x19_CSP1, y19_CSP1, color = 'darkorange', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP1, y20_CSP1, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP1, y21_CSP1, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP1, y22_CSP1, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP1, y23_CSP1, color = 'dimgrey', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP1_weighted_schron, y_CSP1_weighted_schron, marker='o', color='red', s=marker_size, label = '2023 Gravimetric\nSchrön Weighted')

# Plot the line of best fit
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + intercept_CSP1, color='black', label='Best fit', linewidth=2.5)
plt.plot(x_CSP1_weighted_schron, a_CSP1_1_weighted_schron*x_CSP1_weighted_schron+b_CSP1_1_weighted_schron, linestyle = '-', color = 'red', label='2023 Schrön\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP1 = round(r_squared_CSP1, 2)
r_R_sq_CSP1_23weighted_schron = round(R_sq_CSP1_weighted_schron, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2023', '2023']
table_vals = [[r_slope_CSP1,r_intercept_CSP1,r_r_squared_CSP1], [r_mm_CSP1_weighted_schron,r_nn_CSP1_weighted_schron,r_R_sq_CSP1_weighted_schron]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_Multiyear_with_gravimetric.png', dpi=300, bbox_inches='tight')
plt.show()


################################
# Weighted 2023
################################
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
r_R_sq_CSP1_weighted = round(R_sq_CSP1_weighted, 2)

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

marker_size = 70
a_CSP1_1_weighted, b_CSP1_1_weighted = np.polyfit(x_CSP1_weighted,y_CSP1_weighted,1)
plt.scatter(x19_CSP1, y19_CSP1, color = 'dimgrey', marker = 'x', label = '2019')
plt.scatter(x20_CSP1, y20_CSP1, color = 'darkblue', marker = 'v', label = '2020')
plt.scatter(x21_CSP1, y21_CSP1, color = 'deeppink', marker = '^', label = '2021')
plt.scatter(x22_CSP1, y22_CSP1, color = 'purple', marker = 's', label = '2022')
plt.scatter(x23_CSP1, y23_CSP1, color = 'darkorange', marker = 'D', label = '2023')
plt.scatter(x_CSP1_weighted, y_CSP1_weighted, marker='o', color='red', s=marker_size, label = '2023 Gravimetric')

# Plot the line of best fit
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + intercept_CSP1, color='black', label='Best fit')
plt.plot(x_CSP1_weighted, a_CSP1_1_weighted*x_CSP1_weighted+b_CSP1_1_weighted, linestyle = '-', color = 'red', label='2023 Best fit')
plt.xlabel('BWE')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE comparison weighted 2023')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP1 = round(r_squared_CSP1, 2)
r_R_sq_CSP1_23weighted = round(R_sq_CSP1_weighted, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2022', '2023']
table_vals = [[r_slope_CSP1,r_intercept_CSP1,r_r_squared_CSP1], [r_mm_CSP1_weighted,r_nn_CSP1_weighted,r_R_sq_CSP1_weighted]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023_comparison_weighted.png', dpi=300, bbox_inches='tight')
plt.show()



marker_size = 70
a_CSP1_1, b_CSP1_1 = np.polyfit(x_CSP1_23,y_CSP1_23,1)
a_CSP1_1_weighted_schron, b_CSP1_1_weighted_schron = np.polyfit(x_CSP1_weighted_schron,y_CSP1_weighted_schron,1)
a_CSP1_1_weighted, b_CSP1_1_weighted = np.polyfit(x_CSP1_weighted,y_CSP1_weighted,1)
plt.scatter(x19_CSP1, y19_CSP1, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP1, y20_CSP1, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP1, y21_CSP1, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP1, y22_CSP1, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP1, y23_CSP1, color = 'darkorange', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP1_23, y_CSP1_23, marker='o', color='red', s=marker_size, label = '2023 Gravimetric\nArithmetic Average')
plt.scatter(x_CSP1_weighted_schron, y_CSP1_weighted_schron, marker='o', color='blue', s=marker_size, label = '2023 Gravimetric\nSchrön Weighted')
plt.scatter(x_CSP1_weighted, y_CSP1_weighted, marker='o', color='green', s=marker_size, label = '2023 Gravimetric\n10 cm, 25 cm Weighted')

# Plot the line of best fit
plt.plot(x_all_CSP1, slope_CSP1*x_all_CSP1 + intercept_CSP1, color='black', label='TDR Best fit', linewidth=2.5)
plt.plot(x_CSP1_23, a_CSP1_1*x_CSP1_23+b_CSP1_1, linestyle = '-', color = 'red', label='2023 Arithmetic\nBest fit', linewidth=2.5)
plt.plot(x_CSP1_weighted_schron, a_CSP1_1_weighted_schron*x_CSP1_weighted_schron+b_CSP1_1_weighted_schron, linestyle = '-', color = 'blue', label='2023 Schrön\nBest fit', linewidth=2.5)
plt.plot(x_CSP1_weighted, a_CSP1_1_weighted*x_CSP1_weighted+b_CSP1_1_weighted, linestyle = '-', color = 'green', label='2023 10cm, 25 cm\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP1 N$_{0}$ vs BWE comparison')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP1 = round(r_squared_CSP1, 2)
r_R_sq_CSP1_23 = round(R_sq_CSP1_23, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2023', '2023 Arithmetic', '2023 Schrön', '2023 10cm, 25cm']
table_vals = [[r_slope_CSP1,r_intercept_CSP1,r_r_squared_CSP1], [r_mm_CSP1,r_nn_CSP1,r_R_sq_CSP1_23], [r_mm_CSP1_weighted_schron,r_nn_CSP1_weighted_schron,r_R_sq_CSP1_weighted_schron], [r_mm_CSP1_weighted,r_nn_CSP1_weighted,r_R_sq_CSP1_weighted]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP1_2023_comparison_All_Gravimetric.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP2 ################################
#####################with 2023 Comparison#########################
##################################################################
##################################################################

#########2023 Data##########
#import spreadsheet
GWC2023_CSP2 = pd.read_csv('YOUR PATH/2023_CSP2_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP2_23 = GWC2023_CSP2['SWC_1_1_1']
a_CSP2_23 = SWC_CSP2_23

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP2_23 = pd.read_csv('YOUR PATH/Npvi_CSP2_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP2_23 = NPIV_CSP2_23['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP2_23 = a_CSP2_23
N_CSP2_23 = NPIVCount_CSP2_23

#use parameters to calculate N0
N0_CSP2_23 = N_CSP2_23 / ((0.0808 / ((pwc_CSP2_23 + lwc + soc) + 0.115)) + 0.372)

### GET BWE DATA FOR PLOTTING ###
#Import BWE data, determined by destructive sampling
BWEcsv_CSP2_23 = pd.read_csv('YOUR PATH/2023_CSP2_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP2_23 = BWEcsv_CSP2_23['BWE']

#define plot variables
x_CSP2_23 = BWE_CSP2_23
y_CSP2_23 = N0_CSP2_23

#calculate r-squared
corr_matrix_CSP2_23 = np.corrcoef(x_CSP2_23,y_CSP2_23)
corr_CSP2_23 = corr_matrix_CSP2_23[0,1]
R_sq_CSP2_23 = corr_CSP2_23**2

#Least square regression
A_CSP2_23 = np.vstack([x_CSP2_23, np.ones(len(x_CSP2_23))]).T
y_CSP2_23 = np.array(y_CSP2_23)  
y_CSP2_23 = np.array(y_CSP2_23)[:, np.newaxis]
alpha_CSP2 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_23.T,A_CSP2_23)),A_CSP2_23.T)),y_CSP2_23)
m_CSP2 = alpha_CSP2[0]
n_CSP2 = alpha_CSP2[1]
mm_CSP2 = float(m_CSP2)
mmm_CSP2 = [float(m_CSP2)]
nn_CSP2 = float(n_CSP2)
r_mm_CSP2 = round(mm_CSP2,2)
r_nn_CSP2 = round(nn_CSP2,2)

marker_size = 70
a_CSP2_1, b_CSP2_1 = np.polyfit(x_CSP2_23,y_CSP2_23,1)
plt.scatter(x19_CSP2, y19_CSP2, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP2, y20_CSP2, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP2, y21_CSP2, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP2, y22_CSP2, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP2, y23_CSP2, color = 'darkorange', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP2_23, y_CSP2_23, marker='o', color='red', s=marker_size, label = '2023 Gravimetric\nArithmetic Average')

# Plot the line of best fit
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + intercept_CSP2, color='black', label='Best fit', linewidth=2.5)
plt.plot(x_CSP2_23, a_CSP2_1*x_CSP2_23+b_CSP2_1, linestyle = '-', color = 'red', label='2023 Arithmetic\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE comparison')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP2 = round(r_squared_CSP2, 2)
r_R_sq_CSP2_23 = round(R_sq_CSP2_23, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2023', '2023']
table_vals = [[r_slope_CSP2,r_intercept_CSP2,r_r_squared_CSP2], [r_mm_CSP2,r_nn_CSP2,r_R_sq_CSP2_23]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2023_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

#########################################
# Schron et al. Weight 
#########################################
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
r_R_sq_CSP2_weighted_schron = round(R_sq_CSP2_weighted_schron, 2)

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


marker_size = 70
a_CSP2_1_weighted_schron, b_CSP2_1_weighted_schron = np.polyfit(x_CSP2_weighted_schron,y_CSP2_weighted_schron,1)
plt.scatter(x19_CSP2, y19_CSP2, color = 'darkorange', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP2, y20_CSP2, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP2, y21_CSP2, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP2, y22_CSP2, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP2, y23_CSP2, color = 'dimgrey', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP2_weighted_schron, y_CSP2_weighted_schron, marker='o', color='red', s=marker_size, label = '2023 Gravimetric\nSchrön Weighted')

# Plot the line of best fit
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + intercept_CSP2, color='black', label='Best fit', linewidth=2.5)
plt.plot(x_CSP2_weighted_schron, a_CSP2_1_weighted_schron*x_CSP2_weighted_schron+b_CSP2_1_weighted_schron, linestyle = '-', color = 'red', label='2023 Schrön\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP2 = round(r_squared_CSP2, 2)
r_R_sq_CSP2_23weighted_schron = round(R_sq_CSP2_weighted_schron, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2023', '2023']
table_vals = [[r_slope_CSP2,r_intercept_CSP2,r_r_squared_CSP2], [r_mm_CSP2_weighted_schron,r_nn_CSP2_weighted_schron,r_R_sq_CSP2_weighted_schron]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_Multiyear_with_gravimetric.png', dpi=300, bbox_inches='tight')
plt.show()


################################
# Weighted 2023
################################
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
r_R_sq_CSP2_weighted = round(R_sq_CSP2_weighted, 2)

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

marker_size = 70
a_CSP2_1_weighted, b_CSP2_1_weighted = np.polyfit(x_CSP2_weighted,y_CSP2_weighted,1)
plt.scatter(x19_CSP2, y19_CSP2, color = 'dimgrey', marker = 'x', label = '2019')
plt.scatter(x20_CSP2, y20_CSP2, color = 'darkblue', marker = 'v', label = '2020')
plt.scatter(x21_CSP2, y21_CSP2, color = 'deeppink', marker = '^', label = '2021')
plt.scatter(x22_CSP2, y22_CSP2, color = 'purple', marker = 's', label = '2022')
plt.scatter(x23_CSP2, y23_CSP2, color = 'darkorange', marker = 'D', label = '2023')
plt.scatter(x_CSP2_weighted, y_CSP2_weighted, marker='o', color='red', s=marker_size, label = '2023 Gravimetric')

# Plot the line of best fit
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + intercept_CSP2, color='black', label='TDR Best fit')
plt.plot(x_CSP2_weighted, a_CSP2_1_weighted*x_CSP2_weighted+b_CSP2_1_weighted, linestyle = '-', color = 'red', label='2023 Best fit')
plt.xlabel('BWE')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE comparison with weighting')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP2 = round(r_squared_CSP2, 2)
r_R_sq_CSP2_weighted = round(R_sq_CSP2_weighted, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2022', '2023']
table_vals = [[r_slope_CSP2,r_intercept_CSP2,r_r_squared_CSP2], [r_mm_CSP2_weighted,r_nn_CSP2_weighted,r_R_sq_CSP2_weighted]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
plt.savefig('YOUR PATH TO SAVE/CSP2_2023_comparison_weighted.png', dpi=300, bbox_inches='tight')
plt.show()


marker_size = 70
a_CSP2_1, b_CSP2_1 = np.polyfit(x_CSP2_23,y_CSP2_23,1)
a_CSP2_1_weighted_schron, b_CSP2_1_weighted_schron = np.polyfit(x_CSP2_weighted_schron,y_CSP2_weighted_schron,1)
a_CSP2_1_weighted, b_CSP2_1_weighted = np.polyfit(x_CSP2_weighted,y_CSP2_weighted,1)
plt.scatter(x19_CSP2, y19_CSP2, color = 'dimgrey', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP2, y20_CSP2, color = 'darkblue', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP2, y21_CSP2, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP2, y22_CSP2, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP2, y23_CSP2, color = 'darkorange', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP2_23, y_CSP2_23, marker='o', color='red', s=marker_size, label = '2023 Gravimetric\nArithmetic Average')
plt.scatter(x_CSP2_weighted_schron, y_CSP2_weighted_schron, marker='o', color='blue', s=marker_size, label = '2023 Gravimetric\nSchrön Weighted')
plt.scatter(x_CSP2_weighted, y_CSP2_weighted, marker='o', color='green', s=marker_size, label = '2023 Gravimetric\n10 cm, 25 cm Weighted')

# Plot the line of best fit
plt.plot(x_all_CSP2, slope_CSP2*x_all_CSP2 + intercept_CSP2, color='black', label='TDR Best fit', linewidth=2.5)
plt.plot(x_CSP2_23, a_CSP2_1*x_CSP2_23+b_CSP2_1, linestyle = '-', color = 'red', label='2023 Arithmetic\nBest fit', linewidth=2.5)
plt.plot(x_CSP2_weighted_schron, a_CSP2_1_weighted_schron*x_CSP2_weighted_schron+b_CSP2_1_weighted_schron, linestyle = '-', color = 'blue', label='2023 Schrön\nBest fit', linewidth=2.5)
plt.plot(x_CSP2_weighted, a_CSP2_1_weighted*x_CSP2_weighted+b_CSP2_1_weighted, linestyle = '-', color = 'green', label='2023 10cm, 25 cm\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP2 N$_{0}$ vs BWE comparison')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(2340, 3400)
plt.xlim(-0.3,7.8)

#Add confidence table to the plot
r_r_squared_CSP2 = round(r_squared_CSP2, 2)
r_R_sq_CSP2_23 = round(R_sq_CSP2_23, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2019-2023', '2023 Arithmetic', '2023 Schrön', '2023 10cm, 25cm']
table_vals = [[r_slope_CSP2,r_intercept_CSP2,r_r_squared_CSP2], [r_mm_CSP2,r_nn_CSP2,r_R_sq_CSP2_23], [r_mm_CSP2_weighted_schron,r_nn_CSP2_weighted_schron,r_R_sq_CSP2_weighted_schron], [r_mm_CSP2_weighted,r_nn_CSP2_weighted,r_R_sq_CSP2_weighted]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP2_2023_comparison_All_Gravimetric_Method.png', dpi=300, bbox_inches='tight')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
#####################with 2023 Comparison#########################
##################################################################
##################################################################

#########2023 Data##########

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


#########################################
# Schron et al. Weight 
#########################################

##### 2023 #####
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
r_R_sq_CSP3_weighted_schron = round(R_sq_CSP3_weighted_schron, 2)

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


######### Gravimetric Data - Weighted##########

#import spreadsheet
GWC_CSP3_weighted = pd.read_csv('YOUR PATH/2023_CSP3_SWC_depth_weighted.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

SWC_CSP3_Upper = GWC_CSP3_weighted['Upper']
SWC_CSP3_Lower = GWC_CSP3_weighted['Lower']
SWC_CSP3_Weighted = ((SWC_CSP3_Upper * 0.75) + (SWC_CSP3_Lower * 0.25))
a_CSP3_weighted = SWC_CSP3_Weighted

### DATA AND CALCULATIONS FOR NEUTRON COUNTS ###
#Import spreadsheet with NPIV (the version here is filtered cph)
NPIV_CSP3_weighted = pd.read_csv('YOUR PATH/Npvi_CSP3_23.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])

#Define the data for ease in calculations later
NPIVCount_CSP3_weighted = NPIV_CSP3_weighted['NPIV']

### CALCUATE N0 ###
#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc_CSP3_weighted = a_CSP3_weighted
N_CSP3_weighted = NPIVCount_CSP3_weighted

#use parameters to calculate N0
N0_CSP3_weighted = N_CSP3_weighted / ((0.0808 / ((pwc_CSP3_weighted + lwc + soc) + 0.115)) + 0.372)

### GET BWE DATA FOR PLOTTING ###

#Import BWE data, determined by destructive sampling
BWEcsv_CSP3_weighted = pd.read_csv('YOUR PATH/2023_CSP3_BWE.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])
BWE_CSP3_weighted = BWEcsv_CSP3_weighted['BWE']

#define plot variables
x_CSP3_weighted = BWE_CSP3_weighted
y_CSP3_weighted = N0_CSP3_weighted

#calculate r-squared
corr_matrix_CSP3_weighted = np.corrcoef(x_CSP3_weighted,y_CSP3_weighted)
corr_CSP3_weighted = corr_matrix_CSP3_weighted[0,1]
R_sq_CSP3_weighted = corr_CSP3_weighted**2
r_R_sq_CSP3_weighted = round(R_sq_CSP3_weighted, 2)

#Least square regression
A_CSP3_23_weighted = np.vstack([x_CSP3_weighted, np.ones(len(x_CSP3_weighted))]).T
y_CSP3_23_weighted = np.array(y_CSP3_weighted)  
y_CSP3_23_weighted = y_CSP3_weighted.values[:, np.newaxis]
alpha_CSP3_weighted = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_23_weighted.T,A_CSP3_23_weighted)),A_CSP3_23_weighted.T)),y_CSP3_weighted)
m_CSP3_weighted = alpha_CSP3_weighted[0]
n_CSP3_weighted = alpha_CSP3_weighted[1]
mm_CSP3_weighted = float(m_CSP3_weighted)
mmm_CSP3_weighted = [float(m_CSP3_weighted)]
nn_CSP3_weighted = float(n_CSP3_weighted)
r_mm_CSP3_weighted = round(mm_CSP3_weighted,2)
r_nn_CSP3_weighted = round(nn_CSP3_weighted,2)


###### CRNS Distance Only - Weighted ######
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
r_R_sq_CSP3_weighted_CRNSDistance = round(R_sq_CSP3_weighted_CRNSDistance, 2)

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

marker_size = 70
a_CSP3_1_1, b_CSP3_1_1 = np.polyfit(x_CSP3_CRNSDistance,y_CSP3_CRNSDistance,1)
a_CSP3_1_weighted_schron, b_CSP3_1_weighted_schron = np.polyfit(x_CSP3_weighted_schron,y_CSP3_weighted_schron,1)
a_CSP3_1_1_weighted_CRNSDistance, b_CSP3_1_1_weighted_CRNSDistance = np.polyfit(x_CSP3_weighted_CRNSDistance,y_CSP3_weighted_CRNSDistance,1)
plt.scatter(x11_CSP3, y11_CSP3, color = 'saddlebrown', marker = '*', label = '2011', alpha=0.5)
plt.scatter(x12_CSP3, y12_CSP3, color = 'indianred', marker = 'v', label = '2012', alpha=0.5)
plt.scatter(x13_CSP3, y13_CSP3, color = 'lightseagreen', marker = 'X', label = '2013', alpha=0.5)
plt.scatter(x14_CSP3, y14_CSP3, color = 'peru', marker = 's', label = '2014', alpha=0.5)
plt.scatter(x15_CSP3, y15_CSP3, color = 'gold', marker = '<', label = '2015', alpha=0.5)
plt.scatter(x16_CSP3, y16_CSP3, color = 'olivedrab', marker = '>', label = '2016', alpha=0.5)
plt.scatter(x17_CSP3, y17_CSP3, color = 'lawngreen', marker = '.', label = '2017', alpha=0.5)
plt.scatter(x18_CSP3, y18_CSP3, color = 'fuchsia', marker = '1', label = '2018', alpha=0.5)
plt.scatter(x19_CSP3, y19_CSP3, color = 'darkorange', marker = 'x', label = '2019', alpha=0.5)
plt.scatter(x20_CSP3, y20_CSP3, color = 'aquamarine', marker = 'v', label = '2020', alpha=0.5)
plt.scatter(x21_CSP3, y21_CSP3, color = 'deeppink', marker = '^', label = '2021', alpha=0.5)
plt.scatter(x22_CSP3, y22_CSP3, color = 'purple', marker = 's', label = '2022', alpha=0.5)
plt.scatter(x23_CSP3, y23_CSP3, color = 'yellow', marker = 'D', label = '2023', alpha=0.5)
plt.scatter(x_CSP3_CRNSDistance, y_CSP3_CRNSDistance, marker='o', color='red', s=marker_size, label = '2023 Gravimetric Arithmetic')
plt.scatter(x_CSP3_weighted_schron, y_CSP3_weighted_schron, marker='o', color='blue', s=marker_size, label = '2023 Gravimetric\nSchrön Weighted')
plt.scatter(x_CSP3_weighted_CRNSDistance, y_CSP3_weighted_CRNSDistance, marker='o', color='green', s=marker_size, label = '2023 Gravimetric\n10 cm, 25cm Weighted')

# Plot the line of best fit
plt.plot(x_all_CSP3, slope_CSP3*x_all_CSP3 + intercept_CSP3, color='black', label='TDR Best fit', linewidth=2.5)
plt.plot(x_CSP3_CRNSDistance, a_CSP3_1_1*x_CSP3_CRNSDistance+b_CSP3_1_1, linestyle = '-', color = 'red', label='2023 Arithmetic\nBest fit', linewidth=2.5)
plt.plot(x_CSP3_weighted_schron, a_CSP3_1_weighted_schron*x_CSP3_weighted_schron+b_CSP3_1_weighted_schron, linestyle = '-', color = 'blue', label='2023 Schrön\nBest fit', linewidth=2.5)
plt.plot(x_CSP3_weighted_CRNSDistance, a_CSP3_1_1_weighted_CRNSDistance*x_CSP3_weighted_CRNSDistance+b_CSP3_1_1_weighted_CRNSDistance, linestyle = '-', color = 'green', label='2023 10cm, 25cm\nBest fit', linewidth=2.5)
plt.xlabel('BWE (mm)')
plt.ylabel(r'$N_{0}$ (cph)')
plt.title(r'CSP3 N$_{0}$ vs BWE')
plt.gca().set_axisbelow(True)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.grid(which='major', zorder=-1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad=0, fontsize = '7.75')
plt.ylim(1400, 2000)
plt.xlim(-0.3,6.1)

#Add confidence table to the plot
r_r_squared_CSP3 = round(r_squared_CSP3, 2)
r_R_sq_CSP3_23 = round(R_sq_CSP3_CRNSDistance, 2)
r_R_sq_CSP3_weighted_schron = round(R_sq_CSP3_weighted_schron, 2)
r_R_sq_CSP3_weighted_CRNSDistance = round(R_sq_CSP3_weighted_CRNSDistance, 2)
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['2011-2023', '2023 Arithmetic','2023 Schrön', '2023 10cm, 25cm']
table_vals = [[r_slope_CSP3,r_intercept_CSP3,r_r_squared_CSP3], [r_mm_CSP3_CRNSDistance,r_nn_CSP3_CRNSDistance,r_R_sq_CSP3_23],[r_mm_CSP3_weighted_schron,r_nn_CSP3_weighted_schron,r_R_sq_CSP3_weighted_schron],[r_mm_CSP3_weighted_CRNSDistance,r_nn_CSP3_weighted_CRNSDistance,r_R_sq_CSP3_weighted_CRNSDistance]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='best')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/CSP3_Multiyear_with_gravimetric.png', dpi=300, bbox_inches='tight')
plt.show()