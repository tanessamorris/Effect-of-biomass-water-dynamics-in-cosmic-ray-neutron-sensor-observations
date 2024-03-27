#import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
GWC_CSP1 = pd.read_csv('YOUR PATH/AmerifluxCSP1_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1_CSP1 = GWC_CSP1['SWC_1_1_1']/1.42
SWC_1_2_1_CSP1 = GWC_CSP1['SWC_1_2_1']/1.42
SWC_1_3_1_CSP1 = GWC_CSP1['SWC_1_3_1']/1.42
SWC_1_4_1_CSP1 = GWC_CSP1['SWC_1_4_1']/1.42
SWC_2_1_1_CSP1 = GWC_CSP1['SWC_2_1_1']/1.42
SWC_2_2_1_CSP1 = GWC_CSP1['SWC_2_2_1']/1.42
SWC_2_3_1_CSP1 = GWC_CSP1['SWC_2_3_1']/1.42
SWC_2_4_1_CSP1 = GWC_CSP1['SWC_2_4_1']/1.42
SWC_3_1_1_CSP1 = GWC_CSP1['SWC_3_1_1']/1.42
SWC_3_2_1_CSP1 = GWC_CSP1['SWC_3_2_1']/1.42
SWC_3_3_1_CSP1 = GWC_CSP1['SWC_3_3_1']/1.42
SWC_3_4_1_CSP1 = GWC_CSP1['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1 = ((((SWC_1_1_1_CSP1 + SWC_2_1_1_CSP1 + SWC_3_1_1_CSP1)/3)*0.75) + (((SWC_1_2_1_CSP1 + SWC_2_2_1_CSP1 + SWC_3_2_1_CSP1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP1 = twosenwavgall_CSP1.resample('D').mean() 
dailywavgSWC_CSP1.to_csv(r'YOUR PATH TO MAKE/CSP1_DailyAvg.csv')
DailySWC_CSP1 = pd.read_csv(r'YOUR PATH/CSP1_DailyAvg.csv')
Date_CSP1 = DailySWC_CSP1['DATE']
Water_Content_CSP1 = DailySWC_CSP1['SWC']

#Calculate the standard deviation
#Files created here are used in other scripts
stdev_CSP1 = np.std(Water_Content_CSP1)
stderror_CSP1 = stdev_CSP1/(np.sqrt(3))
pos95_CSP1 = Water_Content_CSP1 + (2 * stderror_CSP1)
neg95_CSP1 = Water_Content_CSP1 - (2 * stderror_CSP1)
pos95_CSP1.to_csv(r'YOUR PATH TO MAKE/CSP3_Upper.csv')
neg95_CSP1.to_csv(r'YOUR PATH TO MAKE/CSP3_Lower.csv')

#define plot variables
x_CSP1 = Date_CSP1
y_CSP1 = Water_Content_CSP1
a_CSP1 = pos95_CSP1
b_CSP1 = neg95_CSP1

#plot data
fig_CSP1, ax_CSP1 = plt.subplots()
ax_CSP1.plot(x_CSP1,a_CSP1, '-', color = 'gold', label = "95% Confidence Interval")
ax_CSP1.plot(x_CSP1,b_CSP1, '-', color = 'gold')
ax_CSP1.fill_between(x_CSP1, a_CSP1, b_CSP1, color = 'gold')
ax_CSP1.plot(x_CSP1,y_CSP1, '-', color='navy', linewidth = 0.75, label = "TDR SWC")
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.title('Soil Water Content Confidence CSP1')
plt.xticks(np.linspace(4, 1350, 12))
plt.ylim(0.045,0.38)
plt.xticks(rotation = 35)
plt.legend(loc='upper right')
plt.savefig('YOUR PATH TO MAKE/CSP1_Confidence.png')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP2 ################################
##################################################################
##################################################################
##################################################################
### DATA AND CALCULATIONS FOR SWC ###
GWC_CSP2 = pd.read_csv('YOUR PATH/AmerifluxCSP2_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP2 = GWC_CSP2['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP2 = GWC_CSP2['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP2 = GWC_CSP2['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP2 = GWC_CSP2['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1_CSP2 = GWC_CSP2['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP2 = GWC_CSP2['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP2 = GWC_CSP2['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP2 = GWC_CSP2['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1_CSP2 = GWC_CSP2['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP2 = GWC_CSP2['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP2 = GWC_CSP2['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP2 = GWC_CSP2['SWC_PI_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2 = ((((SWC_PI_F_1_1_1_CSP2 + SWC_PI_F_2_1_1_CSP2 + SWC_PI_F_3_1_1_CSP2)/3)*0.75) + (((SWC_PI_F_1_2_1_CSP2 + SWC_PI_F_2_2_1_CSP2 + SWC_PI_F_3_2_1_CSP2)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP2 = twosenwavgall_CSP2.resample('D').mean() 
dailywavgSWC_CSP2.to_csv(r'YOUR PATH TO MAKE/DailySWC_CSP2.csv')
DailySWC_CSP2 = pd.read_csv(r'YOUR PATH/DailySWC_CSP2.csv')
Date_CSP2 = DailySWC_CSP2['DATE']
Water_Content_CSP2 = DailySWC_CSP2['SWC']

#Calculate the standard deviation
stdev_CSP2 = np.std(Water_Content_CSP2)
stderror_CSP2 = stdev_CSP2/(np.sqrt(3))
pos95_CSP2 = Water_Content_CSP2 + (2 * stderror_CSP2)
neg95_CSP2 = Water_Content_CSP2 - (2 * stderror_CSP2)
pos95_CSP2.to_csv(r'YOUR PATH TO MAKE/CSP2_Upper.csv')
neg95_CSP2.to_csv(r'YOUR PATH TO MAKE/CSP2_Lower.csv')

#define plot variables
x_CSP2 = Date_CSP2
y_CSP2 = Water_Content_CSP2
a_CSP2 = pos95_CSP2
b_CSP2 = neg95_CSP2

#plot data
fig_CSP2, ax_CSP2 = plt.subplots()
ax_CSP2.plot(x_CSP2,a_CSP2, '-', color = 'gold', label = "95% Confidence Interval")
ax_CSP2.plot(x_CSP2,b_CSP2, '-', color = 'gold')
ax_CSP2.fill_between(x_CSP2,a_CSP2,b_CSP2, color = 'gold')
ax_CSP2.plot(x_CSP2,y_CSP2, '-', color='navy', linewidth = 0.75, label = "TDR SWC")
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.title('Soil Water Content Confidence CSP2')
plt.xticks(np.linspace(4, 1350, 12))
plt.xticks(rotation = 35)
plt.ylim(0.045,0.38)
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO MAKE/CSP2_Confidence.png')
plt.show()


##################################################################
##################################################################
##################################################################
############################ CSP3 ################################
##################################################################
##################################################################
##################################################################
### DATA AND CALCULATIONS FOR SWC ###
GWC_CSP3 = pd.read_csv('YOUR PATH/AmerifluxCSP3_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1_CSP3 = GWC_CSP3['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1_CSP3 = GWC_CSP3['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1_CSP3 = GWC_CSP3['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1_CSP3 = GWC_CSP3['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1_CSP3 = GWC_CSP3['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1_CSP3 = GWC_CSP3['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1_CSP3 = GWC_CSP3['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1_CSP3 = GWC_CSP3['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1_CSP3 = GWC_CSP3['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1_CSP3 = GWC_CSP3['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1_CSP3 = GWC_CSP3['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1_CSP3 = GWC_CSP3['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1_CSP3 = GWC_CSP3['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1_CSP3 = GWC_CSP3['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1_CSP3 = GWC_CSP3['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1_CSP3 = GWC_CSP3['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1_CSP3 = GWC_CSP3['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1_CSP3 = GWC_CSP3['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1_CSP3 = GWC_CSP3['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1_CSP3 = GWC_CSP3['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3 = ((((SWC_PI_F_1_1_1_CSP3 + SWC_PI_F_2_1_1_CSP3 + SWC_PI_F_3_1_1_CSP3 + SWC_PI_F_4_1_1_CSP3)/4)*0.75) + (((SWC_PI_F_1_2_1_CSP3 + SWC_PI_F_2_2_1_CSP3 + SWC_PI_F_3_2_1_CSP3 + SWC_PI_F_4_2_1_CSP3)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC_CSP3 = twosenwavgall_CSP3.resample('D').mean() 
dailywavgSWC_CSP3.to_csv(r'YOUR PATH TO MAKE/CSP3_AllDailyAvg.csv')
DailySWC_CSP3 = pd.read_csv(r'YOUR PATH/CSP3_AllDailyAvg.csv')
Date_CSP3 = DailySWC_CSP3['DATE']
Water_Content_CSP3 = DailySWC_CSP3['SWC']

#Calculate the standard deviation
stdev_CSP3 = np.std(Water_Content_CSP3)
stderror_CSP3 = stdev_CSP3/(np.sqrt(4))
pos95_CSP3 = Water_Content_CSP3 + (2 * stderror_CSP3)
neg95_CSP3 = Water_Content_CSP3 - (2 * stderror_CSP3)
pos95_CSP3.to_csv(r'YOUR PATH TO MAKE/CSP3_Upper_Newest.csv')
neg95_CSP3.to_csv(r'YOUR PATH TO MAKE/CSP3_Lower_Newest.csv')

#define plot variables
x_CSP3 = Date_CSP3
y_CSP3 = Water_Content_CSP3
a_CSP3 = pos95_CSP3
b_CSP3 = neg95_CSP3

#plot data
fig_CSP3, ax_CSP3 = plt.subplots()
ax_CSP3.plot(x_CSP3,a_CSP3, '-', color = 'gold', label = "95% Confidence Interval")
ax_CSP3.plot(x_CSP3,b_CSP3, '-', color = 'gold')
ax_CSP3.fill_between(x_CSP3, a_CSP3, b_CSP3, color = 'gold')
ax_CSP3.plot(x_CSP3,y_CSP3, '-', color='navy', linewidth = 0.75, label = "TDR SWC")
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.title('Soil Water Content Confidence CSP3')
plt.xticks(np.linspace(4, 4200, 12))
plt.xticks(rotation = 35)
plt.ylim(0.045,0.38)
plt.legend()
plt.savefig('YOUR PATH TO MAKE/CSP3_Confidence.png')
plt.show()