#import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

### DATA AND CALCULATIONS FOR SWC ###

GWC = pd.read_csv('YOUR PATH/AmerifluxCSP1_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgsen1 = (SWC_1_1_1*0.75)+(SWC_1_2_1*0.25)
twosenwavgsen2 = (SWC_2_1_1*0.75)+(SWC_2_2_1*0.25)
twosenwavgsen3 = (SWC_3_1_1*0.75)+(SWC_3_2_1*0.25)
twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
#A new sheet is made to fix errors in plotting later on
dailywavgSWC = twosenwavgall.resample('D').mean() 
dailywavgSWC.to_csv(r'YOUR PATH TO MAKE/CSP1_DailyAvg.csv')
DailySWC = pd.read_csv(r'YOUR PATH/CSP1_DailyAvg.csv')

Date = DailySWC['DATE']
Water_Content = DailySWC['0']

#Calculate the standard deviation
stdev = np.std(Water_Content)
stderror = stdev/(np.sqrt(3))
pos95 = Water_Content + (2 * stderror)
neg95 = Water_Content - (2 * stderror)
#This step done for other scripts in the repository
pos95.to_csv(r'YOUR PATH TO MAKE/CSP1_Upper.csv')
neg95.to_csv(r'YOUR PATH TO MAKE/CSP1_Lower.csv')

#define plot variables
x = Date
y = Water_Content
a = pos95
b = neg95

#actually plot data with best fit line, slope, and r-squared
fig, ax = plt.subplots()
ax.plot(x,a, '-', color = 'gold', label = "95% Confidence Interval")
ax.plot(x,b, '-', color = 'gold')
ax.fill_between(x, a, b, color = 'gold')
ax.plot(x,y, '-', color='navy', linewidth = 1.5, label = "TDR SWC")
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.title('Soil Water Content Confidence CSP1')
plt.xticks(np.linspace(1, 1550, 10))
plt.ylim(0.045,0.38)
plt.xticks(rotation = 35)
plt.legend()
plt.tight_layout() 
plt.savefig('YOUR PATH TO SAVE/CSP1_SWC_Confidence.png', dpi=300, bbox_inches='tight')
plt.show()


### DATA AND CALCULATIONS FOR SWC ###

GWC = pd.read_csv('YOUR PATH/AmerifluxCSP2_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgsen1 = (SWC_PI_F_1_1_1*0.75)+(SWC_PI_F_1_2_1*0.25)
twosenwavgsen2 = (SWC_PI_F_2_1_1*0.75)+(SWC_PI_F_2_2_1*0.25)
twosenwavgsen3 = (SWC_PI_F_3_1_1*0.75)+(SWC_PI_F_3_2_1*0.25)
twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

dailywavgSWC.to_csv(r'YOUR PATH TO MAKE/CSP2_DailyAvg.csv')
DailySWC = pd.read_csv(r'YOUR PATH TO MAKE/CSP2_DailyAvg.csv')

Date = DailySWC['DATE']
Water_Content = DailySWC['0']

#Calculate the standard deviation
stdev = np.std(Water_Content)
stderror = stdev/(np.sqrt(3))
pos95 = Water_Content + (2 * stderror)
neg95 = Water_Content - (2 * stderror)

pos95.to_csv(r'YOUR PATH/CSP2_Upper.csv')
neg95.to_csv(r'YOUR PATH/CSP2_Lower.csv')

#define plot variables
x = Date
y = Water_Content
a = pos95
b = neg95

#actually plot data with best fit line, slope, and r-squared
fig, ax = plt.subplots()
ax.plot(x,a, '-', color = 'gold', label = "95% Confidence Interval")
ax.plot(x,b, '-', color = 'gold')
ax.fill_between(x, a, b, color = 'gold')
ax.plot(x,y, '-', color='navy', linewidth = 1.5, label = "TDR SWC")
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.title('Soil Water Content Confidence CSP2')
plt.xticks(np.linspace(1, 1550, 10))
plt.xticks(rotation = 35)
plt.ylim(0.045,0.38)
plt.legend(loc = 'upper right')
plt.tight_layout() 
plt.savefig('YOUR PATH TO SAVE/CSP2_SWC_Confidence.png', dpi=300, bbox_inches='tight')
plt.show()


### DATA AND CALCULATIONS FOR SWC ###

GWC = pd.read_csv('YOUR PATH/AmerifluxCSP3_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['SWC_PI_F_4_5_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgsen1 = (SWC_PI_F_1_1_1*0.75)+(SWC_PI_F_1_2_1*0.25)
twosenwavgsen2 = (SWC_PI_F_2_1_1*0.75)+(SWC_PI_F_2_2_1*0.25)
twosenwavgsen3 = (SWC_PI_F_3_1_1*0.75)+(SWC_PI_F_3_2_1*0.25)
twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

dailywavgSWC.to_csv(r'YOUR PATH TO MAKE/CSP3_DailyAvg.csv')
DailySWC = pd.read_csv(r'YOUR PATH/CSP3_DailyAvg.csv')

Date = DailySWC['DATE']
Water_Content = DailySWC['0']

#Calculate the standard deviation
stdev = np.std(Water_Content)
stderror = stdev/(np.sqrt(4))
pos95 = Water_Content + (2 * stderror)
neg95 = Water_Content - (2 * stderror)

pos95.to_csv(r'YOUR PATH TO MAKE/CSP3_Upper.csv')
neg95.to_csv(r'YOUR PATH TO MAKE/CSP3_Lower.csv')

#define plot variables
x = Date
y = Water_Content
a = pos95
b = neg95

#actually plot data with best fit line, slope, and r-squared
fig, ax = plt.subplots()
ax.plot(x,a, '-', color = 'gold', label = "95% Confidence Interval")
ax.plot(x,b, '-', color = 'gold')
ax.fill_between(x, a, b, color = 'gold')
ax.plot(x,y, '-', color='navy', linewidth = 1.5, label = "TDR SWC")
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.title('Soil Water Content Confidence CSP3')
plt.xticks(np.linspace(1, 4655, 14))
plt.xticks(rotation = 35)
plt.ylim(0.045,0.38)
plt.legend()
plt.tight_layout() 
plt.savefig('YOUR PATH TO SAVE/CSP3.png', dpi=300)
plt.show()
