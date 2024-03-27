import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

#All sheets imported in this script are from the SWC_95Confidence_Interval.py script, so it must be run first
#The file paths created from the SWC_95Confidence_Interval.py script must match the paths in this script
#The _DailyAvg, _Upper, and _Lower files are created in the SWC_95Confidence_Interval.py script

##################################
#############CSP1#################
##################################

#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_DailyAvg.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['0']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP1_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/AllNpviDataCSP1.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP1_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP1.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Upper.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['0']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Lower.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['0']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP1_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='lightskyblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='lightskyblue', zorder=0)
ax.fill_between(x, a, b, color = 'lightskyblue')
ax.plot(x,y, color='indigo', label = r'$N_{0}$', linewidth = 1.5)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(1, 1620, 10))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ Time-Series CSP1')
plt.legend(loc = 'upper right')
plt.ylim(1900,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP1_All_Time.png', dpi=300)
plt.show()


##################################
#############CSP2#################
##################################

#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_DailyAvg.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['0']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP2_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/AllNpviDataCSP2.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP2_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP2.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP2.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Upper.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['0']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Lower.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['0']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP2_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='lightskyblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='lightskyblue', zorder=0)
ax.fill_between(x, a, b, color = 'lightskyblue')
ax.plot(x,y, color='indigo', label = r'$N_{0}$', linewidth = 1.5)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(1, 1620, 10))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ Time-Series CSP2')
plt.legend(loc = 'upper right')
plt.ylim(1900,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP2_All_Time.png', dpi=300)
plt.show()


##################################
#############CSP3#################
##################################

#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['0']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP3_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/AllNpviDataCSP3.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP3_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_New.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['0']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_New.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['0']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='lightskyblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='lightskyblue', zorder=0)
ax.fill_between(x, a, b, color = 'lightskyblue')
ax.plot(x,y, color='indigo', label = r'$N_{0}$', linewidth = 1.5)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(1, 4600, 13))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ Time-Series CSP3')
plt.legend(loc = 'upper right')
plt.ylim(950,2000)
plt.savefig('YOUR PATH/N0_CSP3_All_Time.png', dpi=300)
plt.show()