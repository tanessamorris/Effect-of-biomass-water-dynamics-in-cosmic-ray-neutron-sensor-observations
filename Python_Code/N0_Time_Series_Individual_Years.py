import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

#All sheets imported in this script are from the SWC_95Confidence_Interval.py script, so it must be run first
#The file paths created from the SWC_95Confidence_Interval.py script must match the paths in this script
#The _SWC, _upper, and _lower sheets were created in the SWC_95Confidence_Interval.py script, but I isolated each year, so use the path from the repository


#####################################
#####################################
############ CSP1 ###################
#####################################
#####################################

### 2019 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP12019_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP12019_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP12019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP12019_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
print(N0__1.head())
df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP1_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP1_2019.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Upper_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2019_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2019_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']
print(N01.head())

##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Lower_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2019_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP1_2019_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label = r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 180, 7))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2019 CSP1')
plt.legend('upper right')
plt.ylim(2400,3400)
plt.legend()
plt.savefig('YOUR PATH TO SAVE/N0_CSP1_2019.png')
plt.show()



### 2020 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP12020_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP12020_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP12020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP12020_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP1_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP1_2020.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Upper_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2020_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2020_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Lower_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2020_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP1_2020_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 240, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2020 CSP1')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP1_2020.png')
plt.show()



### 2021 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP12021_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP12021_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP12021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP12021_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP1_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP1_2021.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Upper_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2021_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2021_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Lower_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2021_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP1_2021_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 240, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$  2021 CSP1')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP1_2021.png')
plt.show()



### 2022 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP12022_SWC.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP12022_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP12022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP12022_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP1_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP1_2022.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Upper_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2022_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2022_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP1_Lower_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2022_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP1_2022_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 240, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2022 CSP1')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP1_2022.png')
plt.show()



### 2023 ###
#For these files, look in the SWC file with other files used in this script
Real_N0 = pd.read_csv('YOUR PATH/N0_CSP1_2023.csv', parse_dates=True, index_col='DATE')
Upper_N0 = pd.read_csv('YOUR PATH/N0_CSP1_Upper_2023.csv', parse_dates=True, index_col='DATE')
Lower_N0 = pd.read_csv('YOUR PATH/N0_CSP1_Lower_2023.csv', parse_dates=True, index_col='DATE')

x = Real_N0.index
y = Real_N0['0']
y2 = Upper_N0['0']
y3 = Lower_N0['0']

fig, ax = plt.subplots()
ax.plot(x, y, color='purple', label= r'$N_{0}$', linewidth = 2)
ax.plot(x, y2, color='royalblue', label = "95% CI", zorder=0)
ax.fill_between(x, y2, y3, color='royalblue')
ax.plot(x, y3, color='royalblue', zorder=0)
ax.xaxis.set_major_locator(ticker.MaxNLocator(9))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
plt.gcf().autofmt_xdate()  
plt.title(r'$N_{0}$ 2023 CSP1')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.legend()
plt.savefig('YOUR PATH TO SAVE/N0_CSP1_2023.png')
plt.show()


#####################################
#####################################
############ CSP2 ###################
#####################################
#####################################

### 2019 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_DailyAvg_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP22019_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP22019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP22019_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP2_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP2_2019.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Upper_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2019_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2019_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Lower_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2019_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP2_2019_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2019 CSP2')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP2_2019.png')
plt.show()



### 2020 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_DailyAvg_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP22020_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP22020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP22020_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP2_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP2_2020.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Upper_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2020_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2020_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Lower_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2020_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP2_2020_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2020 CSP2')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP2_2020.png')
plt.show()



### 2021 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_DailyAvg_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP22021_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP22021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP22021_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP2_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP2_2021.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Upper_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2021_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2021_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Lower_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2021_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP2_2021_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2021 CSP2')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP2_2021.png')
plt.show()



### 2022 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_DailyAvg_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP22022_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP22022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP22022_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP2_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP2_2022.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Upper_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2022_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2022_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP2_Lower_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2022_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP2_2022_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2022 CSP2')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.savefig('YOUR PATH TO SAVE/N0_CSP2_2022.png')
plt.show()



### 2023 ###
Real_N0 = pd.read_csv('YOUR PATH/N0_CSP2_2023.csv', parse_dates=True, index_col='DATE')
Upper_N0 = pd.read_csv('YOUR PATH/N0_CSP2_Upper_2023.csv', parse_dates=True, index_col='DATE')
Lower_N0 = pd.read_csv('YOUR PATH/N0_CSP2_Lower_2023.csv', parse_dates=True, index_col='DATE')

x = Real_N0.index
y = Real_N0['0']
y2 = Upper_N0['0']
y3 = Lower_N0['0']

fig, ax = plt.subplots()
ax.plot(x, y, color='purple', label= r'$N_{0}$', linewidth = 2)
ax.plot(x, y2, color='royalblue', label = "95% CI", zorder=0)
ax.plot(x, y3, color='royalblue', zorder=0)
ax.fill_between(x, y2, y3, color='royalblue')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
plt.gcf().autofmt_xdate()  
plt.title(r'$N_{0}$ 2023 CSP2')
plt.legend(loc = 'upper right')
plt.ylim(2400,3400)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.legend()
plt.savefig('YOUR PATH TO SAVE/N0_CSP2_2023.png')
plt.show()



#####################################
#####################################
############ CSP3 ###################
#####################################
#####################################

### 2011 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2011.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32011_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32011_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2011.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2011.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2011.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2011_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2011_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2011.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2011_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2011_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2011 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2011.png')
plt.show()



### 2012 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2012.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32012_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH TO MAKE/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32012_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2012.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2012.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2012.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2012_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2012_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2012.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2012_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2012_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2012 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2012.png')
plt.show()



### 2013 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2013.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32013_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32013_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2013.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2013.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2013.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2013_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2013_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2013.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2013_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2013_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2013 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2013.png')
plt.show()



### 2014 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2014.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32014_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32014_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2014.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2014.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2014.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2014_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2014_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2014.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2014_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2014_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2014 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2014.png')
plt.show()



### 2015 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2015.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32015_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32015_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2015.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2015.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2015.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2015_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2015_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2015.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2015_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH TO MAKE/N0_CSP3_2015_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2015 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2015.png')
plt.show()



### 2016 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2016.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32016_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32016_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2016.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2016.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2016.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2016_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2016_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2016.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2016_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2016_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2016 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2016.png')
plt.show()



### 2017 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2017.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32017_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32017_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2017.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2017.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2017.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2017_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2017_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2017.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2017_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2017_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2017 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2017.png')
plt.show()



### 2018 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2018.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32018_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2018.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2018.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2018.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2018_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2018_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2018.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2018_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2018_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2018 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2018.png')
plt.show()



### 2019 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32019_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32019_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'CYOUR PATH TO MAKE/N0_Actual_CSP3_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2019.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2019_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2019_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2019.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2019_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2019_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2019 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2019.png')
plt.show()



### 2020 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32020_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH TO MAKE/CSP32020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32020_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2020.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2020_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2020_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2020.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2020_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2020_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2020 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2020.png')
plt.show()



### 2021 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32021_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32021_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
df.to_csv(r'YOUR PATH TO MAKE/N0_Actual_CSP3_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2021.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2021_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2021_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2021.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2021_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2021_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2021 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2021.png')
plt.show()



### 2022 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()
dailySWC.to_csv('YOUR PATH TO MAKE/CSP32022_SWC_resample.csv')

NPIV = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32022_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2022.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2022_Upper.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2022_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2022.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
df2.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2022_Lower.csv')
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2022_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2022 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2022.png')
plt.show()



### 2023 ###
#Import SWC from Soil Water Confidence Plot
Daily_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_DailyAvg_2023.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
SWC_Value = Daily_Avg_SWC['SWC']
dailySWC = SWC_Value.resample('D').mean()

NPIV = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
dailyNPIV.to_csv('YOUR PATH TO MAKE/CSP32023_NPIV_resample.csv')

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc = SWC_Value
lwc = 0.058
soc = 0.0092
N = dailyNPIV

#use parameters to calculate N0
N0__1 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)

df = pd.DataFrame(N0__1)
Fixed_Data = pd.read_csv('YOUR PATH/N0_Actual_CSP3_2023.csv')
N0 = Fixed_Data['0']
Date = Fixed_Data['DATE']

################################
#Import SWC from Soil Water Confidence Plot
Upper_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Upper_2023.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Upper_SWC_Value = Upper_Avg_SWC['SWC']
Upper_dailySWC = Upper_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc1 = Upper_dailySWC

#use parameters to calculate N0
N01 = N / ((0.0808 / ((pwc1 + lwc + soc) + 0.115)) + 0.372)

df1 = pd.DataFrame(N01)
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2023_Upper.csv')
N0_1 = Fixed_Data1['0']
Date1 = Fixed_Data1['DATE']


##################################
#Import SWC from Soil Water Confidence Plot
Lower_Avg_SWC = pd.read_csv('YOUR PATH/CSP3_Lower_2023.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['-9999'])
Lower_SWC_Value = Lower_Avg_SWC['SWC']
Lower_dailySWC = Lower_SWC_Value.resample('D').mean()

### CALCUATE N0 ###

#Define soil parameters, these are dependent on location and need to be found through a soil analysis lab
pwc2 = Lower_dailySWC

#use parameters to calculate N0
N02 = N / ((0.0808 / ((pwc2 + lwc + soc) + 0.115)) + 0.372)

df2 = pd.DataFrame(N02)
Fixed_Data2 = pd.read_csv('YOUR PATH/N0_CSP3_2023_Lower.csv')
N0_2 = Fixed_Data2['0']
Date2 = Fixed_Data2['DATE']

x = Date
y = N0
a = N0_1
b = N0_2

fig, ax = plt.subplots()
ax.plot(x,a, '-', color='royalblue', label = "95% CI", zorder=0)
ax.plot(x,b, '-', color='royalblue', zorder=0)
ax.fill_between(x, a, b, color = 'royalblue')
ax.plot(x,y, color='purple', label= r'$N_{0}$', linewidth = 2)
plt.xlabel('Date')
plt.ylabel(r'$N_{0}$ (cph)')
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
plt.title(r'$N_{0}$ 2023 CSP3')
plt.legend(loc = 'upper right')
plt.ylim(1250,2000)
plt.savefig('YOUR PATH TO SAVE/N0_CSP3_2023.png')
plt.show()