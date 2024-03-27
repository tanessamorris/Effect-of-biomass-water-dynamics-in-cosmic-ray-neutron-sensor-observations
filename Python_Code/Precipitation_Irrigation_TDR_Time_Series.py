import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#Modifications will be required to use data sheets as they are given
#This code is used to create a datetime plots of precipitation, irrigation, and TDR data for the all sites.


############## CSP1 ################
############## 2019 ################

#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP1_2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP1_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP1_2019.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigate_CSP1_2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP1_2019.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP1_2019.csv')
totirri = Irrigation['Irrigation']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

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

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2019.csv')

TDR = pd.read_csv('YOUR PATH/SWC_CSP1_2019.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 180, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP1 2019')
plt.savefig('YOUR PATH TO SAVE/CSP1_2019.png', dpi=300, bbox_inches='tight')
plt.show()

### 2020 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP1_2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('PATH TO MAKE/Test_Precip_CSP1_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP1_2020.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigate_CSP1_2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP1_2020.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP1_2020.csv')
totirri = Irrigation['Irrigate']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

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

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2020.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP1_2020.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper right')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
ax1.legend()
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP1 2020')
plt.savefig('YOUR PATH TO SAVE/CSP1_2020.png', dpi=300, bbox_inches='tight')
plt.show()


### 2021 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP1_2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP1_2021.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigate_CSP1_2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP1_2021.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP1_2021.csv')
totirri = Irrigation['Irrigate']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2021.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP1_2021.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper right')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
ax1.legend()
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP1 2021')
plt.savefig('YOUR PATH TO SAVE/CSP1_2021.png', dpi=300, bbox_inches='tight')
plt.show()


### 2022 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precipitation_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2022_1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_2022.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/CSP1_22_Irrigation.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP1_2022_1.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP1_2022.csv')
totirri = Irrigation['0']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2022.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP1_2022.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP1 2022')
plt.savefig('YOUR PATH TO SAVE/CSP1_2022.png', dpi=300, bbox_inches='tight')
plt.show()


### 2023 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precipitation_2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2023_1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_2023_1.csv')
totrain = Fixed_Data['Precip (mm)']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigation_CSP1_2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP1_2023_1.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP1_2023_1.csv')
totirri = Irrigation['Irrigate (mm)']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2023.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP1_2023.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP1 2023')
plt.savefig('YOUR PATH TO SAVE/CSP1_2023.png', dpi=300, bbox_inches='tight')
plt.show()



###############################
###### CSP2 ###################
###############################

### 2019 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP2_2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP2_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP2_2019.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigate_CSP2_2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP2_2019.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP2_2019.csv')
totirri = Irrigation['Irrigation']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2019.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP2_2019.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 186, 7))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP2 2019')
plt.savefig('YOUR PATH TO SAVE/CSP2_2019.png', dpi=300, bbox_inches='tight')
plt.show()


### 2020 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP2_2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP2_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP2_2020.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigate_CSP2_2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP2_2020.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP2_2020.csv')
totirri = Irrigation['Irrigation']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2020.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP2_2020.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP2 2020')
plt.savefig('YOUR PATH TO SAVE/CSP2_2020.png', dpi=300, bbox_inches='tight')
plt.show()


### 2021 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP2_2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP2_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP2_2021.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigate_CSP2_2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP2_2021.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP2_2021.csv')
totirri = Irrigation['Irrigation']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2021.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP2_2021.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP2 2021')
plt.savefig('YOUR PATH/CSP2_2021.png', dpi=300, bbox_inches='tight')
plt.show()


### 2022 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precipitation_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2022_1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_2022.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/CSP2_22_Irrigation.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP2_2022.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP2_2022.csv')
totirri = Irrigation['Irrigate (mm)']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2022.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP2_2022.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP2 2022')
plt.savefig('YOUR PATH TO SAVE/CSP2_2022.png', dpi=300, bbox_inches='tight')
plt.show()


### 2023 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precipitation_2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2023_1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_2023_1.csv')
totrain = Fixed_Data['Precip (mm)']
Date = Fixed_Data['DATE']

x = Date
y = totrain

######IRRIGATION#######
irrigate = pd.read_csv('YOUR PATH/Irrigation_CSP2_2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
dailyirrigate = irrigate.resample('D').mean()
df1 = pd.DataFrame(dailyirrigate)
df1.to_csv('YOUR PATH TO MAKE/Irrigation_CSP2_2023.csv')
Irrigation = pd.read_csv('YOUR PATH/Irrigation_CSP2_2023.csv')
totirri = Irrigation['Irrigate (mm)']
Date = Irrigation['DATE']

m = Date
n = totirri

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2023.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP2_2023.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(x, y, color = 'turquoise', width = 1.5, label = 'Precipitation')
lns2 = ax1.bar(m,n, color = 'darkgreen', width=2, label = 'Irrigation')
lns3 = ax2.plot(a,b, color = 'darkred', label = 'TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.legend(loc='upper left')
ax1.tick_params(axis="y")
ax1.set_ylim(0,90)
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(np.linspace(1, 220, 8))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP2 2023')
plt.savefig('YOUR PATH TO SAVE/CSP2_2023.png', dpi=300, bbox_inches='tight')
plt.show()



##############################
###### CSP3 ##################
##############################

### 2011 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2011.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2011.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2011.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2011.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2011')
plt.savefig('YOUR PATH TO SAVE/CSP3_2011.png', dpi=300, bbox_inches='tight')
plt.show()


### 2012 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2012.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2012.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2012.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2012.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2012')
plt.savefig('YOUR PATH TO SAVE/CSP3_2012.png', dpi=300, bbox_inches='tight')
plt.show()


### 2013 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2013.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2013.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2013.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2013.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2013')
plt.savefig('YOUR PATH TO SAVE/CSP3_2013.png', dpi=300, bbox_inches='tight')
plt.show()


### 2014 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2014.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2014.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2014.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2014.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2014')
plt.savefig('YOUR PATH TO SAVE/CSP3_2014.png', dpi=300, bbox_inches='tight')
plt.show()


### 2015 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2015.csv')
Fixed_Data = pd.read_csv('CYOUR PATH/Test_Precip_CSP3_2015.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2015.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2015.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2015')
plt.savefig('YOUR PATH TO SAVE/CSP3_2015.png', dpi=300, bbox_inches='tight')
plt.show()


### 2016 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2016.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2016.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2016.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2016.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2016')
plt.savefig('YOUR PATH TO SAVE/CSP3_2016.png', dpi=300, bbox_inches='tight')
plt.show()


### 2017 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2017.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2017.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2017.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2017.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2017')
plt.savefig('YOUR PATH TO SAVE/CSP3_2017.png', dpi=300, bbox_inches='tight')
plt.show()


### 2018 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2018.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2018.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2018.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2018.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2018')
plt.savefig('YOUR PATH TO SAVE/CSP3_2018.png', dpi=300, bbox_inches='tight')
plt.show()


### 2019 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2019.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH TO MAKE/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2019.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2019.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2019')
plt.savefig('YOUR PATH TO SAVE/CSP3_2019.png', dpi=300, bbox_inches='tight')
plt.show()


### 2020 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_CSP3_2020.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2020.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2020.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2020')
plt.savefig('YOUR PATH TO SAVE/CSP3_2020.png', dpi=300, bbox_inches='tight')
plt.show()


### 2021 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precip_CSP3_2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

P_PI_F_1_1_1 = rain['P_PI_F_1_1_1']
P_PI_F_2_2_1 = rain['P_PI_F_2_2_1']

averageprecip = (P_PI_F_1_1_1 + P_PI_F_2_2_1)/2

dailyprecip = averageprecip.resample('D').sum()
df = pd.DataFrame(dailyprecip)
df.to_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH TO MAKE/Test_Precip_CSP3_2021.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2021.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2021.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2021')
plt.savefig('YOUR PATH TO SAVE/CSP3_2021.png', dpi=300, bbox_inches='tight')
plt.show()


### 2022 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH/Precipitation_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2022_1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_2022.csv')
totrain = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2022.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2022.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2022')
plt.savefig('YOUR PATH TO SAVE/CSP3_2022.png', dpi=300, bbox_inches='tight')
plt.show()


### 2023 ###
#######Rainfall Data##########
rain = pd.read_csv('YOUR PATH TO MAKE/Precipitation_2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

rain.to_csv('YOUR PATH TO MAKE/Test_Precip_2023_1.csv')
Fixed_Data = pd.read_csv('YOUR PATH/Test_Precip_2023_1.csv')
totrain = Fixed_Data['Precip (mm)']
Date = Fixed_Data['DATE']

x = Date
y = totrain

#########TDR##########
#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_F_1_1_1']/1.42
SWC_1_2_1 = GWC['SWC_F_1_2_1']/1.42
SWC_1_3_1 = GWC['SWC_F_1_3_1']/1.42
SWC_1_4_1 = GWC['SWC_F_1_4_1']/1.42
SWC_2_1_1 = GWC['SWC_F_2_1_1']/1.42
SWC_2_2_1 = GWC['SWC_F_2_2_1']/1.42
SWC_2_3_1 = GWC['SWC_F_2_3_1']/1.42
SWC_2_4_1 = GWC['SWC_F_2_4_1']/1.42
SWC_3_1_1 = GWC['SWC_F_3_1_1']/1.42
SWC_3_2_1 = GWC['SWC_F_3_2_1']/1.42
SWC_3_3_1 = GWC['SWC_F_3_3_1']/1.42
SWC_3_4_1 = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

dailywavgSWC = twosenwavgall.resample('D').mean() 

df2 = pd.DataFrame(dailywavgSWC)
df2.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2023.csv')
TDR = pd.read_csv('YOUR PATH/SWC_CSP3_2023.csv')
SWC = TDR['0']
Date = TDR['DATE']

a = Date
b = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x, y, color = 'turquoise', width = 1.5, label='Precipitation')
ax2.plot(a,b, color = 'darkred', label='TDR')
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Applied (mm)")
ax1.set_ylim(0,90)
ax1.tick_params(axis="y")
ax1.legend(loc='upper left')
color_VWC = "#8B0000"
ax2.set_ylabel("SWC (cm\u00b3/cm\u00b3)", color=color_VWC)
ax2.tick_params(axis="y", labelcolor=color_VWC)
ax2.set_ylim(0.1, 0.33)
ax2.legend(loc='upper right')
plt.xticks(np.linspace(1, 220, 12))
fig.autofmt_xdate()
plt.title('Precipitation, Irrigation, TDR CSP3 2023')
plt.savefig('YOUR PATH TO SAVE/CSP3_2023.png', dpi=300, bbox_inches='tight')
plt.show()