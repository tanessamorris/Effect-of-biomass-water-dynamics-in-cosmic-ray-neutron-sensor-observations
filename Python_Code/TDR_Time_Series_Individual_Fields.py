import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

#Import gravimetric water content from the TDR sensors
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the .csv sheet
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

#calculating a vertically weighted average of the volumetric data calculations above
twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

#Define the data-frame more clearly for my own use
df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP1_2019.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

#plot the data in line graph
x = Date
y = SWC
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 180, 12))
plt.ylim(0.1, 0.325)
fig.autofmt_xdate()
plt.title('TDR SWC Time Series CSP1 2019')
plt.savefig('YOUR PATH TO SAVE/SWC_CSP1_2019.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_1_1_1']/1.42
SWC_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_1_2_1']/1.42
SWC_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_1_3_1']/1.42
SWC_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_1_4_1']/1.42
SWC_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_2_1_1']/1.42
SWC_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_2_2_1']/1.42
SWC_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_2_3_1']/1.42
SWC_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_2_4_1']/1.42
SWC_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_3_1_1']/1.42
SWC_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_3_2_1']/1.42
SWC_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_3_3_1']/1.42
SWC_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_3_4_1']/1.42

twosenwavgall = ((((SWC_1_1_1 + SWC_2_1_1 + SWC_3_1_1)/3)*0.75) + (((SWC_1_2_1 + SWC_2_2_1 + SWC_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP1_2020.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP1 2020')
plt.savefig('YOUR PATH TO SAVE/SWC_CSP1_2020.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP1_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP1_2021.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP1 2021')
plt.savefig('YOUR PATH TO SAVE/SWC_CSP1_2021.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE SWC_CSP1_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP1_2022.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP1 2022')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP1_2022.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE SWC_CSP1_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP1_2023.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP1 2023')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP1_2023.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP2_2019.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 180, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP2 2019')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP2_2019.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP2_2020.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP2 2020')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP2_2020.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP2_2021.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP2 2021')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP2_2021.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP2_2022.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP2 2022')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP2_2022.png')
plt.show()


#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_4_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_4_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_4_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1)/3)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1)/3)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP2_2023.csv')

Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP2_2023.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']
x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 220, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP2 2023')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP2_2023.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2011.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2011.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2011')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2011.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2012.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2012.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2012')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2012.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2013.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2013.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2013')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2013.png')
plt.show()


#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2014.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2014.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2014')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2014.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2015.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2015.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2015')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2015.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2016.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2016.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2016')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2016.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2017.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2017.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2017')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2017.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2018.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2018.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2018')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2018.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2019.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2019')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2019.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2020.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2020')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2020.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_PI_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2021.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2021')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2021.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2022.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2022')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2022.png')
plt.show()



#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_PI_F_1_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_1_1']/1.42
SWC_PI_F_1_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_2_1']/1.42
SWC_PI_F_1_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_3_1']/1.42
SWC_PI_F_1_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_4_1']/1.42
SWC_PI_F_1_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_1_5_1']/1.42
SWC_PI_F_2_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_1_1']/1.42
SWC_PI_F_2_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_2_1']/1.42
SWC_PI_F_2_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_3_1']/1.42
SWC_PI_F_2_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_4_1']/1.42
SWC_PI_F_2_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_2_5_1']/1.42
SWC_PI_F_3_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_1_1']/1.42
SWC_PI_F_3_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_2_1']/1.42
SWC_PI_F_3_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_3_1']/1.42
SWC_PI_F_3_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_4_1']/1.42
SWC_PI_F_3_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_3_5_1']/1.42
SWC_PI_F_4_1_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_1_1']/1.42
SWC_PI_F_4_2_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_2_1']/1.42
SWC_PI_F_4_3_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_3_1']/1.42
SWC_PI_F_4_4_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_4_1']/1.42
SWC_PI_F_4_5_1 = GWC['GravimetricWaterContent'] = GWC['SWC_F_4_5_1']/1.42

twosenwavgall = ((((SWC_PI_F_1_1_1 + SWC_PI_F_2_1_1 + SWC_PI_F_3_1_1 + SWC_PI_F_4_1_1)/4)*0.75) + (((SWC_PI_F_1_2_1 + SWC_PI_F_2_2_1 + SWC_PI_F_3_2_1 + SWC_PI_F_4_2_1)/4)*0.25)) / 100

#Calculate the daily weighted averages from the weighted depth above
dailywavgSWC = twosenwavgall.resample('D').mean() 

df = pd.DataFrame(dailywavgSWC)
df.to_csv('YOUR PATH TO MAKE/SWC_CSP3_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/SWC_CSP3_2023.csv')
SWC = Fixed_Data['0']
Date = Fixed_Data['DATE']

x = Date
y = SWC

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, ax = plt.subplots()
plt.plot(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel('SWC (cm\u00b3/cm\u00b3)')
plt.xticks(np.linspace(4, 225, 12))
fig.autofmt_xdate()
plt.ylim(0.1, 0.325)
plt.title('TDR SWC Time Series CSP3 2023')
plt.savefig('YOUR PATH TO SAVE/TDR_Field_Time_Series/SWC_CSP3_2023.png')
plt.show()