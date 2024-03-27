import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

#Date-time plots for Npvi

########################################
################ CSP1 ##################
########################################

### 2019 ###
#Import neutron count data
NPIV = pd.read_csv('YOUR PATH/CSP12019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
#Creating the same file again is done to ensure the dates input properly for plotting
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2019.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 180, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2019 CSP1')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP1_2019.png')
plt.show()


### 2020 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP12020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2020.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2020 CSP1')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP1_2020.png')
plt.show()


### 2021 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP12021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2021.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2021 CSP1')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP1_2021.png')
plt.show()


### 2022 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP12022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2022.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2022 CSP1')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP1_2022.png')
plt.show()


### 2023 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP12023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2023.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2023 CSP1')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP1_2023.png')
plt.show()


########################################
################ CSP2 ##################
########################################

### 2019 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP22019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2019.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 180, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2019 CSP2')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP2_2019.png')
plt.show()


### 2020 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP22020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2020.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2020 CSP2')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP2_2020.png')
plt.show()


### 2021 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP22021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2021.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2021 CSP2')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP2_2021.png')
plt.show()


### 2022 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP22022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2022.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2022 CSP2')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP2_2022.png')
plt.show()


### 2023 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP22023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2023.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(1400,1900)
plt.title(r'N$_{pvi}$ Time-Series 2023 CSP2')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP2_2023.png')
plt.show()


########################################
################ CSP3 ##################
########################################

### 2011 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2011.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2011.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 200, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2011 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2011.png')
plt.show()


### 2012 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2012.csv')
Fixed_Data = pd.read_csv('CYOUR PATH/NPIV_CSP3_2012.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2012 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2012.png')
plt.show()


### 2013 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2013.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2013.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2013 CSP3')
plt.savefig('YOUR PATH TO MAKE/NPIV_CSP3_2013.png')
plt.show()


### 2014 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2014.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2014.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2014 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2014.png')
plt.show()


### 2015 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2015.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2015.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2015 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2015.png')
plt.show()


### 2016 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2016.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2016.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2016 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2016.png')
plt.show()


### 2017 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2017.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2017.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2017 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2017.png')
plt.show()


### 2018 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2018.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2018.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2018 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2018.png')
plt.show()


### 2019 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2019.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2019 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2019.png')
plt.show()


### 2020 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2020.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2020 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2020.png')
plt.show()


### 2021 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2021.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2021 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2021.png')
plt.show()


### 2022 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2022.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2022 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2022.png')
plt.show()


### 2023 ###
#Date-time plots for Npvi
NPIV = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])

#Define the data for ease in calculations later
NPIVCount = NPIV['NPIV']

#Calculate the daily filtered neutron counts
dailyNPIV = NPIVCount.resample('D').mean()
Npvi = dailyNPIV
df = pd.DataFrame(Npvi)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2023.csv')
Npvi = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']

x = Date
y = Npvi

fig, ax = plt.subplots()
plt.scatter(x, y, color = 'purple')
plt.xlabel('Date')
plt.ylabel(r'N$_{pvi}$ (cph)')
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
plt.ylim(800,1100)
plt.title(r'N$_{pvi}$ Time-Series 2023 CSP3')
plt.savefig('YOUR PATH TO SAVE/NPIV_CSP3_2023.png')
plt.show()