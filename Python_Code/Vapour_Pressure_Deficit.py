import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

## Import temperature data and convert to Kelvin
# Use Clausius-Clapeyron equation to solve for saturated vapour pressure

#import environmental data
CSP1_Data = pd.read_csv('YOUR PATH/CSP1_Enviro_data.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP2_Data = pd.read_csv('YOUR PATH/CSP2_Enviro_data.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP3_Data = pd.read_csv('YOUR PATH/CSP3_Enviro_data.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP1_SWC = pd.read_csv('YOUR PATH/CSP1_SWC.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP2_SWC = pd.read_csv('YOUR PATH/CSP2_SWC.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP3_SWC = pd.read_csv('YOUR PATH/CSP3_SWC.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP1_SWC_Season = pd.read_csv('YOUR PATH/CSP1_SWC_Season.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP2_SWC_Season = pd.read_csv('YOUR PATH/CSP2_SWC_Season.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])
CSP3_SWC_Season = pd.read_csv('YOUR PATH/CSP3_SWC_Season.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP1_Date = CSP1_Data['DATE']
CSP2_Date = CSP2_Data['DATE']
CSP3_Date = CSP3_Data['DATE']

CSP1_Data.set_index('DATE', inplace=True)
CSP2_Data.set_index('DATE', inplace=True)
CSP3_Data.set_index('DATE', inplace=True)

###### Data from Npiv sensor #######
#Temperature data
CSP1_temp = CSP1_Data['Temperature']+273.15
CSP1_temperature = CSP1_temp.resample('D').mean() 
CSP2_temp = CSP2_Data['Temperature']+273.15
CSP2_temperature = CSP2_temp.resample('D').mean()
CSP3_temp = CSP3_Data['Temperature']+273.15
CSP3_temperature = CSP3_temp.resample('D').mean()

#Air pressure data
CSP1_pressure = CSP1_Data['Pressure']
CSP1_Pressure = CSP1_pressure.resample('D').mean()
CSP2_pressure = CSP2_Data['Pressure']
CSP2_Pressure = CSP2_pressure.resample('D').mean()
CSP3_pressure = CSP3_Data['Pressure']
CSP3_Pressure = CSP3_pressure.resample('D').mean()

#Humidity data
CSP1_humidity = CSP1_Data['Humidity']
CSP1_Humidity = CSP1_humidity.resample('D').mean()
CSP2_humidity = CSP2_Data['Humidity']
CSP2_Humidity = CSP2_humidity.resample('D').mean()
CSP3_humidity = CSP3_Data['Humidity']
CSP3_Humidity = CSP3_humidity.resample('D').mean()

#NPIV data
CSP1_NPIV = CSP1_Data['Npiv']
CSP1_Npiv = CSP1_NPIV.resample('D').mean()
CSP2_NPIV = CSP2_Data['Npiv']
CSP2_Npiv = CSP2_NPIV.resample('D').mean()
CSP3_NPIV = CSP3_Data['Npiv']
CSP3_Npiv = CSP3_NPIV.resample('D').mean()

CSP1_SWC.set_index('DATE', inplace=True)
CSP2_SWC.set_index('DATE', inplace=True)
CSP3_SWC.set_index('DATE', inplace=True)

CSP1_SWC_Season.set_index('DATE', inplace=True)
CSP2_SWC_Season.set_index('DATE', inplace=True)
CSP3_SWC_Season.set_index('DATE', inplace=True)

####### Mesonet data
CSP1_temp_meso = CSP1_Data['Meso_Temp']+273.15
CSP1_temperature_meso = CSP1_temp_meso.resample('D').mean() 
CSP2_temp_meso = CSP2_Data['Meso_Temp']+273.15
CSP2_temperature_meso = CSP2_temp_meso.resample('D').mean()
CSP3_temp_meso = CSP3_Data['Meso_Temp']+273.15
CSP3_temperature_meso = CSP3_temp_meso.resample('D').mean()

#Humidity data
CSP1_humidity_meso = CSP1_Data['Meso_Hum']
CSP1_Humidity_meso = CSP1_humidity_meso.resample('D').mean()
CSP2_humidity_meso = CSP2_Data['Meso_Hum']
CSP2_Humidity_meso = CSP2_humidity_meso.resample('D').mean()
CSP3_humidity_meso = CSP3_Data['Meso_Hum']
CSP3_Humidity_meso = CSP3_humidity_meso.resample('D').mean()

######## Season Data - from mesonet
CSP1_temp_meso_season = CSP1_Data['Temp_Season']+273.15
CSP1_temperature_meso_season = CSP1_temp_meso_season.resample('D').mean()
CSP2_temp_meso_season = CSP2_Data['Temp_Season']+273.15
CSP2_temperature_meso_season = CSP2_temp_meso_season.resample('D').mean()
CSP3_temp_meso_season = CSP3_Data['Temp_Season']+273.15
CSP3_temperature_meso_season = CSP3_temp_meso_season.resample('D').mean()

#Humidity data
CSP1_humidity_meso_season = CSP1_Data['Hum_Season']
CSP1_Humidity_meso_season = CSP1_humidity_meso_season.resample('D').mean()
CSP2_humidity_meso_season = CSP2_Data['Hum_Season']
CSP2_Humidity_meso_season = CSP2_humidity_meso_season.resample('D').mean()
CSP3_humidity_meso_season = CSP3_Data['Hum_Season']
CSP3_Humidity_meso_season = CSP3_humidity_meso_season.resample('D').mean()

#NPIV data
CSP1_NPIV_season = CSP1_Data['Npiv_Season']
CSP1_Npiv_season = CSP1_NPIV_season.resample('D').mean()
CSP2_NPIV_season = CSP2_Data['Npiv_Season']
CSP2_Npiv_season = CSP2_NPIV_season.resample('D').mean()
CSP3_NPIV_season = CSP3_Data['Npiv_Season']
CSP3_Npiv_season = CSP3_NPIV_season.resample('D').mean()

#Calculate N0
#SWC data
SWC_1_1_1_CSP1 = CSP1_SWC['SWC_1_1_1']/1.42
SWC_1_2_1_CSP1 = CSP1_SWC['SWC_1_2_1']/1.42
SWC_1_3_1_CSP1 = CSP1_SWC['SWC_1_3_1']/1.42
SWC_1_4_1_CSP1 = CSP1_SWC['SWC_1_4_1']/1.42
SWC_2_1_1_CSP1 = CSP1_SWC['SWC_2_1_1']/1.42
SWC_2_2_1_CSP1 = CSP1_SWC['SWC_2_2_1']/1.42
SWC_2_3_1_CSP1 = CSP1_SWC['SWC_2_3_1']/1.42
SWC_2_4_1_CSP1 = CSP1_SWC['SWC_2_4_1']/1.42
SWC_3_1_1_CSP1 = CSP1_SWC['SWC_3_1_1']/1.42
SWC_3_2_1_CSP1 = CSP1_SWC['SWC_3_2_1']/1.42
SWC_3_3_1_CSP1 = CSP1_SWC['SWC_3_3_1']/1.42
SWC_3_4_1_CSP1 = CSP1_SWC['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1 = ((((SWC_1_1_1_CSP1 + SWC_2_1_1_CSP1 + SWC_3_1_1_CSP1)/3)*0.75) + (((SWC_1_2_1_CSP1 + SWC_2_2_1_CSP1 + SWC_3_2_1_CSP1)/3)*0.25)) / 100
dailywavgSWC_CSP1 = twosenwavgall_CSP1.resample('D').mean() 

pwc_CSP1 = dailywavgSWC_CSP1
lwc = 0.058
soc = 0.0092
N_CSP1 = CSP1_Npiv

#use parameters to calculate N0
N0_CSP1 = N_CSP1 / ((0.0808 / ((pwc_CSP1 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1 = N0_CSP1.interpolate(method = 'linear', limit_direction = 'forward')

SWC_1_1_1_CSP2 = CSP2_SWC['SWC_1_1_1']/1.42
SWC_1_2_1_CSP2 = CSP2_SWC['SWC_1_2_1']/1.42
SWC_1_3_1_CSP2 = CSP2_SWC['SWC_1_3_1']/1.42
SWC_1_4_1_CSP2 = CSP2_SWC['SWC_1_4_1']/1.42
SWC_2_1_1_CSP2 = CSP2_SWC['SWC_2_1_1']/1.42
SWC_2_2_1_CSP2 = CSP2_SWC['SWC_2_2_1']/1.42
SWC_2_3_1_CSP2 = CSP2_SWC['SWC_2_3_1']/1.42
SWC_2_4_1_CSP2 = CSP2_SWC['SWC_2_4_1']/1.42
SWC_3_1_1_CSP2 = CSP2_SWC['SWC_3_1_1']/1.42
SWC_3_2_1_CSP2 = CSP2_SWC['SWC_3_2_1']/1.42
SWC_3_3_1_CSP2 = CSP2_SWC['SWC_3_3_1']/1.42
SWC_3_4_1_CSP2 = CSP2_SWC['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2 = ((((SWC_1_1_1_CSP2 + SWC_2_1_1_CSP2 + SWC_3_1_1_CSP2)/3)*0.75) + (((SWC_1_2_1_CSP2 + SWC_2_2_1_CSP2 + SWC_3_2_1_CSP2)/3)*0.25)) / 100
dailywavgSWC_CSP2 = twosenwavgall_CSP2.resample('D').mean()

pwc_CSP2 = dailywavgSWC_CSP2
N_CSP2 = CSP2_Npiv

#use parameters to calculate N0
N0_CSP2 = N_CSP2 / ((0.0808 / ((pwc_CSP2 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2 = N0_CSP2.interpolate(method = 'linear', limit_direction = 'forward')

SWC_1_1_1_CSP3 = CSP3_SWC['SWC_1_1_1']/1.42
SWC_1_2_1_CSP3 = CSP3_SWC['SWC_1_2_1']/1.42
SWC_1_3_1_CSP3 = CSP3_SWC['SWC_1_3_1']/1.42
SWC_1_4_1_CSP3 = CSP3_SWC['SWC_1_4_1']/1.42
SWC_2_1_1_CSP3 = CSP3_SWC['SWC_2_1_1']/1.42
SWC_2_2_1_CSP3 = CSP3_SWC['SWC_2_2_1']/1.42
SWC_2_3_1_CSP3 = CSP3_SWC['SWC_2_3_1']/1.42
SWC_2_4_1_CSP3 = CSP3_SWC['SWC_2_4_1']/1.42
SWC_3_1_1_CSP3 = CSP3_SWC['SWC_3_1_1']/1.42
SWC_3_2_1_CSP3 = CSP3_SWC['SWC_3_2_1']/1.42
SWC_3_3_1_CSP3 = CSP3_SWC['SWC_3_3_1']/1.42
SWC_3_4_1_CSP3 = CSP3_SWC['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3 = ((((SWC_1_1_1_CSP3 + SWC_2_1_1_CSP3 + SWC_3_1_1_CSP3)/3)*0.75) + (((SWC_1_2_1_CSP3 + SWC_2_2_1_CSP3 + SWC_3_2_1_CSP3)/3)*0.25)) / 100
dailywavgSWC_CSP3 = twosenwavgall_CSP3.resample('D').mean()

pwc_CSP3 = dailywavgSWC_CSP3
N_CSP3 = CSP3_Npiv

#use parameters to calculate N0
N0_CSP3 = N_CSP3 / ((0.0808 / ((pwc_CSP3 + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3 = N0_CSP3.interpolate(method = 'linear', limit_direction = 'forward')

#Calculate N0 in season
#SWC data
SWC_1_1_1_CSP1_Season = CSP1_SWC_Season['SWC_1_1_1']/1.42
SWC_1_2_1_CSP1_Season = CSP1_SWC_Season['SWC_1_2_1']/1.42
SWC_1_3_1_CSP1_Season = CSP1_SWC_Season['SWC_1_3_1']/1.42
SWC_1_4_1_CSP1_Season = CSP1_SWC_Season['SWC_1_4_1']/1.42
SWC_2_1_1_CSP1_Season = CSP1_SWC_Season['SWC_2_1_1']/1.42
SWC_2_2_1_CSP1_Season = CSP1_SWC_Season['SWC_2_2_1']/1.42
SWC_2_3_1_CSP1_Season = CSP1_SWC_Season['SWC_2_3_1']/1.42
SWC_2_4_1_CSP1_Season = CSP1_SWC_Season['SWC_2_4_1']/1.42
SWC_3_1_1_CSP1_Season = CSP1_SWC_Season['SWC_3_1_1']/1.42
SWC_3_2_1_CSP1_Season = CSP1_SWC_Season['SWC_3_2_1']/1.42
SWC_3_3_1_CSP1_Season = CSP1_SWC_Season['SWC_3_3_1']/1.42
SWC_3_4_1_CSP1_Season = CSP1_SWC_Season['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP1_Season = ((((SWC_1_1_1_CSP1_Season + SWC_2_1_1_CSP1_Season + SWC_3_1_1_CSP1_Season)/3)*0.75) + (((SWC_1_2_1_CSP1_Season + SWC_2_2_1_CSP1_Season + SWC_3_2_1_CSP1_Season)/3)*0.25)) / 100
dailywavgSWC_CSP1_Season = twosenwavgall_CSP1_Season.resample('D').mean()

pwc_CSP1_Season = dailywavgSWC_CSP1_Season
lwc = 0.058
soc = 0.0092
N_CSP1_Season = CSP1_Npiv_season

#use parameters to calculate N0
N0_CSP1_Season = N_CSP1_Season / ((0.0808 / ((pwc_CSP1_Season + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP1_Season = N0_CSP1_Season.interpolate(method = 'linear', limit_direction = 'forward')

SWC_1_1_1_CSP2_Season = CSP2_SWC_Season['SWC_1_1_1']/1.42
SWC_1_2_1_CSP2_Season = CSP2_SWC_Season['SWC_1_2_1']/1.42
SWC_1_3_1_CSP2_Season = CSP2_SWC_Season['SWC_1_3_1']/1.42
SWC_1_4_1_CSP2_Season = CSP2_SWC_Season['SWC_1_4_1']/1.42
SWC_2_1_1_CSP2_Season = CSP2_SWC_Season['SWC_2_1_1']/1.42
SWC_2_2_1_CSP2_Season = CSP2_SWC_Season['SWC_2_2_1']/1.42
SWC_2_3_1_CSP2_Season = CSP2_SWC_Season['SWC_2_3_1']/1.42
SWC_2_4_1_CSP2_Season = CSP2_SWC_Season['SWC_2_4_1']/1.42
SWC_3_1_1_CSP2_Season = CSP2_SWC_Season['SWC_3_1_1']/1.42
SWC_3_2_1_CSP2_Season = CSP2_SWC_Season['SWC_3_2_1']/1.42
SWC_3_3_1_CSP2_Season = CSP2_SWC_Season['SWC_3_3_1']/1.42
SWC_3_4_1_CSP2_Season = CSP2_SWC_Season['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP2_Season = ((((SWC_1_1_1_CSP2_Season + SWC_2_1_1_CSP2_Season + SWC_3_1_1_CSP2_Season)/3)*0.75) + (((SWC_1_2_1_CSP2_Season + SWC_2_2_1_CSP2_Season + SWC_3_2_1_CSP2_Season)/3)*0.25)) / 100
dailywavgSWC_CSP2_Season = twosenwavgall_CSP2_Season.resample('D').mean()

pwc_CSP2_Season = dailywavgSWC_CSP2_Season
N_CSP2_Season = CSP2_Npiv_season

#use parameters to calculate N0
N0_CSP2_Season = N_CSP2_Season / ((0.0808 / ((pwc_CSP2_Season + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP2_Season = N0_CSP2_Season.interpolate(method = 'linear', limit_direction = 'forward')

SWC_1_1_1_CSP3_Season = CSP3_SWC_Season['SWC_1_1_1']/1.42
SWC_1_2_1_CSP3_Season = CSP3_SWC_Season['SWC_1_2_1']/1.42
SWC_1_3_1_CSP3_Season = CSP3_SWC_Season['SWC_1_3_1']/1.42
SWC_1_4_1_CSP3_Season = CSP3_SWC_Season['SWC_1_4_1']/1.42
SWC_2_1_1_CSP3_Season = CSP3_SWC_Season['SWC_2_1_1']/1.42
SWC_2_2_1_CSP3_Season = CSP3_SWC_Season['SWC_2_2_1']/1.42
SWC_2_3_1_CSP3_Season = CSP3_SWC_Season['SWC_2_3_1']/1.42
SWC_2_4_1_CSP3_Season = CSP3_SWC_Season['SWC_2_4_1']/1.42
SWC_3_1_1_CSP3_Season = CSP3_SWC_Season['SWC_3_1_1']/1.42
SWC_3_2_1_CSP3_Season = CSP3_SWC_Season['SWC_3_2_1']/1.42
SWC_3_3_1_CSP3_Season = CSP3_SWC_Season['SWC_3_3_1']/1.42
SWC_3_4_1_CSP3_Season = CSP3_SWC_Season['SWC_3_4_1']/1.42

#Calculate the weighted depths for just the top 2 - 10cm and 25cm
twosenwavgall_CSP3_Season = ((((SWC_1_1_1_CSP3_Season + SWC_2_1_1_CSP3_Season + SWC_3_1_1_CSP3_Season)/3)*0.75) + (((SWC_1_2_1_CSP3_Season + SWC_2_2_1_CSP3_Season + SWC_3_2_1_CSP3_Season)/3)*0.25)) / 100
dailywavgSWC_CSP3_Season = twosenwavgall_CSP3_Season.resample('D').mean()

pwc_CSP3_Season = dailywavgSWC_CSP3_Season
N_CSP3_Season = CSP3_Npiv_season

#use parameters to calculate N0
N0_CSP3_Season = N_CSP3_Season / ((0.0808 / ((pwc_CSP3_Season + lwc + soc) + 0.115)) + 0.372)
N0interp_CSP3_Season = N0_CSP3_Season.interpolate(method = 'linear', limit_direction = 'forward')


#Calculate SVP
CSP1_Es = (6.112*np.exp((17.67*(CSP1_temperature-273.15))/(243.5 + (CSP1_temperature-273.15))))*100
CSP1_Es.to_csv('YOUR PATH TO MAKE/CSP1_Es_1.csv')
CSP1_Es_1 = pd.read_csv('YOUR PATH/CSP1_Es_1.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP2_Es = (6.112*np.exp((17.67*(CSP2_temperature-273.15))/(243.5 + (CSP2_temperature-273.15))))*100
CSP2_Es.to_csv('YOUR PATH TO MAKE/CSP2_Es_1.csv')
CSP2_Es_1 = pd.read_csv('YOUR PATH/CSP2_Es_1.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP3_Es = (6.112*np.exp((17.67*(CSP3_temperature-273.15))/(243.5 + (CSP3_temperature-273.15))))*100
CSP3_Es.to_csv('YOUR PATH TO MAKE/CSP3_Es_1.csv')
CSP3_Es_1 = pd.read_csv('YOUR PATH/CSP3_Es_1.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

#Calculate SVP in season
CSP1_Es_Season = (6.112*np.exp((17.67*(CSP1_temperature_meso_season-273.15))/(243.5 + (CSP1_temperature_meso_season-273.15))))*100
CSP1_Es_Season.to_csv('YOUR PATH TO MAKE/CSP1_Es_Season.csv')
CSP1_Es_Season_1 = pd.read_csv('YOUR PATH/CSP1_Es_Season.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP2_Es_Season = (6.112*np.exp((17.67*(CSP2_temperature_meso_season-273.15))/(243.5 + (CSP2_temperature_meso_season-273.15))))*100
CSP2_Es_Season.to_csv('YOUR PATH TO MAKE/CSP2_Es_Season.csv')
CSP2_Es_Season_1 = pd.read_csv('YOUR PATH/CSP2_Es_Season.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP3_Es_Season = (6.112*np.exp((17.67*(CSP3_temperature_meso_season-273.15))/(243.5 + (CSP3_temperature_meso_season-273.15))))*100
CSP3_Es_Season.to_csv('YOUR PATH TO MAKE/CSP3_Es_Season.csv')
CSP3_Es_Season_1 = pd.read_csv('YOUR PATH/CSP3_Es_Season.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

#Calculate SVP meso
CSP1_Es_meso = (6.112*np.exp((17.67*(CSP1_temperature_meso-273.15))/(243.5 + (CSP1_temperature_meso-273.15))))*100
CSP1_Es_meso.to_csv('YOUR PATH TO MAKE/CSP1_Es_meso.csv')
CSP1_Es_meso_1 = pd.read_csv('YOUR PATH/CSP1_Es_meso.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP2_Es_meso = (6.112*np.exp((17.67*(CSP2_temperature_meso-273.15))/(243.5 + (CSP2_temperature_meso-273.15))))*100
CSP2_Es_meso.to_csv('YOUR PATH TO MAKE/CSP2_Es_meso.csv')
CSP2_Es_meso_1 = pd.read_csv('YOUR PATH/CSP2_Es_meso.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])

CSP3_Es_meso = (6.112*np.exp((17.67*(CSP3_temperature_meso-273.15))/(243.5 + (CSP3_temperature_meso-273.15))))*100
CSP3_Es_meso.to_csv('YOUR PATH TO MAKE/CSP3_Es_meso.csv')
CSP3_Es_meso_1 = pd.read_csv('YOUR PATH/CSP3_Es_meso.csv', parse_dates= ['DATE'],index_col=False ,na_values=['-9999'])


#Calculate VPD
CSP1_VPD = ((((100 - CSP1_Humidity)/100) * CSP1_Es)/1000)
CSP1_VPD.to_csv('YOUR PATH TO MAKE/CSP1_VPD.csv')

CSP2_VPD = ((((100 - CSP2_Humidity)/100) * CSP2_Es)/1000)
CSP2_VPD.to_csv('YOUR PATH TO MAKE/CSP2_VPD.csv')

CSP3_VPD = ((((100 - CSP3_Humidity)/100) * CSP3_Es)/1000)
CSP3_VPD.to_csv('YOUR PATH TO MAKE/CSP3_VPD.csv')

#Calculate VPD in season
CSP1_VPD_Season = ((((100 - CSP1_Humidity_meso_season)/100) * CSP1_Es_Season)/1000)
CSP1_VPD_Season.to_csv('YOUR PATH TO MAKE/CSP1_VPD_Season.csv')

CSP2_VPD_Season = ((((100 - CSP2_Humidity_meso_season)/100) * CSP2_Es_Season)/1000)
CSP2_VPD_Season.to_csv('YOUR PATH TO MAKE/CSP2_VPD_Season.csv')

CSP3_VPD_Season = ((((100 - CSP3_Humidity_meso_season)/100) * CSP3_Es_Season)/1000)
CSP3_VPD_Season.to_csv('YOUR PATH TO MAKE/CSP3_VPD_Season.csv')

#Calculate VPD meso
CSP1_VPD_meso = ((((100 - CSP1_Humidity_meso)/100) * CSP1_Es_meso)/1000)
CSP1_VPD_meso.to_csv('YOUR PATH TO MAKE/CSP1_VPD_meso.csv')

CSP2_VPD_meso = ((((100 - CSP2_Humidity_meso)/100) * CSP2_Es_meso)/1000)
CSP2_VPD_meso.to_csv('YOUR PATH TO MAKE/CSP2_VPD_meso.csv')

CSP3_VPD_meso = ((((100 - CSP3_Humidity_meso)/100) * CSP3_Es_meso)/1000)
CSP3_VPD_meso.to_csv('YOUR PATH TO MAKE/CSP3_VPD_meso.csv')

y_CSP1 = N0_CSP1
y_CSP2 = N0_CSP2
y_CSP3 = N0_CSP3
x_CSP1 = CSP1_VPD
x_CSP2 = CSP2_VPD
x_CSP3 = CSP3_VPD
y_CSP1_Season = N0_CSP1_Season
y_CSP2_Season = N0_CSP2_Season
y_CSP3_Season = N0_CSP3_Season
x_CSP1_Season = CSP1_VPD_Season
x_CSP2_Season = CSP2_VPD_Season
x_CSP3_Season = CSP3_VPD_Season
y_CSP1_meso = N0_CSP1
y_CSP2_meso = N0_CSP2
y_CSP3_meso = N0_CSP3
x_CSP1_meso = CSP1_VPD_meso
x_CSP2_meso = CSP2_VPD_meso
x_CSP3_meso = CSP3_VPD_meso

# Remove or replace NaN and infinite values
x_CSP1 = x_CSP1.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP2 = x_CSP2.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP3 = x_CSP3.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP1 = y_CSP1.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP2 = y_CSP2.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP3 = y_CSP3.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP1_Season = x_CSP1_Season.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP2_Season = x_CSP2_Season.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP3_Season = x_CSP3_Season.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP1_Season = y_CSP1_Season.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP2_Season = y_CSP2_Season.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP3_Season = y_CSP3_Season.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP1_meso = x_CSP1_meso.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP2_meso = x_CSP2_meso.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP3_meso = x_CSP3_meso.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP1_meso = y_CSP1_meso.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP2_meso = y_CSP2_meso.replace([np.inf, -np.inf], np.nan).dropna()
y_CSP3_meso = y_CSP3_meso.replace([np.inf, -np.inf], np.nan).dropna()

# Create a DataFrame with x_CSP1 and y_CSP1
df_CSP1 = pd.DataFrame({'x_CSP1': x_CSP1, 'y_CSP1': y_CSP1})
df_CSP1 = df_CSP1.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP1_clean = df_CSP1['x_CSP1']
y_CSP1_clean = df_CSP1['y_CSP1']

# Create a DataFrame with x_CSP2 and y_CSP2
df_CSP2 = pd.DataFrame({'x_CSP2': x_CSP2, 'y_CSP2': y_CSP2})
df_CSP2 = df_CSP2.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP2_clean = df_CSP2['x_CSP2']
y_CSP2_clean = df_CSP2['y_CSP2']

# Create a DataFrame with x_CSP3 and y_CSP3
df_CSP3 = pd.DataFrame({'x_CSP3': x_CSP3, 'y_CSP3': y_CSP3})
df_CSP3 = df_CSP3.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP3_clean = df_CSP3['x_CSP3']
y_CSP3_clean = df_CSP3['y_CSP3']

# Create a DataFrame with x_CSP1_Season and y_CSP1_Season
df_CSP1_Season = pd.DataFrame({'x_CSP1_Season': x_CSP1_Season, 'y_CSP1_Season': y_CSP1_Season})
df_CSP1_Season = df_CSP1_Season.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP1_clean_Season = df_CSP1_Season['x_CSP1_Season']
y_CSP1_clean_Season = df_CSP1_Season['y_CSP1_Season']

# Create a DataFrame with x_CSP2_Season and y_CSP2_Season
df_CSP2_Season = pd.DataFrame({'x_CSP2_Season': x_CSP2_Season, 'y_CSP2_Season': y_CSP2_Season})
df_CSP2_Season = df_CSP2_Season.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP2_clean_Season = df_CSP2_Season['x_CSP2_Season']
y_CSP2_clean_Season = df_CSP2_Season['y_CSP2_Season']

# Create a DataFrame with x_CSP3_Season and y_CSP3_Season
df_CSP3_Season = pd.DataFrame({'x_CSP3_Season': x_CSP3_Season, 'y_CSP3_Season': y_CSP3_Season})
df_CSP3_Season = df_CSP3_Season.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP3_clean_Season = df_CSP3_Season['x_CSP3_Season']
y_CSP3_clean_Season = df_CSP3_Season['y_CSP3_Season']

# Create a DataFrame with x_CSP1_meso and y_CSP1_meso
df_CSP1_meso = pd.DataFrame({'x_CSP1_meso': x_CSP1_meso, 'y_CSP1_meso': y_CSP1_meso})
df_CSP1_meso = df_CSP1_meso.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP1_clean_meso = df_CSP1_meso['x_CSP1_meso']
y_CSP1_clean_meso = df_CSP1_meso['y_CSP1_meso']

# Create a DataFrame with x_CSP2_meso and y_CSP2_meso
df_CSP2_meso = pd.DataFrame({'x_CSP2_meso': x_CSP2_meso, 'y_CSP2_meso': y_CSP2_meso})
df_CSP2_meso = df_CSP2_meso.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP2_clean_meso = df_CSP2_meso['x_CSP2_meso']
y_CSP2_clean_meso = df_CSP2_meso['y_CSP2_meso']

# Create a DataFrame with x_CSP3_meso and y_CSP3_meso
df_CSP3_meso = pd.DataFrame({'x_CSP3_meso': x_CSP3_meso, 'y_CSP3_meso': y_CSP3_meso})
df_CSP3_meso = df_CSP3_meso.replace([np.inf, -np.inf], np.nan).dropna()
x_CSP3_clean_meso = df_CSP3_meso['x_CSP3_meso']
y_CSP3_clean_meso = df_CSP3_meso['y_CSP3_meso']


plt.scatter(x_CSP1_clean, y_CSP1_clean, s=1, c='red', label='CSP1')
m, b = np.polyfit(x_CSP1_clean, y_CSP1_clean, 1)
plt.plot(x_CSP1_clean, m*x_CSP1_clean + b, color='red')
plt.scatter(x_CSP2_clean, y_CSP2_clean, s=1, c='blue', label='CSP2')
m, b = np.polyfit(x_CSP2_clean, y_CSP2_clean, 1)
plt.plot(x_CSP2_clean, m*x_CSP2_clean + b, color='blue')
plt.scatter(x_CSP3_clean, y_CSP3_clean, s=1, c='green', label='CSP3')
m, b = np.polyfit(x_CSP3_clean, y_CSP3_clean, 1)
plt.plot(x_CSP3_clean, m*x_CSP3_clean + b, color='green')
legend = plt.legend(loc='upper right')
for handle in legend.legendHandles:
    handle._sizes = [30]
plt.xlabel('VPD (kPa)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'VPD vs N$_{0}$')

#Least square regression
A_CSP1_19 = np.vstack([x_CSP1_clean, np.ones(len(x_CSP1_clean))]).T
y_CSP1_19 = np.array(y_CSP1_clean)  # convert y_CSP1_19 to a numpy array
y_CSP1_19 = y_CSP1_19[:, np.newaxis]
alpha_CSP1_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_19.T,A_CSP1_19)),A_CSP1_19.T)),y_CSP1_19)
m_CSP1_19 = alpha_CSP1_19[0]
n_CSP1_19 = alpha_CSP1_19[1]
mm_CSP1_19 = float(m_CSP1_19)
mmm_CSP1_19 = [float(m_CSP1_19)]
nn_CSP1_19 = float(n_CSP1_19)
r_mm_CSP1_19 = round(mm_CSP1_19,2)
r_nn_CSP1_19 = round(nn_CSP1_19,2)

#Calculate r squared
yhat_CSP1_19 = m_CSP1_19*x_CSP1_clean + n_CSP1_19
ybar_CSP1_19 = np.sum(y_CSP1_19)/len(y_CSP1_19)
ssreg_CSP1_19 = np.sum((yhat_CSP1_19-ybar_CSP1_19)**2)
sstot_CSP1_19 = np.sum((y_CSP1_19 - ybar_CSP1_19)**2)
r_squared_CSP1_19 = ssreg_CSP1_19 / sstot_CSP1_19
r2_CSP1_19 = round(r_squared_CSP1_19,2)


A_CSP2_19 = np.vstack([x_CSP2_clean, np.ones(len(x_CSP2_clean))]).T
y_CSP2_19 = np.array(y_CSP2_clean)  # convert y_CSP2_19 to a numpy array
y_CSP2_19 = y_CSP2_19[:, np.newaxis]
alpha_CSP2_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_19.T,A_CSP2_19)),A_CSP2_19.T)),y_CSP2_19)
m_CSP2_19 = alpha_CSP2_19[0]
n_CSP2_19 = alpha_CSP2_19[1]
mm_CSP2_19 = float(m_CSP2_19)
mmm_CSP2_19 = [float(m_CSP2_19)]
nn_CSP2_19 = float(n_CSP2_19)
r_mm_CSP2_19 = round(mm_CSP2_19,2)
r_nn_CSP2_19 = round(nn_CSP2_19,2)

#Calculate r squared
yhat_CSP2_19 = m_CSP2_19*x_CSP2_clean + n_CSP2_19
ybar_CSP2_19 = np.sum(y_CSP2_19)/len(y_CSP2_19)
ssreg_CSP2_19 = np.sum((yhat_CSP2_19-ybar_CSP2_19)**2)
sstot_CSP2_19 = np.sum((y_CSP2_19 - ybar_CSP2_19)**2)
r_squared_CSP2_19 = ssreg_CSP2_19 / sstot_CSP2_19
r2_CSP2_19 = round(r_squared_CSP2_19,2)

A_CSP3_19 = np.vstack([x_CSP3_clean, np.ones(len(x_CSP3_clean))]).T
y_CSP3_19 = np.array(y_CSP3_clean)  # convert y_CSP3_19 to a numpy array
y_CSP3_19 = y_CSP3_19[:, np.newaxis]
alpha_CSP3_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_19.T,A_CSP3_19)),A_CSP3_19.T)),y_CSP3_19)
m_CSP3_19 = alpha_CSP3_19[0]
n_CSP3_19 = alpha_CSP3_19[1]
mm_CSP3_19 = float(m_CSP3_19)
mmm_CSP3_19 = [float(m_CSP3_19)]
nn_CSP3_19 = float(n_CSP3_19)
r_mm_CSP3_19 = round(mm_CSP3_19,2)
r_nn_CSP3_19 = round(nn_CSP3_19,2)

#Calculate r squared
yhat_CSP3_19 = m_CSP3_19*x_CSP3_clean + n_CSP3_19
ybar_CSP3_19 = np.sum(y_CSP3_19)/len(y_CSP3_19)
ssreg_CSP3_19 = np.sum((yhat_CSP3_19-ybar_CSP3_19)**2)
sstot_CSP3_19 = np.sum((y_CSP3_19 - ybar_CSP3_19)**2)
r_squared_CSP3_19 = ssreg_CSP3_19 / sstot_CSP3_19
r2_CSP3_19 = round(r_squared_CSP3_19,2)

col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['CSP1', 'CSP2', 'CSP3']
table_vals = [[r_mm_CSP1_19,r_nn_CSP1_19, r2_CSP1_19], [r_mm_CSP2_19,r_nn_CSP2_19, r2_CSP2_19], [r_mm_CSP3_19,r_nn_CSP3_19, r2_CSP3_19]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='center right')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
#This first plot uses is not as good as the following two
plt.savefig('YOUR PATH TO SAVE/VPD.png', dpi=300, bbox_inches='tight')
plt.show()


#plot in season
plt.scatter(x_CSP1_clean_Season, y_CSP1_clean_Season, s=1, c='red', label='CSP1')
m, b = np.polyfit(x_CSP1_clean_Season, y_CSP1_clean_Season, 1)
plt.plot(x_CSP1_clean_Season, m*x_CSP1_clean_Season + b, color='red')
plt.scatter(x_CSP2_clean_Season, y_CSP2_clean_Season, s=1, c='blue', label='CSP2')
m, b = np.polyfit(x_CSP2_clean_Season, y_CSP2_clean_Season, 1)
plt.plot(x_CSP2_clean_Season, m*x_CSP2_clean_Season + b, color='blue')
plt.scatter(x_CSP3_clean_Season, y_CSP3_clean_Season, s=1, c='green', label='CSP3')
m, b = np.polyfit(x_CSP3_clean_Season, y_CSP3_clean_Season, 1)
plt.plot(x_CSP3_clean_Season, m*x_CSP3_clean_Season + b, color='green')
legend = plt.legend(loc='upper right')
for handle in legend.legendHandles:
    handle._sizes = [30]
plt.xlabel('VPD (kPa)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title('VPD vs N$_{0}$')

#Least square regression
A_CSP1_Season_19 = np.vstack([x_CSP1_clean_Season, np.ones(len(x_CSP1_clean_Season))]).T
y_CSP1_Season_19 = np.array(y_CSP1_clean_Season) 
y_CSP1_Season_19 = y_CSP1_Season_19[:, np.newaxis]
alpha_CSP1_Season_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_Season_19.T,A_CSP1_Season_19)),A_CSP1_Season_19.T)),y_CSP1_Season_19)
m_CSP1_Season_19 = alpha_CSP1_Season_19[0]
n_CSP1_Season_19 = alpha_CSP1_Season_19[1]
mm_CSP1_Season_19 = float(m_CSP1_Season_19)
mmm_CSP1_Season_19 = [float(m_CSP1_Season_19)]
nn_CSP1_Season_19 = float(n_CSP1_Season_19)
r_mm_CSP1_Season_19 = round(mm_CSP1_Season_19,2)
r_nn_CSP1_Season_19 = round(nn_CSP1_Season_19,2)

A_CSP2_Season_19 = np.vstack([x_CSP2_clean_Season, np.ones(len(x_CSP2_clean_Season))]).T
y_CSP2_Season_19 = np.array(y_CSP2_clean_Season)  
y_CSP2_Season_19 = y_CSP2_Season_19[:, np.newaxis]
alpha_CSP2_Season_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_Season_19.T,A_CSP2_Season_19)),A_CSP2_Season_19.T)),y_CSP2_Season_19)
m_CSP2_Season_19 = alpha_CSP2_Season_19[0]
n_CSP2_Season_19 = alpha_CSP2_Season_19[1]
mm_CSP2_Season_19 = float(m_CSP2_Season_19)
mmm_CSP2_Season_19 = [float(m_CSP2_Season_19)]
nn_CSP2_Season_19 = float(n_CSP2_Season_19)
r_mm_CSP2_Season_19 = round(mm_CSP2_Season_19,2)
r_nn_CSP2_Season_19 = round(nn_CSP2_Season_19,2)

A_CSP3_Season_19 = np.vstack([x_CSP3_clean_Season, np.ones(len(x_CSP3_clean_Season))]).T
y_CSP3_Season_19 = np.array(y_CSP3_clean_Season)  
y_CSP3_Season_19 = y_CSP3_Season_19[:, np.newaxis]
alpha_CSP3_Season_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_Season_19.T,A_CSP3_Season_19)),A_CSP3_Season_19.T)),y_CSP3_Season_19)
m_CSP3_Season_19 = alpha_CSP3_Season_19[0]
n_CSP3_Season_19 = alpha_CSP3_Season_19[1]
mm_CSP3_Season_19 = float(m_CSP3_Season_19)
mmm_CSP3_Season_19 = [float(m_CSP3_Season_19)]
nn_CSP3_Season_19 = float(n_CSP3_Season_19)
r_mm_CSP3_Season_19 = round(mm_CSP3_Season_19,2)
r_nn_CSP3_Season_19 = round(nn_CSP3_Season_19,2)

#Calculate r squared
yhat_CSP1_Season_19 = m_CSP1_Season_19*x_CSP1_clean_Season + n_CSP1_Season_19
ybar_CSP1_Season_19 = np.sum(y_CSP1_Season_19)/len(y_CSP1_Season_19)
ssreg_CSP1_Season_19 = np.sum((yhat_CSP1_Season_19-ybar_CSP1_Season_19)**2)
sstot_CSP1_Season_19 = np.sum((y_CSP1_Season_19 - ybar_CSP1_Season_19)**2)
r_squared_CSP1_Season_19 = ssreg_CSP1_Season_19 / sstot_CSP1_Season_19
r2_CSP1_Season_19 = round(r_squared_CSP1_Season_19,2)

#Calculate r squared
yhat_CSP2_Season_19 = m_CSP2_Season_19*x_CSP2_clean_Season + n_CSP2_Season_19
ybar_CSP2_Season_19 = np.sum(y_CSP2_Season_19)/len(y_CSP2_Season_19)
ssreg_CSP2_Season_19 = np.sum((yhat_CSP2_Season_19-ybar_CSP2_Season_19)**2)
sstot_CSP2_Season_19 = np.sum((y_CSP2_Season_19 - ybar_CSP2_Season_19)**2)
r_squared_CSP2_Season_19 = ssreg_CSP2_Season_19 / sstot_CSP2_Season_19
r2_CSP2_Season_19 = round(r_squared_CSP2_Season_19,2)

#Calculate r squared
yhat_CSP3_Season_19 = m_CSP3_Season_19*x_CSP3_clean_Season + n_CSP3_Season_19
ybar_CSP3_Season_19 = np.sum(y_CSP3_Season_19)/len(y_CSP3_Season_19)
ssreg_CSP3_Season_19 = np.sum((yhat_CSP3_Season_19-ybar_CSP3_Season_19)**2)
sstot_CSP3_Season_19 = np.sum((y_CSP3_Season_19 - ybar_CSP3_Season_19)**2)
r_squared_CSP3_Season_19 = ssreg_CSP3_Season_19 / sstot_CSP3_Season_19
r2_CSP3_Season_19 = round(r_squared_CSP3_Season_19,2)


col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['CSP1', 'CSP2', 'CSP3']
table_vals = [[r_mm_CSP1_Season_19,r_nn_CSP1_Season_19, r2_CSP1_Season_19], [r_mm_CSP2_Season_19,r_nn_CSP2_Season_19, r2_CSP2_Season_19], [r_mm_CSP3_Season_19,r_nn_CSP3_Season_19, r2_CSP3_Season_19]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='center right')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/VPD_Season.png', dpi=300, bbox_inches='tight')
plt.show()

#plot meso
plt.scatter(x_CSP1_clean_meso, y_CSP1_clean_meso, s=1, c='red', label='CSP1')
m, b = np.polyfit(x_CSP1_clean_meso, y_CSP1_clean_meso, 1)
plt.plot(x_CSP1_clean_meso, m*x_CSP1_clean_meso + b, color='red')
plt.scatter(x_CSP2_clean_meso, y_CSP2_clean_meso, s=1, c='blue', label='CSP2')
m, b = np.polyfit(x_CSP2_clean_meso, y_CSP2_clean_meso, 1)
plt.plot(x_CSP2_clean_meso, m*x_CSP2_clean_meso + b, color='blue')
plt.scatter(x_CSP3_clean_meso, y_CSP3_clean_meso, s=1, c='green', label='CSP3')
m, b = np.polyfit(x_CSP3_clean_meso, y_CSP3_clean_meso, 1)
plt.plot(x_CSP3_clean_meso, m*x_CSP3_clean_meso + b, color='green')
legend = plt.legend(loc='upper right')
for handle in legend.legendHandles:
    handle._sizes = [30]
plt.xlabel('VPD (kPa)')
plt.ylabel(r'N$_{0}$ (cph)')
plt.title(r'VPD vs N$_{0}$')

#Least square regression
A_CSP1_meso_19 = np.vstack([x_CSP1_clean_meso, np.ones(len(x_CSP1_clean_meso))]).T
y_CSP1_meso_19 = np.array(y_CSP1_clean_meso)  
y_CSP1_meso_19 = y_CSP1_meso_19[:, np.newaxis]
alpha_CSP1_meso_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP1_meso_19.T,A_CSP1_meso_19)),A_CSP1_meso_19.T)),y_CSP1_meso_19)
m_CSP1_meso_19 = alpha_CSP1_meso_19[0]
n_CSP1_meso_19 = alpha_CSP1_meso_19[1]
mm_CSP1_meso_19 = float(m_CSP1_meso_19)
mmm_CSP1_meso_19 = [float(m_CSP1_meso_19)]
nn_CSP1_meso_19 = float(n_CSP1_meso_19)
r_mm_CSP1_meso_19 = round(mm_CSP1_meso_19,2)
r_nn_CSP1_meso_19 = round(nn_CSP1_meso_19,2)

A_CSP2_meso_19 = np.vstack([x_CSP2_clean_meso, np.ones(len(x_CSP2_clean_meso))]).T
y_CSP2_meso_19 = np.array(y_CSP2_clean_meso)  
y_CSP2_meso_19 = y_CSP2_meso_19[:, np.newaxis]
alpha_CSP2_meso_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP2_meso_19.T,A_CSP2_meso_19)),A_CSP2_meso_19.T)),y_CSP2_meso_19)
m_CSP2_meso_19 = alpha_CSP2_meso_19[0]
n_CSP2_meso_19 = alpha_CSP2_meso_19[1]
mm_CSP2_meso_19 = float(m_CSP2_meso_19)
mmm_CSP2_meso_19 = [float(m_CSP2_meso_19)]
nn_CSP2_meso_19 = float(n_CSP2_meso_19)
r_mm_CSP2_meso_19 = round(mm_CSP2_meso_19,2)
r_nn_CSP2_meso_19 = round(nn_CSP2_meso_19,2)

A_CSP3_meso_19 = np.vstack([x_CSP3_clean_meso, np.ones(len(x_CSP3_clean_meso))]).T
y_CSP3_meso_19 = np.array(y_CSP3_clean_meso)  
y_CSP3_meso_19 = y_CSP3_meso_19[:, np.newaxis]
alpha_CSP3_meso_19 = np.dot((np.dot(np.linalg.inv(np.dot(A_CSP3_meso_19.T,A_CSP3_meso_19)),A_CSP3_meso_19.T)),y_CSP3_meso_19)
m_CSP3_meso_19 = alpha_CSP3_meso_19[0]
n_CSP3_meso_19 = alpha_CSP3_meso_19[1]
mm_CSP3_meso_19 = float(m_CSP3_meso_19)
mmm_CSP3_meso_19 = [float(m_CSP3_meso_19)]
nn_CSP3_meso_19 = float(n_CSP3_meso_19)
r_mm_CSP3_meso_19 = round(mm_CSP3_meso_19,2)
r_nn_CSP3_meso_19 = round(nn_CSP3_meso_19,2)

#Calculate r squared
yhat_CSP1_meso_19 = m_CSP1_meso_19*x_CSP1_clean_meso + n_CSP1_meso_19
ybar_CSP1_meso_19 = np.sum(y_CSP1_meso_19)/len(y_CSP1_meso_19)
ssreg_CSP1_meso_19 = np.sum((yhat_CSP1_meso_19-ybar_CSP1_meso_19)**2)
sstot_CSP1_meso_19 = np.sum((y_CSP1_meso_19 - ybar_CSP1_meso_19)**2)
r_squared_CSP1_meso_19 = ssreg_CSP1_meso_19 / sstot_CSP1_meso_19
r2_CSP1_meso_19 = round(r_squared_CSP1_meso_19,2)

#Calculate r squared
yhat_CSP2_meso_19 = m_CSP2_meso_19*x_CSP2_clean_meso + n_CSP2_meso_19
ybar_CSP2_meso_19 = np.sum(y_CSP2_meso_19)/len(y_CSP2_meso_19)
ssreg_CSP2_meso_19 = np.sum((yhat_CSP2_meso_19-ybar_CSP2_meso_19)**2)
sstot_CSP2_meso_19 = np.sum((y_CSP2_meso_19 - ybar_CSP2_meso_19)**2)
r_squared_CSP2_meso_19 = ssreg_CSP2_meso_19 / sstot_CSP2_meso_19
r2_CSP2_meso_19 = round(r_squared_CSP2_meso_19,2)

#Calculate r squared
yhat_CSP3_meso_19 = m_CSP3_meso_19*x_CSP3_clean_meso + n_CSP3_meso_19
ybar_CSP3_meso_19 = np.sum(y_CSP3_meso_19)/len(y_CSP3_meso_19)
ssreg_CSP3_meso_19 = np.sum((yhat_CSP3_meso_19-ybar_CSP3_meso_19)**2)
sstot_CSP3_meso_19 = np.sum((y_CSP3_meso_19 - ybar_CSP3_meso_19)**2)
r_squared_CSP3_meso_19 = ssreg_CSP3_meso_19 / sstot_CSP3_meso_19
r2_CSP3_meso_19 = round(r_squared_CSP3_meso_19,2)

col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['CSP1', 'CSP2', 'CSP3']
table_vals = [[r_mm_CSP1_meso_19,r_nn_CSP1_meso_19, r2_CSP1_meso_19], [r_mm_CSP2_meso_19,r_nn_CSP2_meso_19, r2_CSP2_meso_19], [r_mm_CSP3_meso_19,r_nn_CSP3_meso_19, r2_CSP3_meso_19]]
table = plt.table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='center right')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

plt.savefig('YOUR PATH TO SAVE/VPD_mesonet.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with two subplots side by side

axs[0].scatter(x_CSP1_clean_Season, y_CSP1_clean_Season, s=1, c='red', label='CSP1')
m, b = np.polyfit(x_CSP1_clean_Season, y_CSP1_clean_Season, 1)
axs[0].plot(x_CSP1_clean_Season, m*x_CSP1_clean_Season + b, color='red')
axs[0].scatter(x_CSP2_clean_Season, y_CSP2_clean_Season, s=1, c='blue', label='CSP2')
m, b = np.polyfit(x_CSP2_clean_Season, y_CSP2_clean_Season, 1)
axs[0].plot(x_CSP2_clean_Season, m*x_CSP2_clean_Season + b, color='blue')
axs[0].scatter(x_CSP3_clean_Season, y_CSP3_clean_Season, s=1, c='green', label='CSP3')
m, b = np.polyfit(x_CSP3_clean_Season, y_CSP3_clean_Season, 1)
axs[0].plot(x_CSP3_clean_Season, m*x_CSP3_clean_Season + b, color='green')
legend = axs[0].legend(loc='upper right')
for handle in legend.legendHandles:
    handle._sizes = [30]
axs[0].set_xlabel('VPD (kPa)')
axs[0].set_ylabel(r'N$_{0}$ (cph)')
axs[0].set_title('VPD vs N$_{0}$ Full Season')
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['CSP1', 'CSP2', 'CSP3']
table_vals = [[r_mm_CSP1_Season_19,r_nn_CSP1_Season_19, r2_CSP1_Season_19], [r_mm_CSP2_Season_19,r_nn_CSP2_Season_19, r2_CSP2_Season_19], [r_mm_CSP3_Season_19,r_nn_CSP3_Season_19, r2_CSP3_Season_19]]
table = axs[0].table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='center right')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)

#plot meso
axs[1].scatter(x_CSP1_clean_meso, y_CSP1_clean_meso, s=1, c='red', label='CSP1')
m, b = np.polyfit(x_CSP1_clean_meso, y_CSP1_clean_meso, 1)
axs[1].plot(x_CSP1_clean_meso, m*x_CSP1_clean_meso + b, color='red')
axs[1].scatter(x_CSP2_clean_meso, y_CSP2_clean_meso, s=1, c='blue', label='CSP2')
m, b = np.polyfit(x_CSP2_clean_meso, y_CSP2_clean_meso, 1)
axs[1].plot(x_CSP2_clean_meso, m*x_CSP2_clean_meso + b, color='blue')
axs[1].scatter(x_CSP3_clean_meso, y_CSP3_clean_meso, s=1, c='green', label='CSP3')
m, b = np.polyfit(x_CSP3_clean_meso, y_CSP3_clean_meso, 1)
axs[1].plot(x_CSP3_clean_meso, m*x_CSP3_clean_meso + b, color='green')
legend = axs[1].legend(loc='upper right')
for handle in legend.legendHandles:
    handle._sizes = [30]
axs[1].set_xlabel('VPD (kPa)')
axs[1].set_ylabel(r'N$_{0}$ (cph)')
axs[1].set_title(r'VPD vs N$_{0}$ July to September')
col_labels = ['Slope', 'Intercept', 'r\u00b2']
row_labels = ['CSP1', 'CSP2', 'CSP3']
table_vals = [[r_mm_CSP1_meso_19,r_nn_CSP1_meso_19, r2_CSP1_meso_19], [r_mm_CSP2_meso_19,r_nn_CSP2_meso_19, r2_CSP2_meso_19], [r_mm_CSP3_meso_19,r_nn_CSP3_meso_19, r2_CSP3_meso_19]]
table = axs[1].table(cellText=table_vals,
                       colWidths=[0.1] * 3,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='center right')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.1)
for i, label in enumerate(['a)', 'b)']):
    axs[i].text(-0.1, 1.1, label, transform=axs[i].transAxes, fontsize=16, va='top', ha='right')

#This combines the plots above into one plot
plt.savefig('YOUR PATH TO SAVE/vpd.png', dpi=300, bbox_inches='tight')
plt.show()