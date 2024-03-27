import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given

#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP1_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

####10 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_1_1_1']/1.42
SWC_2_1_1 = GWC['SWC_2_1_1']/1.42
SWC_3_1_1 = GWC['SWC_3_1_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df1 = SWC_1_1_1.resample('D').mean() 
df2 = SWC_2_1_1.resample('D').mean()
df3 = SWC_3_1_1.resample('D').mean()

df_SWC_1_1_1 = pd.DataFrame(df1)
df_SWC_2_1_1 = pd.DataFrame(df2)
df_SWC_3_1_1 = pd.DataFrame(df3)

df_SWC_1_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_1_CSP1.csv')
df_SWC_2_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_2_CSP1.csv')
df_SWC_3_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_3_CSP1.csv')

Fixed_Data_1_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_1_CSP1.csv')
Fixed_Data_2_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_2_CSP1.csv')
Fixed_Data_3_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_3_CSP1.csv')

CSP1_1 = Fixed_Data_1_1_1['SWC_1_1_1']/100
CSP1_2 = Fixed_Data_2_1_1['SWC_2_1_1']/100
CSP1_3 = Fixed_Data_3_1_1['SWC_3_1_1']/100

Date = Fixed_Data_1_1_1['DATE']
Date5 = Fixed_Data_2_1_1['DATE']
Date6 = Fixed_Data_3_1_1['DATE']

x = Date
y = CSP1_1
a = CSP1_2
b = CSP1_3

####25 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_2_1 = GWC['SWC_1_2_1']/1.42
SWC_2_2_1 = GWC['SWC_2_2_1']/1.42
SWC_3_2_1 = GWC['SWC_3_2_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df4 = SWC_1_2_1.resample('D').mean() 
df5 = SWC_2_2_1.resample('D').mean()
df6 = SWC_3_2_1.resample('D').mean()

df_SWC_1_2_1 = pd.DataFrame(df4)
df_SWC_2_2_1 = pd.DataFrame(df5)
df_SWC_3_2_1 = pd.DataFrame(df6)

df_SWC_1_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_1_CSP1.csv')
df_SWC_2_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_2_CSP1.csv')
df_SWC_3_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_3_CSP1.csv')

Fixed_Data_1_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_1_CSP1.csv')
Fixed_Data_2_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_2_CSP1.csv')
Fixed_Data_3_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_3_CSP1.csv')

CSP1_4 = Fixed_Data_1_2_1['SWC_1_2_1']/100
CSP1_5 = Fixed_Data_2_2_1['SWC_2_2_1']/100
CSP1_6 = Fixed_Data_3_2_1['SWC_3_2_1']/100

Date1 = Fixed_Data_3_2_1['DATE']
Date7 = Fixed_Data_2_2_1['DATE']
Date8 = Fixed_Data_3_2_1['DATE']

m = Date1
n = CSP1_4
p = CSP1_5
q = CSP1_6


####50 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_3_1 = GWC['SWC_1_3_1']/1.42
SWC_2_3_1 = GWC['SWC_2_3_1']/1.42
SWC_3_3_1 = GWC['SWC_3_3_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df7 = SWC_1_3_1.resample('D').mean() 
df8 = SWC_2_3_1.resample('D').mean()
df9 = SWC_3_3_1.resample('D').mean()

df_SWC_1_3_1 = pd.DataFrame(df7)
df_SWC_2_3_1 = pd.DataFrame(df8)
df_SWC_3_3_1 = pd.DataFrame(df9)

df_SWC_1_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_1_CSP1.csv')
df_SWC_2_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_2_CSP1.csv')
df_SWC_3_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_3_CSP1.csv')

Fixed_Data_1_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_1_CSP1.csv')
Fixed_Data_2_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_2_CSP1.csv')
Fixed_Data_3_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_3_CSP1.csv')

CSP1_7 = Fixed_Data_1_3_1['SWC_1_3_1']/100
CSP1_8 = Fixed_Data_2_3_1['SWC_2_3_1']/100
CSP1_9 = Fixed_Data_3_3_1['SWC_3_3_1']/100

Date2 = Fixed_Data_3_3_1['DATE']

r = Date2
s = CSP1_7
t = CSP1_8
v = CSP1_9


####100 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_4_1 = GWC['SWC_1_4_1']/1.42
SWC_2_4_1 = GWC['SWC_2_4_1']/1.42
SWC_3_4_1 = GWC['SWC_3_4_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df10 = SWC_1_4_1.resample('D').mean() 
df11 = SWC_2_4_1.resample('D').mean()
df12 = SWC_3_4_1.resample('D').mean()

df_SWC_1_4_1 = pd.DataFrame(df10)
df_SWC_2_4_1 = pd.DataFrame(df11)
df_SWC_3_4_1 = pd.DataFrame(df12)

df_SWC_1_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_1_CSP1.csv')
df_SWC_2_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_2_CSP1.csv')
df_SWC_3_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_3_CSP1.csv')

Fixed_Data_1_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_1_CSP1.csv')
Fixed_Data_2_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_2_CSP1.csv')
Fixed_Data_3_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_3_CSP1.csv')

CSP1_10 = Fixed_Data_1_4_1['SWC_1_4_1']/100
CSP1_11 = Fixed_Data_2_4_1['SWC_2_4_1']/100
CSP1_12 = Fixed_Data_3_4_1['SWC_3_4_1']/100

Date3 = Fixed_Data_3_4_1['DATE']

g = Date3
h = CSP1_10
j = CSP1_11
k = CSP1_12


months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, axs = plt.subplots(2,2,figsize=(15,10),
                        constrained_layout = True)
axs[0,0].plot(x,y, color='darkmagenta', label='IMZ 4')
axs[0,0].plot(x,a, color = 'limegreen', label='IMZ 5')
axs[0,0].plot(x,b, color = 'deeppink', label='IMZ 6')
axs[0,0].set_title('10cm Depth')
axs[0,0].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[0,0].set_ylim(0.07,0.385)
axs[0,0].tick_params(axis = 'x', rotation = 25)
axs[0,0].set_xticks(np.linspace(1,1700,11))
axs[0,0].legend(loc='upper right')
axs[0,0].set_xticklabels([])
axs[0,1].plot(m,n, color = 'darkmagenta', label='IMZ 4')
axs[0,1].plot(m,p, color = 'limegreen', label='IMZ 5')
axs[0,1].plot(m,q, color = 'deeppink', label='IMZ 6')
axs[0,1].set_title('25cm Depth')
axs[0,1].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[0,1].set_xticks(np.linspace(1,1700,11))
axs[0,1].set_ylim(0.07,0.385)
axs[0,1].tick_params(axis = 'x', rotation = 25)
axs[0,1].legend(loc='upper right')
axs[0,1].set_xticklabels([])
axs[1,0].plot(r,s, color = 'darkmagenta', label='IMZ 4')
axs[1,0].plot(r,t, color = 'limegreen', label='IMZ 5')
axs[1,0].plot(r,v, color = 'deeppink', label='IMZ 6')
axs[1,0].set_title('50cm Depth')
axs[1,0].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[1,0].set_xlabel('Date')
axs[1,0].set_xticks(np.linspace(1,1700,11))
axs[1,0].set_ylim(0.07,0.385)
axs[1,0].tick_params(axis = 'x', rotation = 25)
axs[1,0].legend(loc='upper right')
axs[1,1].plot(g,h, color = 'darkmagenta', label='IMZ 4')
axs[1,1].plot(g,j, color = 'limegreen', label='IMZ 5')
axs[1,1].plot(g,k, color = 'deeppink', label='IMZ 6')
axs[1,1].set_title('100cm Depth')
axs[1,1].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[1,1].set_xlabel('Date')
axs[1,1].set_xticks(np.linspace(1,1700,11))
axs[1,1].set_ylim(0.07,0.385)
axs[1,1].tick_params(axis = 'x', rotation = 25)
axs[1,1].legend(loc='upper right')
fig.suptitle('TDR Measured Soil Water Content 2019 to 2023 - CSP1', fontsize = 20)
fig.savefig('YOUR PATH TO SAVE/TDR_SoilWaterContent_CSP1.png')
plt.show()




#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP2_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

####10 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df1 = SWC_1_1_1.resample('D').mean() 
df2 = SWC_2_1_1.resample('D').mean()
df3 = SWC_3_1_1.resample('D').mean()

df_SWC_1_1_1 = pd.DataFrame(df1)
df_SWC_2_1_1 = pd.DataFrame(df2)
df_SWC_3_1_1 = pd.DataFrame(df3)

df_SWC_1_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_1_CSP2.csv')
df_SWC_2_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_2_CSP2.csv')
df_SWC_3_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_3_CSP2.csv')

Fixed_Data_1_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_1_CSP2.csv')
Fixed_Data_2_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_2_CSP2.csv')
Fixed_Data_3_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_3_CSP2.csv')

CSP1_1 = Fixed_Data_1_1_1['SWC_PI_F_1_1_1']/100
CSP1_2 = Fixed_Data_2_1_1['SWC_PI_F_2_1_1']/100
CSP1_3 = Fixed_Data_3_1_1['SWC_PI_F_3_1_1']/100

Date = Fixed_Data_3_1_1['DATE']

x = Date
y = CSP1_1
a = CSP1_2
b = CSP1_3


####25 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df4 = SWC_1_2_1.resample('D').mean() 
df5 = SWC_2_2_1.resample('D').mean()
df6 = SWC_3_2_1.resample('D').mean()

df_SWC_1_2_1 = pd.DataFrame(df4)
df_SWC_2_2_1 = pd.DataFrame(df5)
df_SWC_3_2_1 = pd.DataFrame(df6)

df_SWC_1_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_1_CSP2.csv')
df_SWC_2_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_2_CSP2.csv')
df_SWC_3_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_3_CSP2.csv')

Fixed_Data_1_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_1_CSP2.csv')
Fixed_Data_2_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_2_CSP2.csv')
Fixed_Data_3_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_3_CSP2.csv')

CSP1_4 = Fixed_Data_1_2_1['SWC_PI_F_1_2_1']/100
CSP1_5 = Fixed_Data_2_2_1['SWC_PI_F_2_2_1']/100
CSP1_6 = Fixed_Data_3_2_1['SWC_PI_F_3_2_1']/100

Date1 = Fixed_Data_3_2_1['DATE']

m = Date1
n = CSP1_4
p = CSP1_5
q = CSP1_6


####50 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df7 = SWC_1_3_1.resample('D').mean() 
df8 = SWC_2_3_1.resample('D').mean()
df9 = SWC_3_3_1.resample('D').mean()

df_SWC_1_3_1 = pd.DataFrame(df7)
df_SWC_2_3_1 = pd.DataFrame(df8)
df_SWC_3_3_1 = pd.DataFrame(df9)

df_SWC_1_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_1_CSP2.csv')
df_SWC_2_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_2_CSP2.csv')
df_SWC_3_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_3_CSP2.csv')

Fixed_Data_1_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_1_CSP2.csv')
Fixed_Data_2_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_2_CSP2.csv')
Fixed_Data_3_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_3_CSP2.csv')

CSP1_7 = Fixed_Data_1_3_1['SWC_PI_F_1_3_1']/100
CSP1_8 = Fixed_Data_2_3_1['SWC_PI_F_2_3_1']/100
CSP1_9 = Fixed_Data_3_3_1['SWC_PI_F_3_3_1']/100

Date2 = Fixed_Data_3_3_1['DATE']

r = Date2
s = CSP1_7
t = CSP1_8
v = CSP1_9


####100 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df10 = SWC_1_4_1.resample('D').mean() 
df11 = SWC_2_4_1.resample('D').mean()
df12 = SWC_3_4_1.resample('D').mean()

df_SWC_1_4_1 = pd.DataFrame(df10)
df_SWC_2_4_1 = pd.DataFrame(df11)
df_SWC_3_4_1 = pd.DataFrame(df12)

df_SWC_1_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_1_CSP2.csv')
df_SWC_2_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_2_CSP2.csv')
df_SWC_3_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_3_CSP2.csv')

Fixed_Data_1_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_1_CSP2.csv')
Fixed_Data_2_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_2_CSP2.csv')
Fixed_Data_3_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_3_CSP2.csv')

CSP1_10 = Fixed_Data_1_4_1['SWC_PI_F_1_4_1']/100
CSP1_11 = Fixed_Data_2_4_1['SWC_PI_F_2_4_1']/100
CSP1_12 = Fixed_Data_3_4_1['SWC_PI_F_3_4_1']/100

Date3 = Fixed_Data_3_4_1['DATE']

g = Date3
h = CSP1_10
j = CSP1_11
k = CSP1_12


months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, axs = plt.subplots(2,2,figsize=(15,10),
                        constrained_layout = True)
axs[0,0].plot(x,y, color='darkmagenta', label='IMZ 2')
axs[0,0].plot(x,a, color = 'limegreen', label='IMZ 5')
axs[0,0].plot(x,b, color = 'deeppink', label='IMZ 6')
axs[0,0].set_title('10cm Depth')
axs[0,0].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[0,0].set_ylim(0.07,0.385)
axs[0,0].tick_params(axis = 'x', rotation = 25)
axs[0,0].set_xticks(np.linspace(1,1650,10))
axs[0,0].legend(loc='upper right')
axs[0,0].set_xticklabels([])
axs[0,1].plot(m,n, color = 'darkmagenta', label='IMZ 2')
axs[0,1].plot(m,p, color = 'limegreen', label='IMZ 5')
axs[0,1].plot(m,q, color = 'deeppink', label='IMZ 6')
axs[0,1].set_title('25cm Depth')
axs[0,1].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[0,1].set_xticks(np.linspace(1,1650,10))
axs[0,1].set_ylim(0.07,0.385)
axs[0,1].tick_params(axis = 'x', rotation = 25)
axs[0,1].legend(loc='upper right')
axs[0,1].set_xticklabels([])
axs[1,0].plot(r,s, color = 'darkmagenta', label='IMZ 2')
axs[1,0].plot(r,t, color = 'limegreen', label='IMZ 5')
axs[1,0].plot(r,v, color = 'deeppink', label='IMZ 6')
axs[1,0].set_title('50cm Depth')
axs[1,0].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[1,0].set_xlabel('Date')
axs[1,0].set_xticks(np.linspace(1,1650,10))
axs[1,0].set_ylim(0.07,0.385)
axs[1,0].tick_params(axis = 'x', rotation = 25)
axs[1,0].legend(loc='upper right')
axs[1,1].plot(g,h, color = 'darkmagenta', label='IMZ 2')
axs[1,1].plot(g,j, color = 'limegreen', label='IMZ 5')
axs[1,1].plot(g,k, color = 'deeppink', label='IMZ 6')
axs[1,1].set_title('100cm Depth')
axs[1,1].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
#axs[1,1].set_ylabel('SWC')
axs[1,1].set_xlabel('Date')
axs[1,1].set_xticks(np.linspace(1,1650,10))
axs[1,1].set_ylim(0.07,0.385)
axs[1,1].tick_params(axis = 'x', rotation = 25)
axs[1,1].legend(loc='upper right')
fig.suptitle('TDR Measured Soil Water Content 2019 to 2023 - CSP2', fontsize = 20)
fig.savefig('YOUR PATH TO SAVE/TDR_SoilWaterContent_CSP2.png')
plt.show()




#Date-time plots to check BWE and dates
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP3_AllDates.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])

####10 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_1_1 = GWC['SWC_PI_F_1_1_1']/1.42
SWC_2_1_1 = GWC['SWC_PI_F_2_1_1']/1.42
SWC_3_1_1 = GWC['SWC_PI_F_3_1_1']/1.42
SWC_4_1_1 = GWC['SWC_PI_F_4_1_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df1 = SWC_1_1_1.resample('D').mean() 
df2 = SWC_2_1_1.resample('D').mean()
df3 = SWC_3_1_1.resample('D').mean()
df4 = SWC_4_1_1.resample('D').mean()

df_SWC_1_1_1 = pd.DataFrame(df1)
df_SWC_2_1_1 = pd.DataFrame(df2)
df_SWC_3_1_1 = pd.DataFrame(df3)
df_SWC_4_1_1 = pd.DataFrame(df4)

df_SWC_1_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_1_CSP3.csv')
df_SWC_2_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_2_CSP3.csv')
df_SWC_3_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_3_CSP3.csv')
df_SWC_4_1_1.to_csv(r'YOUR PATH TO MAKE/TDR_10cm_4_CSP3.csv')

Fixed_Data_1_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_1_CSP3.csv')
Fixed_Data_2_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_2_CSP3.csv')
Fixed_Data_3_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_3_CSP3.csv')
Fixed_Data_4_1_1 = pd.read_csv('YOUR PATH/TDR_10cm_4_CSP3.csv')

CSP3_1 = Fixed_Data_1_1_1['SWC_PI_F_1_1_1']/100
CSP3_2 = Fixed_Data_2_1_1['SWC_PI_F_2_1_1']/100
CSP3_3 = Fixed_Data_3_1_1['SWC_PI_F_3_1_1']/100
CSP3_4 = Fixed_Data_4_1_1['SWC_PI_F_4_1_1']/100

Date = Fixed_Data_3_1_1['DATE']

x = Date
y = CSP3_1
a = CSP3_2
b = CSP3_3
c = CSP3_4


####25 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_2_1 = GWC['SWC_PI_F_1_2_1']/1.42
SWC_2_2_1 = GWC['SWC_PI_F_2_2_1']/1.42
SWC_3_2_1 = GWC['SWC_PI_F_3_2_1']/1.42
SWC_4_2_1 = GWC['SWC_PI_F_4_2_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df5 = SWC_1_2_1.resample('D').mean() 
df6 = SWC_2_2_1.resample('D').mean()
df7 = SWC_3_2_1.resample('D').mean()
df8 = SWC_4_2_1.resample('D').mean()

df_SWC_1_2_1 = pd.DataFrame(df5)
df_SWC_2_2_1 = pd.DataFrame(df6)
df_SWC_3_2_1 = pd.DataFrame(df7)
df_SWC_4_2_1 = pd.DataFrame(df8)

df_SWC_1_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_1_CSP3.csv')
df_SWC_2_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_2_CSP3.csv')
df_SWC_3_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_3_CSP3.csv')
df_SWC_4_2_1.to_csv(r'YOUR PATH TO MAKE/TDR_25cm_4_CSP3.csv')

Fixed_Data_1_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_1_CSP3.csv')
Fixed_Data_2_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_2_CSP3.csv')
Fixed_Data_3_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_3_CSP3.csv')
Fixed_Data_4_2_1 = pd.read_csv('YOUR PATH/TDR_25cm_4_CSP3.csv')

CSP3_5 = Fixed_Data_1_2_1['SWC_PI_F_1_2_1']/100
CSP3_6 = Fixed_Data_2_2_1['SWC_PI_F_2_2_1']/100
CSP3_7 = Fixed_Data_3_2_1['SWC_PI_F_3_2_1']/100
CSP3_8 = Fixed_Data_4_2_1['SWC_PI_F_4_2_1']/100

Date1 = Fixed_Data_3_2_1['DATE']

m = Date1
n = CSP3_5
p = CSP3_6
q = CSP3_7
d = CSP3_8


####50 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_3_1 = GWC['SWC_PI_F_1_3_1']/1.42
SWC_2_3_1 = GWC['SWC_PI_F_2_3_1']/1.42
SWC_3_3_1 = GWC['SWC_PI_F_3_3_1']/1.42
SWC_4_3_1 = GWC['SWC_PI_F_4_3_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df9 = SWC_1_3_1.resample('D').mean() 
df10 = SWC_2_3_1.resample('D').mean()
df11 = SWC_3_3_1.resample('D').mean()
df12 = SWC_4_3_1.resample('D').mean()

df_SWC_1_3_1 = pd.DataFrame(df9)
df_SWC_2_3_1 = pd.DataFrame(df10)
df_SWC_3_3_1 = pd.DataFrame(df11)
df_SWC_4_3_1 = pd.DataFrame(df12)

df_SWC_1_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_1_CSP3.csv')
df_SWC_2_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_2_CSP3.csv')
df_SWC_3_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_3_CSP3.csv')
df_SWC_4_3_1.to_csv(r'YOUR PATH TO MAKE/TDR_50cm_4_CSP3.csv')

Fixed_Data_1_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_1_CSP3.csv')
Fixed_Data_2_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_2_CSP3.csv')
Fixed_Data_3_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_3_CSP3.csv')
Fixed_Data_4_3_1 = pd.read_csv('YOUR PATH/TDR_50cm_4_CSP3.csv')

CSP3_9 = Fixed_Data_1_3_1['SWC_PI_F_1_3_1']/100
CSP3_10 = Fixed_Data_2_3_1['SWC_PI_F_2_3_1']/100
CSP3_11 = Fixed_Data_3_3_1['SWC_PI_F_3_3_1']/100
CSP3_12 = Fixed_Data_4_3_1['SWC_PI_F_4_3_1']/100

Date2 = Fixed_Data_3_3_1['DATE']

r = Date2
s = CSP3_9
t = CSP3_10
v = CSP3_11
f = CSP3_12


####100 cm CSP1#####
#turn volumetric to gravimetric by dividing by bulk density
#calculation is done for each individual column, named the same as in the Excel sheet
#Bulk Density=1.42g/cm^3
SWC_1_4_1 = GWC['SWC_PI_F_1_4_1']/1.42
SWC_2_4_1 = GWC['SWC_PI_F_2_4_1']/1.42
SWC_3_4_1 = GWC['SWC_PI_F_3_4_1']/1.42
SWC_4_4_1 = GWC['SWC_PI_F_4_4_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
df13 = SWC_1_4_1.resample('D').mean() 
df14 = SWC_2_4_1.resample('D').mean()
df15 = SWC_3_4_1.resample('D').mean()
df16 = SWC_4_4_1.resample('D').mean()

df_SWC_1_4_1 = pd.DataFrame(df13)
df_SWC_2_4_1 = pd.DataFrame(df14)
df_SWC_3_4_1 = pd.DataFrame(df15)
df_SWC_4_4_1 = pd.DataFrame(df16)

df_SWC_1_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_1_CSP3.csv')
df_SWC_2_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_2_CSP3.csv')
df_SWC_3_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_3_CSP3.csv')
df_SWC_4_4_1.to_csv(r'YOUR PATH TO MAKE/TDR_100cm_4_CSP3.csv')

Fixed_Data_1_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_1_CSP3.csv')
Fixed_Data_2_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_2_CSP3.csv')
Fixed_Data_3_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_3_CSP3.csv')
Fixed_Data_4_4_1 = pd.read_csv('YOUR PATH/TDR_100cm_4_CSP3.csv')

CSP3_13 = Fixed_Data_1_4_1['SWC_PI_F_1_4_1']/100
CSP3_14 = Fixed_Data_2_4_1['SWC_PI_F_2_4_1']/100
CSP3_15 = Fixed_Data_3_4_1['SWC_PI_F_3_4_1']/100
CSP3_16 = Fixed_Data_4_4_1['SWC_PI_F_4_4_1']/100

Date3 = Fixed_Data_3_4_1['DATE']

g = Date3
h = CSP3_13
j = CSP3_14
k = CSP3_15
w = CSP3_16


####175 cm CSP1#####
# This code is provided, but the data is not included in the final figure

#SWC_1_5_1 = GWC['SWC_PI_F_1_5_1']/1.42
#SWC_2_5_1 = GWC['SWC_PI_F_2_5_1']/1.42
#SWC_3_5_1 = GWC['SWC_PI_F_3_5_1']/1.42
#SWC_4_5_1 = GWC['SWC_PI_F_4_5_1']/1.42

#Calculate the daily weighted averages from the weighted depth above
#df17 = SWC_1_5_1.resample('D').mean() 
#df18 = SWC_2_5_1.resample('D').mean()
#df19 = SWC_3_5_1.resample('D').mean()
#df20 = SWC_4_5_1.resample('D').mean()

#df_SWC_1_5_1 = pd.DataFrame(df17)
#df_SWC_2_5_1 = pd.DataFrame(df18)
#df_SWC_3_5_1 = pd.DataFrame(df19)
#df_SWC_4_5_1 = pd.DataFrame(df20)

#df_SWC_1_5_1.to_csv(r'YOUR PATH TO MAKE/TDR_175cm_1_CSP3.csv')
#df_SWC_2_5_1.to_csv(r'YOUR PATH TO MAKE/TDR_175cm_2_CSP3.csv')
#df_SWC_3_5_1.to_csv(r'YOUR PATH TO MAKE/TDR_175cm_3_CSP3.csv')
#df_SWC_4_5_1.to_csv(r'YOUR PATH TO MAKE/TDR_175cm_4_CSP3.csv')

#Fixed_Data_1_5_1 = pd.read_csv('YOUR PATH/TDR_175cm_1_CSP3.csv')
#Fixed_Data_2_5_1 = pd.read_csv('YOUR PATH/TDR_175cm_2_CSP3.csv')
#Fixed_Data_3_5_1 = pd.read_csv('YOUR PATH/TDR_175cm_3_CSP3.csv')
#Fixed_Data_4_5_1 = pd.read_csv('YOUR PATH/TDR_175cm_4_CSP3.csv')

#CSP3_17 = Fixed_Data_1_5_1['SWC_PI_F_1_5_1']/100
#CSP3_18 = Fixed_Data_2_5_1['SWC_PI_F_2_5_1']/100
#CSP3_19 = Fixed_Data_3_5_1['SWC_PI_F_3_5_1']/100
#CSP3_20 = Fixed_Data_4_5_1['SWC_PI_F_4_5_1']/100

#Date4 = Fixed_Data_3_5_1['DATE']

#e = Date4
#i = CSP3_17
#o = CSP3_18
#u = CSP3_19
#l = CSP3_20


months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig, axs = plt.subplots(2,2,figsize=(15,10),
                        constrained_layout = True)
axs[0,0].plot(x,y, color='darkmagenta', label = 'IMZ 1')
axs[0,0].plot(x,a, color = 'limegreen', label = 'IMZ 2')
axs[0,0].plot(x,b, color = 'deeppink', label = 'IMZ 3')
axs[0,0].plot(x,c, color = 'darkturquoise', label = 'IMZ 5')
axs[0,0].set_title('10cm Depth')
axs[0,0].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[0,0].tick_params(axis = 'x', rotation = 25)
axs[0,0].set_xticks(np.linspace(1,4600,13))
axs[0,0].set_ylim(0.07,0.385)
axs[0,0].set_xticklabels([])
axs[0,0].legend(loc='upper right')
axs[0,1].plot(m,n, color = 'darkmagenta', label = 'IMZ 1')
axs[0,1].plot(m,p, color = 'limegreen', label = 'IMZ 2')
axs[0,1].plot(m,q, color = 'deeppink', label = 'IMZ 3')
axs[0,1].plot(m,d, color = 'darkturquoise', label = 'IMZ 5')
axs[0,1].set_title('25cm Depth')
axs[0,1].set_xticks(np.linspace(1,4600,13))
axs[0,1].tick_params(axis = 'x', rotation = 25)
axs[0,1].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[0,1].set_ylim(0.07,0.385)
axs[0,1].set_xticklabels([])
axs[0,1].legend(loc='upper right')
axs[1,0].plot(r,s, color = 'darkmagenta', label = 'IMZ 1')
axs[1,0].plot(r,t, color = 'limegreen', label = 'IMZ 2')
axs[1,0].plot(r,v, color = 'deeppink', label = 'IMZ 3')
axs[1,0].plot(r,f, color = 'darkturquoise', label = 'IMZ 5')
axs[1,0].set_title('50cm Depth')
axs[1,0].set_xlabel('Date')
axs[1,0].set_xticks(np.linspace(1,4600,13))
axs[1,0].tick_params(axis = 'x', rotation = 25)
axs[1,0].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[1,0].set_ylim(0.07,0.385)
axs[1,0].legend(loc='upper right')
axs[1,1].plot(g,h, color = 'darkmagenta', label = 'IMZ 1')
axs[1,1].plot(g,j, color = 'limegreen', label = 'IMZ 2')
axs[1,1].plot(g,k, color = 'deeppink', label = 'IMZ 3')
axs[1,1].plot(g,w, color = 'darkturquoise', label = 'IMZ 5')
axs[1,1].set_title('100cm Depth')
axs[1,1].set_xlabel('Date')
axs[1,1].set_xticks(np.linspace(1,4600,13))
axs[1,1].tick_params(axis = 'x', rotation = 25)
axs[1,1].set_ylabel('SWC (cm\u00b3/cm\u00b3)')
axs[1,1].set_ylim(0.07,0.385)
axs[1,1].legend(loc='upper right')
fig.suptitle('TDR Measured Soil Water Content 2011 to 2023 - CSP3', fontsize = 22)
fig.savefig('YOUR PATH TO SAVE/TDR_SoilWaterContent_CSP3.png')
plt.show()