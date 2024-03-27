import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given
#This code is used to create plots of N, N0, and BWE for each site and year


##################################
#CSP1
##################################

### 2019 ###
NPIV = pd.read_csv('YOUR PATH/CSP12019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2019.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2019.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2019.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP12019BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel(u'BWE (mm)', color=color_VWC1, fontsize = 10)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 180, 10))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2019 CSP1')
plt.savefig('YOUR PATH TO SAVE/CSP1_2019.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2020 ###
NPIV = pd.read_csv('YOUR PATH/CSP12020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2020.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2020.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2020.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP12020BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2020 CSP1')
plt.savefig('YOUR PATH TO SAVE/CSP1_2020.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2021 ###
NPIV = pd.read_csv('YOUR PATH/CSP12021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2021.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2021.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2021.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP12021BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2021 CSP1')
plt.savefig('YOUR PATH TO SAVE/CSP1_2021.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2022 ###
NPIV = pd.read_csv('YOUR PATH/CSP12022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2022.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2022.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2022.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP12022BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2022 CSP1')
plt.savefig('YOUR PATH TO SAVE/CSP1_2022.png', dpi=300, bbox_inches='tight')
plt.show()



### 2023 ###
NPIV = pd.read_csv('YOUR PATH/CSP12023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP1_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP1_2023.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP12023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP1_2023.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP1_2023.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP12023BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 10))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2023 CSP1')
plt.savefig('YOUR PATH TO SAVE/CSP1_2023.png', dpi=300, bbox_inches='tight')
plt.show()



##########################################
# CSP2
##########################################
 

### 2019 ###
NPIV = pd.read_csv('YOUR PATH/CSP22019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2019.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2019.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2019.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP22019BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 180, 10))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2019 CSP2')
plt.savefig('YOUR PATH TO SAVE/CSP2_2019.png', dpi=300, bbox_inches='tight')
plt.show()



### 2020 ###
NPIV = pd.read_csv('YOUR PATH/CSP22020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2020.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2020.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2020.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP22020BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2020 CSP2')
plt.savefig('YOUR PATH TO SAVE/CSP2_2020.png', dpi=300, bbox_inches='tight')
plt.show()



### 2021 ###
NPIV = pd.read_csv('YOUR PATH/CSP22021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2021.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2021.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2021.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP22021BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2021 CSP2')
plt.savefig('YOUR PATH TO SAVE/CSP2_2021.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2022 ###
NPIV = pd.read_csv('YOUR PATH/CSP22022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2022.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2022.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2022.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP22022BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2022 CSP2')
plt.savefig('YOUR PATH TO SAVE/CSP2_2022.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2023 ###
NPIV = pd.read_csv('YOUR PATH/CSP22023N0.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP2_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP2_2023.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/AmerifluxCSP22023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP2_2023.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP2_2023.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP22023BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(1250, 3300)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2023 CSP2')
plt.savefig('YOUR PATH TO SAVE/CSP2_2023.png', dpi=300, bbox_inches='tight')
plt.show()




########################################
# CSP3
########################################
 

### 2011 ###
NPIV = pd.read_csv('YOUR PATH/CSP32011Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2011.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2011.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2011.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2011.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2011.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32011BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2011 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2011.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2012 ###
NPIV = pd.read_csv('YOUR PATH/CSP32012Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2012.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2012.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2012.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2012.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2012.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32012BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2012 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2012.png', dpi=300, bbox_inches='tight')
plt.show()



### 2013 ###
NPIV = pd.read_csv('YOUR PATH/CSP32013Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2013.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2013.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2013.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2013.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2013.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32013BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2013 CSP3')
plt.savefig('YOUR PATH/CSP3_2013.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2014 ###
NPIV = pd.read_csv('YOUR PATH/CSP32014Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2014.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2014.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2014.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2014.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2014.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32014BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2014 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2014.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2015 ###
NPIV = pd.read_csv('YOUR PATH/CSP32015Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2015.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2015.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2015.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2015.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2015.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32015BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2015 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2015.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2016 ###
NPIV = pd.read_csv('YOUR PATH/CSP32016Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2016.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2016.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2016.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2016.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2016.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32016BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2016 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2016.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2017 ###
NPIV = pd.read_csv('YOUR PATH/CSP32017Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2017.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2017.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2017.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2017.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2017.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32017BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2017 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2017.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2018 ###
NPIV = pd.read_csv('YOUR PATH/CSP32018Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2018.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2018.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2018.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2018.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32018BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2018 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2018.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2019 ###
NPIV = pd.read_csv('YOUR PATH/CSP32019Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2019.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2019.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2019.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2019.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2019.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32019BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2019 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2019.png', dpi=300, bbox_inches='tight')
plt.show()

 

### 2020 ###
NPIV = pd.read_csv('YOUR PATH/CSP32020Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2020.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2020.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2020.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2020.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2020.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32020BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2020 CSP3')
plt.savefig('YOUR PATH/CSP3_2020.png', dpi=300, bbox_inches='tight')
plt.show()



### 2021 ###
NPIV = pd.read_csv('YOUR PATH/CSP32021Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2021.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2021.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2021.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2021.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2021.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32021BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2021 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2021.png', dpi=300, bbox_inches='tight')
plt.show()



### 2022 ###
NPIV = pd.read_csv('YOUR PATH/CSP32022Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2022.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2022.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux_2022.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2022.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2022.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32022BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2022 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2022.png', dpi=300, bbox_inches='tight')
plt.show()



### 2023 ###
NPIV = pd.read_csv('YOUR PATH/CSP32023Npvi.csv', parse_dates= ['DATE'], index_col=['DATE'], na_values=['NaN'])
#NPIV
NPIV1 = NPIV['NPIV']
dailyNPIV1 = NPIV1.resample('D').mean()
N = dailyNPIV1
df = pd.DataFrame(N)
df.to_csv(r'YOUR PATH TO MAKE/NPIV_CSP3_2023.csv')
Fixed_Data = pd.read_csv('YOUR PATH/NPIV_CSP3_2023.csv')
NP = Fixed_Data['NPIV']
Date = Fixed_Data['DATE']
x = Date
y = NP

#N0
GWC = pd.read_csv('YOUR PATH/Ameriflux2023.csv', parse_dates= ['DATE'], index_col=['DATE'],na_values=['-9999'])
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
NPIVCount = NPIV['NPIV']
dailyNPIV = NPIVCount.resample('D').mean()
pwc = dailywavgSWC
lwc = 0.058
soc = 0.0092
N = dailyNPIV
N0 = N / ((0.0808 / ((pwc + lwc + soc) + 0.115)) + 0.372)
df1 = pd.DataFrame(N0)
df1.to_csv(r'YOUR PATH TO MAKE/N0_CSP3_2023.csv')
Fixed_Data1 = pd.read_csv('YOUR PATH/N0_CSP3_2023.csv')
N0 = Fixed_Data1['0']
Date = Fixed_Data1['DATE']
m = Date
n = N0

#BWE
BWEcsv = pd.read_csv('YOUR PATH/CSP32023BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
a = Date
b = BWE

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(m,y, label = r'$N_{pvi}$', color = 'purple', lw=3)
lns2 = ax1.plot(m,n, label = r'$N_{0}$', color = 'orange', lw=3)
lns3 = ax2.scatter(a,b, label = 'BWE', color = 'deeppink')
ax1.set_xlabel("Date")
ax1.set_ylabel("Neutron Counts (cph)")
ax1.tick_params(axis="y")
ax1.set_ylim(750, 1900)
ax1.legend(loc='center right')
ax2.legend(loc='upper right')
color_VWC1 = "#FF1493"
ax2.set_ylabel("BWE (mm)", color=color_VWC1)
ax2.tick_params(axis="y", labelcolor=color_VWC1)
plt.xticks(np.linspace(4, 220, 8))
fig.autofmt_xdate()
ax2.set_ylim(-0.5,8)
plt.title(r'$N_{pvi}$, $N_{0}$, and BWE 2023 CSP3')
plt.savefig('YOUR PATH TO SAVE/CSP3_2023.png', dpi=300, bbox_inches='tight')
plt.show()