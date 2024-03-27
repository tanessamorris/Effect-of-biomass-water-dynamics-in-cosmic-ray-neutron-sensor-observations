import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To use these sheets as is, an individual .csv file must be made for each year for each data source
#These files are provided in the repository, but the user must change the file path to their own
#Modifications will be required to use data sheets as they are given


#################################################################
############################CSP1#################################
#################################################################

######################2019##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('C:/Users/tessm/OneDrive/Documents/Documents/U of N/CRNS Research/CSP1Data/CSP12019BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP1_2019_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2019 CSP1')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 186, 7))
plt.legend(loc = 'upper right')
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.savefig('YOUR PATH TO SAVE/2019_CSP1_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2020##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP12020BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP1_2020_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2020 CSP1')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2020_CSP1_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2021##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP12021BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP1_2021_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2021 CSP1')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2021_CSP1_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2022##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP12022BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP1_2022_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2022 CSP1')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2022_CSP1_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2023##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP12023BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp.to_csv(r'YOUR PATH TO MAKE/2023_allBWE_frompy.csv')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP1_2023_SE.csv')
BWESEinterp = BWESE.interpolate(method = 'linear', limit_direction = 'forward')
BWESEinterp.to_csv(r'YOUR PATH TO MAKE/2023_allBWESE_frompy.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2023 CSP1')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2023_CSP1_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


#################################################################
############################CSP2#################################
#################################################################

######################2019##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP22019BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP2_2019_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2019 CSP2')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 186, 7))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2019_CSP2_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2020##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP22020BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP2_2020_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2020 CSP2')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2020_CSP2_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2021##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP22021BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP2_2021_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2021 CSP2')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2021_CSP2_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2022##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP22022BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP2_2022_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2022 CSP2')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2022_CSP2_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2023##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP22023BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp.to_csv(r'YOUR PATH TO MAKE/2023_CSP2_allBWE_frompy.csv')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP2_2023_SE.csv')
BWESEinterp.to_csv(r'YOUR PATH TO MAKE/2023_CSP2_allBWESE_frompy.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2023 CSP2')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2023_CSP2_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


#################################################################
############################CSP3#################################
#################################################################

######################2011##########################

#Date-time plots to check BWE and dates

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32011BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2011_SE.csv')
BWESEinterp = BWESE.interpolate(method = 'linear', limit_direction = 'forward')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2011 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 205, 8))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2011_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2012##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32012BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2012_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2012 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2012_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2013##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32013BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2013_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2013 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2013_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2014##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32014BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2014_SE.csv')
BWESEinterp = BWESE.interpolate(method = 'linear', limit_direction = 'forward')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2014 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2014_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2015##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32015BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2015_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2015 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2015_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2016##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32016BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2016_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2016 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2016_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2017##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32017BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2017_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2017 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2017_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2018##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32018BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2018_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2018 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2018_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2019##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32019BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2019_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2019 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO MAKE/2019_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2020##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32020BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2020_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2020 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2020_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2021##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32021BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2021_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2021 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig('YOUR PATH TO SAVE/2021_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2022##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32022BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')

x = Date
y = BWE


#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2022_SE.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95

fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2022 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 244, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig(r'YOUR PATH TO SAVE/2022_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()


######################2023##########################

#Import BWE data, determined by destructive sampling
BWEcsv = pd.read_csv('YOUR PATH/CSP32023BWE.csv')
BWE = BWEcsv['BWE']
Date = BWEcsv['DATE']

#Interpolate between dates
BWEinterp = BWE.interpolate(method = 'linear', limit_direction = 'forward')
BWEinterp.to_csv(r'YOUR PATH TO MAKE/2023_CSP3_allBWE_frompy.csv')

x = Date
y = BWE

#BWE Standard Error
BWESE = pd.read_csv('YOUR PATH/CSP3_2023_SE.csv')
BWESEinterp.to_csv(r'YOUR PATH TO MAKE/2023_CSP3_allBWESE_frompy.csv')
SE = BWESE['SE']
pos95 = (1.96*SE)
a = pos95
fig, ax = plt.subplots()
ax.scatter(x,y, color='darkgreen', label = "BWE", zorder=1)
ax.errorbar(x, y, yerr = a, color='purple', capsize=3, label = "95% CI", zorder=0)
plt.title('BWE Time-Series 2023 CSP3')
fig.autofmt_xdate()
plt.ylim(-0.5,8.2)
plt.xticks(np.linspace(3, 243, 9))
plt.ylabel('BWE (mm)')
plt.xlabel('Date')
plt.legend(loc = 'upper right')
plt.savefig(r'YOUR PATH TO SAVE/2023_CSP3_BWE.png', dpi = 300, bbox_inches = 'tight')
plt.show()