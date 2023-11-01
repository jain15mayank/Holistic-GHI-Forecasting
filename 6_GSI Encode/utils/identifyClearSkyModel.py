# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:09:39 2023

@author: Mayank Jain
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

from ClearskyModelFunction import HLJ, FR, DPP, BD, KC, RS, BD14, BR, Sc, SP, HS
from preProcess import preProcessComplete
from NOAAsolcalc import calcSZAandSAA

def calcSolarZenithAngle(N, H, lat):
    # N is the Day of the year (from 1-365)
    # H is the Hour of the day (from 0-23)
    # lat is the Latitude of the location

    # Hour angle
    hourAngle = (H-12)*15 #(in degrees)
    
    # Declination of the Sun
    #sunDeclination = (-23.44)*(np.cos(np.radians((360/365)*(N+10)))) #(in degrees)
    sunDeclination = np.arcsin(0.39779* np.cos(np.radians((0.98565*(N+10)) +
                (1.914*np.sin(np.radians(0.98565*(N-2))))))) #(in radians) - Better than earlier approximantion

    # Solar Zenith Angle
    solarZenithAngle = np.degrees(np.arccos((np.sin(np.radians(lat))*np.sin(sunDeclination)) + 
                            (np.cos(np.radians(lat))*np.cos(sunDeclination)*np.cos(np.radians(hourAngle)))))

    return solarZenithAngle

def calcETirradiance_exact(n, sza):
    # n = nth day of the year [0, 366]
    b = (2*np.pi*n)/365
    # Calculate (Rav/R)^2 ; where Rav is the average and R is the actual sun-earth distance
    RavR2 = 1.00011 + (0.034221 * np.cos(b)) + (0.001280 * np.sin(b)) + (
        0.000719 * np.cos(2*b)) + (0.000077 * np.sin(2*b))
    i0 = 1361 * RavR2 * np.cos(np.radians(sza))
    return i0

def addSZAandEtHI(dataDF, Latitude, Longitude):
    # finData['SZA'] = [calcSolarZenithAngle(n, h, Latitude) for n, h in [(ts.dayofyear, ts.hour) 
    #                                                                     for ts in finData['TimeStamp']]]
    szaVals = np.zeros(len(dataDF['TimeStamp']))
    saaVals = np.zeros(len(dataDF['TimeStamp']))
    ethiVals = np.zeros(len(dataDF['TimeStamp']))
    for i, ts in enumerate(dataDF['TimeStamp']):
        temp = calcSZAandSAA(ts, Latitude, Longitude)
        szaVals[i] = temp[0]
        saaVals[i] = temp[1]
        ethiVals[i] = calcETirradiance_exact(ts.dayofyear, temp[0])
    dataDF['SZA'] = szaVals
    dataDF['SAA'] = saaVals
    dataDF['EtHI'] = ethiVals
    # dataDF['SZA'] = [calcSZAandSAA(ts, Latitude, Longitude)[0] for ts in dataDF['TimeStamp']]
    # dataDF['SAA'] = [calcSZAandSAA(ts, Latitude, Longitude)[1] for ts in dataDF['TimeStamp']]
    # dataDF['EtHI'] = [calcETirradiance_exact(n, sum(dataDF['SZA'].where(dataDF['TimeStamp']==TS, other=0))) 
    #                    for (n,TS) in [(ts.dayofyear,ts) for ts in dataDF['TimeStamp']]]
    # Adjustment for Clear Sky Model Processing
    dataDF.loc[dataDF.SZA>=90, 'SZA'] = 89.999
    dataDF.loc[dataDF.EtHI<=0, 'EtHI'] = 0.001
    return dataDF

def addClearSkyModelValues(dataDF, elevation):
    GHI_HLJ,DNI_HLJ,DHI_HLJ = HLJ(dataDF, elevation)
    GHI_FR,DNI_FR,DHI_FR = FR(dataDF, elevation)
    GHI_DPP,DNI_DPP,DHI_DPP = DPP(dataDF)
    GHI_BD = BD(dataDF)
    GHI_KC = KC(dataDF)
    GHI_RS = RS(dataDF)
    GHI_BD14 = BD14(dataDF)
    GHI_BR,DNI_BR,DHI_BR = BR(dataDF)
    GHI_Sc,DNI_Sc,DHI_Sc = Sc(dataDF)
    GHI_SP,DNI_SP,DHI_SP = SP(dataDF)
    GHI_HS,DNI_HS,DHI_HS = HS(dataDF)
    
    ghi_model = pd.DataFrame(list(zip(GHI_HLJ,GHI_FR,GHI_DPP,GHI_BD,GHI_KC,
                                      GHI_RS,GHI_BD14,GHI_BR,GHI_Sc,GHI_SP,GHI_HS)),
                             columns =['GHI_HLJ','GHI_FR','GHI_DPP','GHI_BD','GHI_KC',
                                       'GHI_RS','GHI_BD14','GHI_BR','GHI_Sc','GHI_SP','GHI_HS'])
    ghiDF = pd.concat([dataDF,ghi_model],axis=1, join='inner')
    return ghiDF

def getRMSErrorMonthwise(data,model):
    RMError = {}
    monthNo = np.unique(data['Month'])
    month = ['January','February','March','April','May','June','July','August',
             'September','October','November','December']
    for i in monthNo:
        M_data = data[(data['Month'] == i)]
        RMError[month[i-1]] = np.round(np.sqrt(mean_squared_error(M_data['GHI']/M_data['GHI'],
                                                                M_data['GHI_'+model]/M_data['GHI'])),2)
    for i in range(len(month)):
        if not month[i] in RMError:
            RMError[month[i]] = np.nan
    return (RMError)

def getValuesForDate(data, date):
    # Date must be in format: dd-mm-yyyy as a string
    date = pd.to_datetime(date, format='%d-%m-%Y')
    tempData = data.where(np.logical_and(np.logical_and(data["TimeStamp"].dt.day==date.day,
                                                        data["TimeStamp"].dt.month==date.month),
                                         data["TimeStamp"].dt.year==date.year)).dropna()
    return tempData

def plotGHIDateData(ghiDateData, inclOriData=True, figPath='GHIDateData.pdf'):
    plt.locator_params(axis='x', nbins=10)
    _ = plt.figure(figsize=(15,10))
    x = ghiDateData.TimeStamp.dt.strftime('%H:%M').to_numpy()
    xTicks = list(range(0,len(x),int(np.ceil(len(x)/15))))
    xTicks = np.take(x, xTicks)
    plt.plot(x,ghiDateData.GHI_HLJ,'mediumvioletred',linestyle='dashed',label = 'HLJ')
    plt.plot(x,ghiDateData.GHI_FR,'darkmagenta',linestyle='dashed',label = 'FR')
    plt.plot(x,ghiDateData.GHI_DPP,'darkorchid',linestyle='dashed',label = 'DPP')
    plt.plot(x,ghiDateData.GHI_BD,'mediumblue',linestyle='dashed',label = 'BD')
    plt.plot(x,ghiDateData.GHI_KC,'teal',linestyle='dashed',label = 'KC')
    plt.plot(x,ghiDateData.GHI_RS,'seagreen',linestyle='dashed',label = 'RS')
    plt.plot(x,ghiDateData.GHI_BD14,'olive',linestyle='dashed',label = 'BD14')
    plt.plot(x,ghiDateData.GHI_BR,'darkgoldenrod',linestyle='dashed',label = 'BR')
    plt.plot(x,ghiDateData.GHI_Sc,'darkorange',linestyle='dashed',label = 'Sc')
    plt.plot(x,ghiDateData.GHI_SP,'firebrick',linestyle='dashed',label = 'SP')
    plt.plot(x,ghiDateData.GHI_HS,'deeppink',linestyle='dashed',label = 'HS')
    if inclOriData:
        plt.plot(x,ghiDateData.GHI,'r-*', label = 'Observed data')
    plt.title(ghiDateData.TimeStamp.dt.date[0])
    plt.xticks(xTicks, rotation = 45)
    plt.xlabel('Time of Day')
    plt.ylabel('Global Horizontal Irradiance $(W/m^2)$')
    plt.legend(ncol=6,bbox_to_anchor=(0.49, -0.15), loc="upper center")
    plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

def getCSMvaluesForDate(myDate, latitude, longitude, elevation, valPerDay=(60*24)):
    z = pd.date_range(start=pd.to_datetime(myDate, format='%d-%m-%Y'),
                      end=pd.to_datetime(myDate, format='%d-%m-%Y')+pd.Timedelta(1,'day'),
                      periods=valPerDay+1)[:-1]
    z = pd.DataFrame(z, columns=['TimeStamp'])
    z = addSZAandEtHI(z, latitude, longitude)
    z1 = addClearSkyModelValues(z, elevation)
    num = z1._get_numeric_data()
    num[num < 0] = 0
    return z1

def addCSMvaluesToDataframe(myDataframe, latitude, longitude, elevation):
    if 'TimeStamp' in myDataframe.columns:
        colName = None
    elif 'Timestamp' in myDataframe.columns:
        colName = 'Timestamp'
    elif 'timestamp' in myDataframe.columns:
        colName = 'timestamp'
    else:
        raise KeyError('TimeStamp column not found in the inputted dataframe!')
    z = deepcopy(myDataframe).rename(columns={'Timestamp':'TimeStamp', 'timestamp':'TimeStamp'})
    z = addSZAandEtHI(z, latitude, longitude)
    z1 = addClearSkyModelValues(z, elevation)
    num = z1._get_numeric_data()
    num[num < 0] = 0
    if colName is not None:
        z1 = z1.rename(columns={'TimeStamp' : colName})
    return z1

def getBestCSMvaluesForDate(myDate, latitude, longitude, elevation, valPerDay=(60*24)):
    allCSMdf = getCSMvaluesForDate(myDate, latitude, longitude, elevation, valPerDay=valPerDay)
    monthNum = pd.to_datetime(myDate, format='%d-%m-%Y').month
    if monthNum==1 or monthNum==7 or monthNum==11:
        bestCSMdf = deepcopy(allCSMdf[['TimeStamp','SZA','SAA','EtHI','GHI_RS']])
        bestCSMdf.rename(columns={'GHI_RS':'CSM'}, inplace=True)
    elif monthNum==2 or monthNum==3 or monthNum==12:
        bestCSMdf = deepcopy(allCSMdf[['TimeStamp','SZA','SAA','EtHI','GHI_Sc']])
        bestCSMdf.rename(columns={'GHI_Sc':'CSM'}, inplace=True)
    else:
        bestCSMdf = deepcopy(allCSMdf[['TimeStamp','SZA','SAA','EtHI','GHI_HLJ']])
        bestCSMdf.rename(columns={'GHI_HLJ':'CSM'}, inplace=True)
    return bestCSMdf

def addBestCSMvaluesToDataframe(myDataframe, latitude, longitude, elevation):
    if 'TimeStamp' in myDataframe.columns:
        colName = 'TimeStamp'
    elif 'Timestamp' in myDataframe.columns:
        colName = 'Timestamp'
    elif 'timestamp' in myDataframe.columns:
        colName = 'timestamp'
    else:
        raise KeyError('TimeStamp column not found in the inputted dataframe!')
    ori_columns = list(myDataframe.columns)
    allCSMdf = addCSMvaluesToDataframe(myDataframe, latitude, longitude, elevation)
    bestCSMdf = deepcopy(allCSMdf)
    bestCSMdf['CSM'] = np.where(
        np.logical_or(np.logical_or(allCSMdf[colName].dt.month==1, allCSMdf[colName].dt.month==7), allCSMdf[colName].dt.month==11),
        allCSMdf['GHI_RS'],
        np.where(
           np.logical_or(np.logical_or(allCSMdf[colName].dt.month==2, allCSMdf[colName].dt.month==3), allCSMdf[colName].dt.month==12),
            allCSMdf['GHI_Sc'],
            allCSMdf['GHI_HLJ']
        ))
    ori_columns += ['SZA','SAA','CSM']
    return bestCSMdf[ori_columns]

def plotBestCSMDateData(csmDateData, figPath='BestCSMDateData.pdf'):
    plt.locator_params(axis='x', nbins=10)
    _ = plt.figure(figsize=(15,10))
    x = csmDateData.TimeStamp.dt.strftime('%H:%M').to_numpy()
    xTicks = list(range(0,len(x),int(np.ceil(len(x)/15))))
    xTicks = np.take(x, xTicks)
    plt.plot(x,csmDateData.CSM,'r-',label = 'Clear Sky Model')
    plt.title(ghiDateData.TimeStamp.dt.date[0])
    plt.xticks(xTicks, rotation = 45)
    plt.xlabel('Time of Day')
    plt.ylabel('Global Horizontal Irradiance $(W/m^2)$')
    plt.legend(loc="upper right")
    plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

if __name__ == '__main__':
    Latitude = 48.7
    Longitude = 2.2
    Elevation = 176
    
    '''
    # Read and Pre-Process the Satellite and Ground-Based GHI Data
    '''
    finData = preProcessComplete(returnCols='OnlyGround', th=32)
    # Adjustment to include more refined data for other months, except December
    tempData = finData.where(finData["TimeStamp"].dt.month==12).dropna()
    finData = preProcessComplete(returnCols='OnlyGround', th=35)
    finData = pd.concat([tempData, finData], ignore_index=True)
    finData.sort_values(by='TimeStamp', inplace = True)
    finData = finData.reset_index(drop = True)
    
    '''
    # Calculate SZA, EtHI and Clear Sky Model Estimates
    '''
    finData = addSZAandEtHI(finData, Latitude, Longitude)

    G_data = addClearSkyModelValues(finData, Elevation)
    
    G_data['Month'] = G_data['TimeStamp'].dt.month
    
    G_data.sort_values(by='Month',inplace=True)
    G_data = G_data.reset_index(drop = True)
    
    '''
    # Perform Comparative analysis to identify best Clear Sky Model
    '''
    merror_HLJ = getRMSErrorMonthwise(G_data,'HLJ')
    merror_FR = getRMSErrorMonthwise(G_data,'FR')
    merror_DPP = getRMSErrorMonthwise(G_data,'DPP')
    merror_BD = getRMSErrorMonthwise(G_data,'BD')
    merror_KC = getRMSErrorMonthwise(G_data,'KC')
    merror_RS = getRMSErrorMonthwise(G_data,'RS')
    merror_BD14 = getRMSErrorMonthwise(G_data,'BD14')
    merror_BR = getRMSErrorMonthwise(G_data,'BR')
    merror_Sc = getRMSErrorMonthwise(G_data,'Sc')
    merror_SP = getRMSErrorMonthwise(G_data,'SP')
    merror_HS = getRMSErrorMonthwise(G_data,'HS')
    
    def dict2monthwiseList(monthDict):
        months = ['January','February','March','April','May','June','July','August',
                  'September','October','November','December']
        monthWiseList = []
        for month in months:
            monthWiseList += [monthDict[month]]
        return monthWiseList
    
    MSE_dict = {
                'BD14': dict2monthwiseList(merror_BD),
                'HLJ': dict2monthwiseList(merror_HLJ),
                'DPP': dict2monthwiseList(merror_DPP),
                'FR': dict2monthwiseList(merror_FR),
                'BR': dict2monthwiseList(merror_BR),
                'BD': dict2monthwiseList(merror_BD),
                'KC': dict2monthwiseList(merror_KC),
                'RS': dict2monthwiseList(merror_RS),
                'SP': dict2monthwiseList(merror_SP),
                'Sc': dict2monthwiseList(merror_Sc),
                'HS': dict2monthwiseList(merror_HS)
        }
    
    MSE_df = pd.DataFrame(MSE_dict).T
    
    add_font_size = 5
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = False
    plt.rc('font', size=16+add_font_size)         # controls default text sizes
    plt.rc('axes', titlesize=20+add_font_size)    # fontsize of the axes title
    plt.rc('axes', labelsize=20+add_font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20+add_font_size)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=20+add_font_size)   # fontsize of the tick labels
    plt.rc('legend', fontsize=20+add_font_size)   # legend fontsize
    plt.rc('figure', titlesize=20+add_font_size)  # fontsize of the figure title
    
    month = ['January','February','March','April','May','June','July','August','September',
                                   'October','November','December']
    min_in_each_column = np.min(MSE_df, axis=0)
    fig, ax = plt.subplots(figsize=(11, 9))
    g = sns.heatmap(MSE_df, mask=MSE_df == min_in_each_column, annot = True, cmap="YlGnBu", xticklabels=month)
    g = sns.heatmap(MSE_df, mask=MSE_df != min_in_each_column,
                    annot_kws={'fontsize':25,'fontweight': 1000,'c':'r'},
                    annot = True, cmap="YlGnBu",cbar = False,xticklabels=month)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
    
    plt.ylabel('Models')
    plt.xlabel('Months')
    plt.savefig('RMSEheatmap.pdf', bbox_inches = 'tight', pad_inches = 0.05)
    plt.show()
    
    '''
    # Create plots for specific date
    '''
    myDate = '16-10-2017' # dd-mm-yyyy
    ghiDateData = addClearSkyModelValues(finData, Elevation)
    ghiDateData = getValuesForDate(ghiDateData, myDate)
    ghiDateData = ghiDateData.reset_index(drop = True)
    plotGHIDateData(ghiDateData)
    plotGHIDateData(getCSMvaluesForDate(myDate, Latitude, Longitude, Elevation),
                    inclOriData=False, figPath='wholeDayCSM.pdf')
    
    '''
    # Identify Best CSM for a specific date and plot its curve
    '''
    bestCSM = getBestCSMvaluesForDate(myDate, Latitude, Longitude, Elevation)
    plotBestCSMDateData(bestCSM)