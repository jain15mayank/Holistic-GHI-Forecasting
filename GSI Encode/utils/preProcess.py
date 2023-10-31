# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:37:22 2023

@author: Mayank Jain
"""

# Main code starts here
import numpy as np
import pandas as pd
from copy import copy, deepcopy

def GetSFilteredData(input_data):
    input_data = input_data.sort_values(by = ['Year', 'Month', 'Day'])
    input_data = input_data.reset_index(drop = True)
    ts = []
    for i in range(len(input_data)):
        ts.append(pd.Timestamp(year = input_data['Year'][i], month = input_data['Month'][i], 
                               day = input_data['Day'][i], hour = input_data['Hour'][i],
                               minute = input_data['Minute'][i]))
        
    input_data.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)
    input_data.insert(loc = 0, column='TimeStamp', value = ts)
    input_data = input_data[(input_data['DHI'] != 0) & (input_data['DNI'] != 0) & (input_data['GHI'] != 0)
                            & (input_data['Clearsky DHI'] != 0) & (input_data['Clearsky DNI'] != 0) & 
                            (input_data['Clearsky GHI'] != 0)]
    input_data = input_data.reset_index(drop = True)
    
    return input_data

def getClearSkyDataGHI(SData,tGhi,tDni):
    Sdata,t_ghi, t_dni = deepcopy(SData),deepcopy(tGhi),deepcopy(tDni)
    Sdata.drop(Sdata[(Sdata["DHI"] > Sdata["Clearsky DHI"] + 1)].index, inplace = True)
    Sdata.drop(Sdata[(Sdata['GHI']/Sdata['Clearsky GHI']) < t_ghi].index, inplace = True)
    Sdata.drop(Sdata[(Sdata['DNI']/Sdata['Clearsky DNI'] < t_dni)].index, inplace = True)
    Sdata = Sdata.reset_index(drop = True)
    return Sdata

def GetGFilteredData(input_data):
    ts = pd.to_datetime(input_data['TSstr'], format="%Y%m%d-%H%M%S")
    input_data.insert(loc = 0, column='TimeStamp', value = ts)
    input_data.drop(['TSstr'], axis=1, inplace=True)
    input_data.dropna(inplace=True)
    input_data = input_data.reset_index(drop = True)
    return input_data

def removePartialClearSkyDates(CSdata, th=24):
    # th: minimum readings which must be present on a date for it to be retained
    newData = deepcopy(CSdata)
    newData['Date'] = newData['TimeStamp'].dt.normalize()
    a = newData.groupby("Date").size().values
    a = pd.DataFrame(newData["Date"].drop_duplicates()).assign(Count=a)
    
    for index, row in a.iterrows():
        if row["Count"] < 24:
            newData.drop(newData[(newData["Date"]==row["Date"])].index, inplace = True)
            newData = newData.reset_index(drop = True)
    newData.drop(labels = 'Date', axis = 1, inplace = True)
    return newData

def checkContinuousSubSequence(dataDF, intPeriod=pd.Timedelta(15, 'm'), th=24):
    newData = deepcopy(dataDF)
    newData['Date'] = newData['TimeStamp'].dt.normalize()
    date = pd.to_datetime(0)
    prevTS = pd.to_datetime(0)
    contCount = 0
    rowIndicesToRemove = []
    for index, row in newData.iterrows():
        if date==row["Date"]:
            if row["TimeStamp"] - prevTS == intPeriod:
                contCount += 1
            else:
                if contCount<th:
                    # print(index-1, prevTS, contCount)
                    rowIndicesToRemove += list(range(index-contCount,index))
                contCount = 1
        else:
            if not contCount==0:
                if contCount<th:
                    # print(index-1, prevTS, contCount)
                    rowIndicesToRemove += list(range(index-contCount,index))
            contCount = 1
            date = row["Date"]
        prevTS = row["TimeStamp"]
    if contCount<th:
        # print(index, prevTS, contCount)
        rowIndicesToRemove += list(range(index+1-contCount,index+1))
    newData.drop(rowIndicesToRemove, inplace=True)
    newData.drop(labels = 'Date', axis = 1, inplace = True)
    newData = newData.reset_index(drop = True)
    return newData

def preProcessComplete(returnCols='OnlyGround', th=40):
    # Read NSRDB Satellite Data and Create TimeStamp Column and Remove Other Date/Time Realted Columns
    # Also Remove Values where DHI/DNI/GHI/Clearsky DHI/Clearsky DNI/Clearsky GHI are 0
    data_2017 = pd.read_csv('data/NSRDB Satellite Data - 2017-2019/399155_48.69_2.18_2017.csv', skiprows=2)
    data_2018 = pd.read_csv('data/NSRDB Satellite Data - 2017-2019/399155_48.69_2.18_2018.csv', skiprows=2)
    data_2019 = pd.read_csv('data/NSRDB Satellite Data - 2017-2019/399155_48.69_2.18_2019.csv', skiprows=2)
    
    S_data = pd.concat([GetSFilteredData(data_2017), GetSFilteredData(data_2018),
                        GetSFilteredData(data_2019)], axis=0, join='inner')
    S_data = S_data.reset_index(drop = True)
    
    th1, th2 = 0.95, 0.95
    CS_data = getClearSkyDataGHI(S_data,th1,th2)
    
    # Read SIRTA Ground Based Data for GHI
    G_data_complete = pd.read_csv('data/SIRTA Ground Data - 2017-2019.csv', names=['TSstr', 'GHI'])
    
    G_data_complete = GetGFilteredData(G_data_complete)
    
    # Perform Final Merging and removal of partial or non-continuous dates
    finData = G_data_complete.merge(CS_data, on='TimeStamp', how='inner')
    finData = finData.reset_index(drop = True)
    
    finData = removePartialClearSkyDates(finData, th=th)
    finData = checkContinuousSubSequence(finData, th=th)
    
    if returnCols=='OnlyGround':
        finData.drop(labels = ['Clearsky GHI', 'Clearsky DHI', 'Clearsky DNI', 'DNI', 'DHI', 'GHI_y'], axis = 1, inplace = True)
        finData.rename(columns={"TimeStamp": "TimeStamp", "GHI_x": "GHI"}, inplace=True)
    return finData