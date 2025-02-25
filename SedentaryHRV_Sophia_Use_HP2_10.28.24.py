# STEP1: Read the data
import os
import pandas as pd
import datetime as dt
import numpy as np

# define the polder path, you can modifiy the following folder path
# Note: the folder is unzipped from the data downloaded from Labfront, there are participant folders inside
folderPath = '/Users/sophiaholmqvist/Documents/AAIC Cog Neuro Lab'

# obtain the participants list inside the data folder
allSubList = os.listdir(folderPath)
allSubList.sort()
# Note: In this list, we will ignore the '.DS_Store' and 'Powered by Labfront.txt'

print('The participant foldes found:')
display(allSubList)

# Define an empty DataFrame variable. We will save the CSV data into this dataframe so that we can access the data easily
db = pd.DataFrame()
for i, subName in enumerate(allSubList):
    print('##### Subject ' + str(i+1) + '/' + str(len(allSubList)) + ' is ' + subName + ' (extracted at ' + str(dt.datetime.now()) + ') #####')
    
    # define an empy Series variable to save the CSV data of this participant
    dataOfThisSub = pd.Series(dtype='float64')
    
    # First, we define the participant name (but ignoare the suffix of particiapnt ID)
    dataOfThisSub['A1_subject'] = subName[0:subName.find('_')]
    
    # Then , we read the data we need in this study: 1.BBI, 2.epoch, 3.sleep summary
    # 1. BBI
    dataFolderName = 'garmin-device-bbi'
    dataFolderPath = folderPath + '/' + subName + '/' + dataFolderName
    # make sure that this participant has this folder
    if os.path.isdir(dataFolderPath):
        print('     ----- ' + dataFolderName + ' folder found ')
        # obtain the csv files list
        csvFileList = os.listdir(dataFolderPath)
        # define an empty dataframe varialbe to savethe data from all the CSV files here
        data = pd.DataFrame()
        for csvFileName in csvFileList:
            try:
                # read the data in this CSV file
                data1 = pd.read_csv(dataFolderPath + '/' + csvFileName, skiprows=5) 
                # then, append this data into the dataframe
                data = pd.concat([data, data1], ignore_index=True)
            except:
                # Sometimes, the content of a csv file maybe empty and the above code returns error
                # To avoide this issue, we use the loop of try-except
                pass
        # sort by timestamp
        data = data.sort_values(by='unixTimestampInMs', ascending=True).reset_index(drop=True)
        # remove the duplicate ones by timestamp
        data = data.drop_duplicates(subset=['unixTimestampInMs']).reset_index(drop=True)
        dataOfThisSub['C1_BBI'] = data
    else:
        print('     ===== NO ' + dataFolderName + ' folder... ')
    # 2. Epoch
    dataFolderName = 'garmin-connect-epoch'
    dataFolderPath = folderPath + '/' + subName + '/' + dataFolderName
    # make sure that this participant has this folder
    if os.path.isdir(dataFolderPath):
        print('     ----- ' + dataFolderName + ' folder found ')
        # obtain the csv files list
        csvFileList = os.listdir(dataFolderPath)
        # define an empty dataframe varialbe to savethe data from all the CSV files here
        data = pd.DataFrame()
        for csvFileName in csvFileList:
            try:
                # read the data in this CSV file
                data1 = pd.read_csv(dataFolderPath + '/' + csvFileName, skiprows=5) 
                # then, append this data into the dataframe
                data = pd.concat([data, data1], ignore_index=True)
            except:
                # Sometimes, the content of a csv file maybe empty and the above code returns error
                # To avoide this issue, we use the loop of try-except
                pass
        # sort by timestamp
        data = data.sort_values(by='unixTimestampInMs', ascending=True).reset_index(drop=True)
        # remove the duplicate ones by timestamp
        data = data.drop_duplicates(subset=['unixTimestampInMs']).reset_index(drop=True)
        dataOfThisSub['B1_epoch'] = data
    else:
        print('     ===== NO ' + dataFolderName + ' folder... ')
    # 3. Sleep Summary
    dataFolderName = 'garmin-connect-sleep-summary'
    dataFolderPath = folderPath + '/' + subName + '/' + dataFolderName
    # make sure that this participant has this folder
    if os.path.isdir(dataFolderPath):
        print('     ----- ' + dataFolderName + ' folder found ')
        # obtain the csv files list
        csvFileList = os.listdir(dataFolderPath)
        # define an empty dataframe varialbe to savethe data from all the CSV files here
        data = pd.DataFrame()
        for csvFileName in csvFileList:
            try:
                # read the data in this CSV file
                data1 = pd.read_csv(dataFolderPath + '/' + csvFileName, skiprows=5) 
                # then, append this data into the dataframe
                data = pd.concat([data, data1], ignore_index=True)
            except:
                # Sometimes, the content of a csv file maybe empty and the above code returns error
                # To avoide this issue, we use the loop of try-except
                pass
        # sort by timestamp
        data = data.sort_values(by='unixTimestampInMs', ascending=True).reset_index(drop=True)
        # remove the duplicate ones by timestamp
        data = data.drop_duplicates(subset=['unixTimestampInMs']).reset_index(drop=True)
        dataOfThisSub['B2_sleepSummary'] = data
    else:
        print('     ===== NO ' + dataFolderName + ' folder... ')
    
    # transfer the Series varibale in to DataFrame
    dataOfThisSub = pd.DataFrame([dataOfThisSub])
    dataOfThisSub = dataOfThisSub.sort_index()
    # append the data of this participant into the db
    db = pd.concat([db, dataOfThisSub], ignore_index=True)

db = db.sort_index(axis=1)
# All the data are saved into the db variable
# Now, we can start process the data 
display(db)


# STEP2: Process the data

import pandas as pd
import datetime as dt
import numpy as np

def unixTimeNumber2DtDatetime(t,hourShift):
    import datetime as dt
    import numpy as np
    func = lambda t: dt.datetime.utcfromtimestamp(t/1000)+dt.timedelta(hours=hourShift)
    tt = np.array(list(map(func, t)))
    return tt

# HP: I made modification of this function, now it support two types of filter algo
def bbiFilter(bbiT, bbiMs, methodIndex=1):
    import numpy as np
    import pandas as pd
    # methodIndex=1: Remove outliers by upper and lower limit & moving window
    # methodIndex=2: Remove outliers by upper and lower limit
    if methodIndex==1:
        # Remove outliers by upper and lower limit
        lim1 = 300
        lim2 = 1800
        tf1 = np.logical_and(bbiMs > lim1, bbiMs < lim2)
        bbiMs1 = bbiMs[tf1]
        bbiT1 = bbiT[tf1]
        # Remove outliers by moving window
        win = 10
        lim3 = 0.3 # reject BBI < (70% ave) or > (130% ave)
        lim4 = 2 # reject BBI < (ave - 2*std) or > (ave + 2*std)
        tf2 = np.full(bbiMs1.shape[0], False) # modify this tf2 into True if the BBI is ok
        for i, b in enumerate(bbiMs1):
            if i >= win and i <= bbiMs1.shape[0] - win - 1: # ignaore the beginning and ending part
                ave = np.mean([np.mean(bbiMs1[i - win:i - 1]), np.mean(bbiMs1[i + 1:i + win])])
                std = np.mean([np.std(bbiMs1[i - win:i - 1]), np.std(bbiMs1[i + 1:i + win])])
                if b > (1 - lim3) * ave and b < (1 + lim3) * ave:
                    if b > ave - lim4 * std and b < ave + lim4 * std:
                        tf2[i] = True
        bbiMs2 = bbiMs1[tf2]
        bbiT2 = bbiT1[tf2]
        bbiT_final = bbiT2
        bbiMs_final = bbiMs2
    elif methodIndex==2:
        # Remove outliers by upper and lower limit
        lim1 = 300
        lim2 = 1800
        tf1 = np.logical_and(bbiMs > lim1, bbiMs < lim2)
        bbiMs1 = bbiMs[tf1]
        bbiT1 = bbiT[tf1]
        bbiT_final = bbiT1
        bbiMs_final = bbiMs1
    return bbiT_final, bbiMs_final

def findNearestFiveMinute(t):
    import datetime as dt
    tt = dt.datetime(t.year, t.month, t.day, t.hour, 5*(t.minute//5))
    return tt

# HP: I made modification of this function, now it will output more reliable HRV
## calculate SDNN and RMSSD
def window_SdnnRmssd(data):
    import numpy as np
    t = data['bbiT'].values[0]
    bbi = data['bbiMs'].values
    # only output HRV if there is enough BBI (at least 70%) within this 5min chunk
    if np.sum(bbi)>300000*0.7:
        SDRR = np.std(bbi, ddof=1)
        diff_bbi = np.diff(bbi)
        RMSSD = np.sqrt(np.mean(diff_bbi ** 2))
    else:
        SDRR=np.nan
        RMSSD=np.nan
    idx = ['startTimeOfThisFiveMinChunck','SDRR','RMSSD']
    ser = pd.Series(index=idx,data=[t,SDRR,RMSSD])
    return ser

def isoTimeString2DtDatetime(t):
    import datetime as dt
    import numpy as np
    func = lambda t: dt.datetime.strptime(t[0:19], '%Y-%m-%dT%H:%M:%S')
    tt = np.array(list(map(func, t)))
    return tt

# The datatime variable in dataframe will trun into np.datatime64, which is annoying...
# I make this function to transfer np.datatime64 back to datetime bc I am used to use datetime
def NpDatetime642DtDatetime(t):
    import datetime as dt
    import numpy as np
    func = lambda t: dt.datetime.utcfromtimestamp(t.astype(dt.datetime) / 1000000000)
    tt = np.array(list(map(func, t)))
    return tt

def findNearestDay(t):
    import datetime as dt
    tt = dt.datetime(t.year, t.month, t.day)
    return tt

## calculate mean SDNN and RMSSD every day
def window_meanSdnnRmssd(data):
    import numpy as np
    t = data['startTimeOfThisFiveMinChunck'].values[0]
    # HP: I modified the following a little bit to provide bbi minutes used for the daily mean HRV
    mSDRR = np.nanmean(data['SDRR'].values)
    mRMSSD = np.nanmean(data['RMSSD'].values)
    bbi_consideredMinutes = sum(~np.isnan(data['RMSSD'].values)) * 5
    idx = ['startTimeOfThisOneDayChunck', 'mSDRR', 'mRMSSD', 'bbiMin']
    ser = pd.Series(index=idx, data=[t, mSDRR, mRMSSD, bbi_consideredMinutes])
    return ser
    
# Start to look at each participant, and process the data
# you can do this by a for loop like: for subIndex in range(db.shape[0]): ...
# Here, I'd like to assign a single participant to do the procees out of a for loop, 
# bc you then can stop anywhere, insert a display(varaible) to understand what's going on 
subIndex = 1

# 1. process BBI into SDNN and RMSSD
bbiT = unixTimeNumber2DtDatetime(db.C1_BBI[subIndex].unixTimestampInMs.values,8)
bbiMs = db.C1_BBI[subIndex].bbi.values
# filter the BBI, this part may take some time
bbiT, bbiMs = bbiFilter(bbiT, bbiMs, 1)
# Turn them into a dataframe for processing every 5min easily
df_BBI  = pd.DataFrame()
df_BBI['bbiT'] = bbiT
df_BBI['bbiMs'] = bbiMs
fiveMinLabel = (bbiT-findNearestFiveMinute(bbiT[0]))//dt.timedelta(minutes=5)
df_BBI['fiveMinLabel'] = fiveMinLabel
hrvResult = df_BBI.groupby('fiveMinLabel').apply(window_SdnnRmssd)
# I don't know why sometimes the startTimeOfThisFiveMinChunck is not a datetime variable from my side
# remove the rows that the bbiT is int instead of datetime
func = lambda t: isinstance(t,np.datetime64)
tf = np.array(list(map(func, hrvResult.startTimeOfThisFiveMinChunck.values)))
hrvResult = hrvResult[tf]

# 2. Obtain the duration those are totally sedentary, then derive the mean SDRR and RMSSD everyday
epochT = isoTimeString2DtDatetime(db.B1_epoch[subIndex].isoDate.values)
epochActivityType = db.B1_epoch[subIndex].activityType.values
epochActiveTimeInMs = db.B1_epoch[subIndex].activeTimeInMs.values
# totally sedentary = activityType is 'SEDENTARY' and activeTimeInMs is '900000'
tf = np.logical_and(epochActivityType=='SEDENTARY', epochActiveTimeInMs==900000)
# Now, we obtain the start time of 15min sedentary epoch
startTimeOfFifteenMinSedentaryEpoch = epochT[tf]
# Then, we can refer to these, keep the hrvResult when sedentary, and average them everyday
hrvResult_sedentary = pd.DataFrame()
for st in startTimeOfFifteenMinSedentaryEpoch:
    end = st + dt.timedelta(minutes=15)
    tf = np.logical_and(NpDatetime642DtDatetime(hrvResult.startTimeOfThisFiveMinChunck.values)>=st, NpDatetime642DtDatetime(hrvResult.startTimeOfThisFiveMinChunck.values)<end)
    hrvResult_sedentary = pd.concat([hrvResult_sedentary,hrvResult[tf]], ignore_index=True)
hrvResult_sedentary_t = NpDatetime642DtDatetime(hrvResult_sedentary.startTimeOfThisFiveMinChunck.values)
hrvResult_sedentary['oneDayLabel'] = (hrvResult_sedentary_t-findNearestDay(hrvResult_sedentary_t[0]))//dt.timedelta(days=1)
hrvResult_sedentary_mean = hrvResult_sedentary.groupby('oneDayLabel').apply(window_meanSdnnRmssd)
display(hrvResult_sedentary_mean)

# Convert DataFrame to CSV file
csv_filename = "hrvResult_sedentary_mean.csv"
output_folder = '/Users/sophiaholmqvist/Documents/BBI Processed Data'
csv_filepath = os.path.join(output_folder, csv_filename)
hrvResult_sedentary_mean.to_csv(csv_filepath, index=False)
print(f"HRV sedentary mean data saved to: {csv_filepath}")




# 3. Obtain the duration of sleep period, then derive the mean SDRR and RMSSD during every sleep and day time
# PS: I think you could able to do this by tour own : ), feel free to let me know if you have questions

For example, you have
avg_rmssd = np.nanmean(rmssd_values)
and
daily_rmssd_sdnn.append({'date': date, 'RMSSD': avg_rmssd, 'SDNN': avg_sdnn})

You can modify these like
avg_rmssd = np.nanmean(rmssd_values)
bbi_consideredMinutes = sum(~np.isnan(rmssd_values))*5
and
daily_rmssd_sdnn.append({'date': date, 'RMSSD': avg_rmssd, 'SDNN': avg_sdnn, 'bbiMin': bbi_consideredMinutes})
