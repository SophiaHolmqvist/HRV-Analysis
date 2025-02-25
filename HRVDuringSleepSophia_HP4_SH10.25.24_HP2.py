# Step 1: Read the data (modified to include sleep summary)
import os
import pandas as pd
import datetime as dt
import numpy as np

# Define the folder path, you can modify the following folder path
folderPath = '/Users/sophiaholmqvist/Documents/AAIC Cog Neuro Lab'

# Obtain the participants list inside the data folder
allSubList = os.listdir(folderPath)
allSubList.sort()
print('The participant folders found:')
display(allSubList)

# Define an empty DataFrame variable. We will save the CSV data into this dataframe so that we can access the data easily
db = pd.DataFrame()
for i, subName in enumerate(allSubList):
    print('##### Subject ' + str(i+1) + '/' + str(len(allSubList)) + ' is ' + subName + ' (extracted at ' + str(dt.datetime.now()) + ') #####')
    
    # Define an empty Series variable to save the CSV data of this participant
    dataOfThisSub = pd.Series(dtype='float64')
    
    # First, we define the participant name (but ignore the suffix of participant ID)
    dataOfThisSub['A1_subject'] = subName[0:subName.find('_')]
    
    # Then, we read the data we need in this study: 1. BBI, 2. epoch, 3. sleep summary
    # 1. BBI
    dataFolderName = 'garmin-device-bbi'
    dataFolderPath = folderPath + '/' + subName + '/' + dataFolderName
    if os.path.isdir(dataFolderPath):
        print('     ----- ' + dataFolderName + ' folder found ')
        csvFileList = os.listdir(dataFolderPath)
        data = pd.DataFrame()
        for csvFileName in csvFileList:
            try:
                data1 = pd.read_csv(dataFolderPath + '/' + csvFileName, skiprows=5) 
                data = pd.concat([data, data1], ignore_index=True)
            except:
                pass
        data = data.sort_values(by='unixTimestampInMs', ascending=True).reset_index(drop=True)
        data = data.drop_duplicates(subset=['unixTimestampInMs']).reset_index(drop=True)
        dataOfThisSub['C1_BBI'] = data
    else:
        print('     ===== NO ' + dataFolderName + ' folder... ')
    # 2. Epoch
    dataFolderName = 'garmin-connect-epoch'
    dataFolderPath = folderPath + '/' + subName + '/' + dataFolderName
    if os.path.isdir(dataFolderPath):
        print('     ----- ' + dataFolderName + ' folder found ')
        csvFileList = os.listdir(dataFolderPath)
        data = pd.DataFrame()
        for csvFileName in csvFileList:
            try:
                data1 = pd.read_csv(dataFolderPath + '/' + csvFileName, skiprows=5) 
                data = pd.concat([data, data1], ignore_index=True)
            except:
                pass
        data = data.sort_values(by='unixTimestampInMs', ascending=True).reset_index(drop=True)
        data = data.drop_duplicates(subset=['unixTimestampInMs']).reset_index(drop=True)
        dataOfThisSub['B1_epoch'] = data
    else:
        print('     ===== NO ' + dataFolderName + ' folder... ')
    # 3. Sleep Summary
    dataFolderName = 'garmin-connect-sleep-summary'
    dataFolderPath = folderPath + '/' + subName + '/' + dataFolderName
    if os.path.isdir(dataFolderPath):
        print('     ----- ' + dataFolderName + ' folder found ')
        csvFileList = os.listdir(dataFolderPath)
        data = pd.DataFrame()
        for csvFileName in csvFileList:
            try:
                data1 = pd.read_csv(dataFolderPath + '/' + csvFileName, skiprows=5) 
                data = pd.concat([data, data1], ignore_index=True)
            except:
                pass
        data = data.sort_values(by='unixTimestampInMs', ascending=True).reset_index(drop=True)
        data = data.drop_duplicates(subset=['unixTimestampInMs']).reset_index(drop=True)
        dataOfThisSub['B2_sleepSummary'] = data
    else:
        print('     ===== NO ' + dataFolderName + ' folder... ')
    
    dataOfThisSub = pd.DataFrame([dataOfThisSub])
    dataOfThisSub = dataOfThisSub.sort_index()
    db = pd.concat([db, dataOfThisSub], ignore_index=True)

db = db.sort_index(axis=1)
display(db)

# Step 2: Process the data to calculate RMSSD and SDNN per day during sleep periods
import pandas as pd
import datetime as dt
import numpy as np

def NpDatetime642DtDatetime(t):
    # 輸入 np.dateimte64 的時間序列，轉換成 dt.datetime 的時間序列
    import datetime as dt
    import numpy as np
    func = lambda t: dt.datetime.utcfromtimestamp(t.astype(dt.datetime) / 1000000000)
    tt = np.array(list(map(func, t)))
    return tt
	
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
        bbiT2 = bbiT2.reset_index(drop=True)
        bbiMs2 = bbiMs2.reset_index(drop=True)
        data = pd.DataFrame({'time': bbiT2, 'bbi': bbiMs2})
    elif methodIndex==2:
        # Remove outliers by upper and lower limit
        lim1 = 300
        lim2 = 1800
        tf1 = np.logical_and(bbiMs > lim1, bbiMs < lim2)
        bbiMs1 = bbiMs[tf1]
        bbiT1 = bbiT[tf1]
        bbiT1 = bbiT1.reset_index(drop=True)
        bbiMs1 = bbiMs1.reset_index(drop=True)
        data = pd.DataFrame({'time': bbiT1, 'bbi': bbiMs1})
    return data

# HP: I made modification of these two functions, now it will output more reliable HRV
def calculate_rmssd(sdnn_df):
    # only output HRV if there is enough BBI (at least 70%) within this 5min chunk
    if np.sum(sdnn_df) > 300000 * 0.7:
        RMSSD = np.sqrt(np.mean(np.square(np.diff(sdnn_df))))
    else:
        RMSSD = np.nan
    return RMSSD
def calculate_sdnn(sdnn_df):
    # only output HRV if there is enough BBI (at least 70%) within this 5min chunk
    if np.sum(sdnn_df) > 300000 * 0.7:
        SDNN = np.std(sdnn_df)
    else:
        SDNN = np.nan
    return SDNN

subIndex = 1# Change this to the participant index you want to process
participant_data = db.loc[subIndex]
bbi_data = participant_data['C1_BBI']
sleep_summary = participant_data['B2_sleepSummary']

if not bbi_data.empty and not sleep_summary.empty:
    daily_rmssd_sdnn = []

	## HP: We can easily obtain the timezone shift info by the first sleep record. So that you don't need to manually change it if your participants are from different region.
    tzNum = int(sleep_summary['isoDate'][0][23:26])

    ## HP: the 'date' of each sleep should be determined by 'calendarDate', which is not duplicated between multiple sleep records 
	#sleep_summary['date'] = pd.to_datetime(sleep_summary['isoDate']).dt.date
    sleep_summary['date'] = pd.to_datetime(sleep_summary['calendarDate']).dt.date
    for date, group in sleep_summary.groupby('date'):
        sleep_times = []
        for index, row in group.iterrows():
            start_time = pd.to_datetime(row['isoDate'])
			### HP: The "durationInMs" doesn't contain the awake period. So I modify the end time definition.
            end_time = start_time + pd.to_timedelta(row['durationInMs']+row['awakeDurationInMs'], unit='ms')
            sleep_times.append((start_time, end_time))

        rmssd_values = []
        sdnn_values = []

        for start, end in sleep_times:
            mask = (bbi_data['unixTimestampInMs'] >= start.timestamp() * 1000) & (bbi_data['unixTimestampInMs'] <= end.timestamp() * 1000)
            filtered_bbi = bbi_data[mask]

            if not filtered_bbi.empty:
                filtered_bbi['datetime'] = unixTimeNumber2DtDatetime(filtered_bbi['unixTimestampInMs'], tzNum)
                filtered_bbi = bbiFilter(filtered_bbi['datetime'], filtered_bbi['bbi'], 1)

                if not filtered_bbi.empty:
					## HP: You calculate the whole-night RMSSD and SDNN here, instead of deriving every 5 minutes.
					# I suppose that you should want to derive every 5 minutes so that the following np.mean(rmssd_values) and np.mean(sdnn_values) make sense.
					# The code could be like:
                    num_fiveMin = int(np.floor((end-start)/dt.timedelta(minutes=5)))
                    for index_fiveMin in range(num_fiveMin):
                        filtered_bbi_t = NpDatetime642DtDatetime(filtered_bbi['time'].values)
                        fiveMinChunck_t1 = start.to_pydatetime()+dt.timedelta(minutes=5)*index_fiveMin
                        fiveMinChunck_t2 = start.to_pydatetime()+dt.timedelta(minutes=5)*(index_fiveMin+1)
                        fiveMinChunck_t1 = fiveMinChunck_t1.replace(tzinfo=None)
                        fiveMinChunck_t2 = fiveMinChunck_t2.replace(tzinfo=None)
                        mask = (filtered_bbi_t>=fiveMinChunck_t1)&(filtered_bbi_t<fiveMinChunck_t2)
                        filtered_bbi_fiveMinChunck = filtered_bbi[mask]
                        rmssd = calculate_rmssd(filtered_bbi_fiveMinChunck['bbi'])
                        sdnn = calculate_sdnn(filtered_bbi_fiveMinChunck['bbi'])
                        rmssd_values.append(rmssd)
                        sdnn_values.append(sdnn)
        if rmssd_values:
            #avg_rmssd = np.mean(rmssd_values)
            # HP: use np.nanmean to avoid outcomes of nan
            avg_rmssd = np.nanmean(rmssd_values)
            bbi_consideredMinutes = sum(~np.isnan(rmssd_values)) * 5
        else:
            avg_rmssd = np.nan
            bbi_consideredMinutes = np.nan

        if sdnn_values:
            #avg_sdnn = np.mean(sdnn_values)
            # HP: use np.nanmean to avoid outcomes of nan
            avg_sdnn = np.nanmean(sdnn_values)
        else:
            avg_sdnn = np.nan
        # HP: add more reference to be compared with bbiMin
        daily_rmssd_sdnn.append({'date': date, 'RMSSD': avg_rmssd, 'SDNN': avg_sdnn, 'bbiMin': bbi_consideredMinutes, 'slpMin': (end_time-start_time).total_seconds()/60, 'slpStart':start_time, 'slpEnd':end_time})

    daily_rmssd_sdnn_df = pd.DataFrame(daily_rmssd_sdnn)
    display(daily_rmssd_sdnn_df)
else:
    print("No valid BBI or sleep summary data for this participant.")

    
    
    # Convert DataFrame to CSV file
    csv_filename = "daily_rmssd_sdnn_df.csv"
    output_folder = '/Users/sophiaholmqvist/Documents/BBI Processed Data'
    csv_filepath = os.path.join(output_folder, csv_filename)
    daily_rmssd_sdnn_df.to_csv(csv_filepath, index=False)
    print(f"Sleep HRV data saved to: {csv_filepath}")


