import os
from scipy.signal import butter, filtfilt, argrelextrema
import pandas as pd
import numpy as np
from statistics import mean, stdev, variance
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pickle
from math import isnan


def max_min(data):
    # return array of index(frame) local max and min after removing maximum values are greater than mean max
    # and minimum values that are lower than the mean minima
    max_index = argrelextrema(np.asarray(data), np.greater_equal)
    max_index = max_index[0]  # indexs of max_index
    last = max_index[-1]  # add last max since the repetition can finish with lower value than mean
    max_index = max_index[data[max_index] > mean(data)]
    max_index = np.append(max_index, last)
    min_index = argrelextrema(np.asarray(data), np.less_equal)
    min_index = min_index[0]  # indexs of min_index
    min_index = min_index[data[min_index] < mean(data)]
    return max_index, min_index


def max_min_plots():
    # only plot the max_min_plots:
    path = 'CSV/Raw Data/raw_data_scaled.csv'
    df = pd.read_csv(path)
    fps = 30
    cutoff_freq = 6
    filter_order = 2
    y, x = butter(filter_order, cutoff_freq/(fps/2))

    for i in range(0, len(df.index), 2):
        rightdata = (df.iloc[i]).dropna().to_numpy()
        name = rightdata[1]  # Participant ID
        ex = rightdata[3]
        rightdata = filtfilt(y, x, rightdata[5:])
        leftdata = (df.iloc[i+1]).dropna().to_numpy()
        leftdata = filtfilt(y, x, leftdata[5:])

        right_maxmin = max_min(rightdata)
        left_maxmin = max_min(leftdata)
        # plots
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.set(ylabel="Armpit Degree", xlabel="Frame")
        ax.title.set_text(name+' right hand')
        ax.plot(rightdata, label="Angle", color='dimgray')
        ax.plot(right_maxmin[0], np.asarray(rightdata)[right_maxmin[0]], "x", color='deepskyblue', label="max")
        ax.plot(right_maxmin[1], np.asarray(rightdata)[right_maxmin[1]], "x", color='deeppink', label="min")
        plt.legend(loc='lower right')
        ax = fig.add_subplot(1, 2, 2)
        ax.set(ylabel="Armpit Degree", xlabel="Frame")
        ax.title.set_text(name+' left hand')
        ax.plot(leftdata, label="Angle", color='dimgray')
        ax.plot(left_maxmin[0], np.asarray(leftdata)[left_maxmin[0]], "x", color='deepskyblue', label="max")
        ax.plot(left_maxmin[1], np.asarray(leftdata)[left_maxmin[1]], "x", color='deeppink', label="min")
        plt.legend(loc='lower right')
        fig.savefig('Plots/MaxMin/'+ex+"_"+name+'.png')


def vel_acc_calc(data, fps):
    # Input: np array of data
    # Output: velocity, acceleration, time array.
    N = len(data)
    time = np.linspace(0, N/fps, N)
    vel = np.diff(data)
    delta_t = time[1]-time[0]
    vel = vel/delta_t
    acc = np.diff(vel)/delta_t
    return vel, acc, time


def repetition_divison(max_indx, min_indx):
    """
    :param max_indx - np array of times (index) of local max
    :param min_indx - np array of times (index) of local min
    :return df of repetition : {repetition number, start ind, peak ind, end ind}
    """
    done = False
    rep_count = 0
    iter_maxindx = 0
    iter_minindx = 0
    repetition = []

    while iter_maxindx < len(max_indx) and iter_minindx < len(min_indx):
        # rep_ind = []
        while max_indx[iter_maxindx] > min_indx[iter_minindx]:
            iter_minindx += 1
            if iter_minindx >= len(min_indx):
                done = True
                break
        rep_count += 1
        if iter_minindx-1 < 0:  # The motion starts without local minimum
            rep_ind = [rep_count, float("Nan"), max_indx[iter_maxindx]]
            # print(rep_count, " rep - start in:", 0, " peak in:", max_indx[iter_maxindx], end=" ")
        else:
            rep_ind = [rep_count, min_indx[iter_minindx-1], max_indx[iter_maxindx]]
            # print(rep_count, " rep-start in:", min_indx[iter_minindx-1], " peak in:", max_indx[iter_maxindx], end=" ")
        if done:
            repetition.append(rep_ind)
            # print('\n')
            break
        while max_indx[iter_maxindx] < min_indx[iter_minindx]:
            iter_maxindx += 1
            if iter_maxindx >= len(max_indx):
                break
        # print("finish in:", min_indx[iter_minindx])
        rep_ind.append(min_indx[iter_minindx])
        repetition.append(rep_ind)
    cols_name = ['rep', 'start_frame', 'peak_frame', 'end_frame']
    df_rep = pd.DataFrame(repetition, columns=cols_name)
    return df_rep


def plot_rep_hand(data, data_repDF, fig, num):
    rep_count = 1
    colorsarr = ['brown', 'salmon', 'orange', 'yellow', 'lime', 'dodgerblue', 'blueviolet', 'violet', 'hotpink']*3
    ax = fig.add_subplot(1, 2, num)
    ax.plot(data, color='dimgray')
    for row in range(0, len(data_repDF)):
        rep_temp = data_repDF.iloc[row]
        lab = 'repetition'+str(rep_count)
        if isnan(rep_temp['start_frame']):
            ax.plot(range(0, int(rep_temp['end_frame'])), data[0:int(rep_temp['end_frame'])], label=lab,
                    color=colorsarr[rep_count-1])
        else:
            if isnan(rep_temp['end_frame']):
                ax.plot(range(int(rep_temp['start_frame']), len(data)), data[int(rep_temp['start_frame']):],
                        label=lab, color=colorsarr[rep_count-1])
            else:
                ax.plot(range(int(rep_temp['start_frame']), int(rep_temp['end_frame'])),
                        data[int(rep_temp['start_frame']):int(rep_temp['end_frame'])], label=lab,
                        color=colorsarr[rep_count-1])
        rep_count += 1
        # ax.plot(rep_temp['peak_frame'], data[int(rep_temp['peak_frame'])], "x", color='black', label="peak")
    return ax
    # ax.legend()


def plot_rep(rightdata, rightdata_repDF, leftdata, leftdata_repDF):
    path = 'CSV/Raw Data/raw_data_scaled.csv'
    path = 'CSV/Raw Data/ODS_YDS_all_raw_data.csv'
    df = pd.read_csv(path)
    framepersec = 30  # frame per seconds of nuitrack
    # For filtering
    cutoff_freq = 6
    filter_order = 2
    y, x = butter(filter_order, cutoff_freq/(framepersec/2))

    vel_df = pd.DataFrame()
    fft_df = pd.DataFrame()
    for i in range(0, len(df.index), 2):
        rightdata = (df.iloc[i]).dropna().to_numpy()
        name = rightdata[1]  # Participant ID
        ds = rightdata[2]
        ex = rightdata[3] # ex name
        rightdata = filtfilt(y, x, rightdata[5:])
        leftdata = (df.iloc[i+1]).dropna().to_numpy()
        leftdata = filtfilt(y, x, leftdata[5:])

        # vel_features(right, fps, True, name + ' - right hand')
        rightdata_repDF = repetition_features(rightdata, name, 'right', framepersec, ds, ex)
        leftdata_repDF = repetition_features(leftdata, name, 'left', framepersec, ds, ex)
        fig = plt.figure()
        ax1 = plot_rep_hand(rightdata, rightdata_repDF, fig, 1)
        ax2 = plot_rep_hand(leftdata, leftdata_repDF, fig, 2)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        # plt.show()
        fig.savefig('Plots/Rep/'+ex+"_"+name+'.png')


def vel_plots(data, title, vel, acc, time, mean_sd):
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    ax.set(ylabel="Angle Degree", xlabel="Time (seconds)")
    ax.plot(time, data)
    ax = fig.add_subplot(3, 1, 2)
    ax.set(ylabel="Velocity", xlabel="Time (seconds)")
    ax.plot(time[1:], vel)
    ax.annotate('Mean='+str(round(mean_sd[0], 2))+'\nSD='+str(round(mean_sd[1], 2)), xy=(0, 1),
                xycoords='axes fraction')
    ax = fig.add_subplot(3, 1, 3)
    ax.set(ylabel="Acceleration", xlabel="Time (seconds)")
    ax.plot(time[2:], acc)
    ax.annotate('Mean='+str(round(mean_sd[2], 2))+'\nSD='+str(round(mean_sd[3], 2)), xy=(0, 1),
                xycoords='axes fraction')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def vel_features(data, fps, show_plt, title="no title"):
    # return features of velocity after filtering data by buterworth filter 2nd degree with 6
    if len(data) <= 3:
        mean_sd = [float("Nan")]*4
    else:
        vel, acc, time = vel_acc_calc(data, fps)
        mean_sd = [mean(vel), stdev(vel), mean(acc), stdev(acc)]
        if show_plt:
            vel_plots(data, title, vel, acc, time, mean_sd)
    return mean_sd


def repetition_features(data, participant, hand, framepersec, ds, e, plot_maxmin=False):
    data_maxmin = max_min(data)
    if plot_maxmin:
        max_min_plots(participant+" "+hand, data)
    data_repDF = repetition_divison(data_maxmin[0], data_maxmin[1])
    frames_up_list = []  # The amount of frames in raising (up)
    frames_down_list = []  # The amount of frames in descent (down)
    start_values = []  # The angle value at the beginning of the repetition
    peak_values = []  # The angle value at the peak of the repetition (before decreasing)
    end_values = []  # The angle value at the end of the repetition
    vel_features_up = []  # The velocity and acceleration mean and sd values in raising
    vel_features_down = []  # The velocity and acceleration mean and sd values in descent
    for row in range(0, len(data_repDF)):
        rep_temp = data_repDF.iloc[row]
        peak_values.append(data[int(rep_temp[2])])
        try:
            start_values.append(data[int(rep_temp[1])])
        except ValueError:
            start_values.append(float("Nan"))
        try:
            end_values.append(data[int(rep_temp[3])])
        except ValueError:
            end_values.append(float("Nan"))
        frames_up = rep_temp[2] - rep_temp[1]  # Peak - Start
        frames_up_list.append(frames_up)
        frames_down = rep_temp[3] - rep_temp[2]  # End - Peak
        frames_down_list.append(frames_down)
        try:
            vel_features_up.append(vel_features(data[int(rep_temp[1]):int(rep_temp[2])], framepersec, False))
        except ValueError:
            vel_features_up.append([float("Nan") * 4])
        try:
            vel_features_down.append(vel_features(data[int(rep_temp[2]):int(rep_temp[3])], framepersec, False))
        except ValueError:
            vel_features_down.append([float("Nan") * 4])

    data_repDF["start_value"] = start_values
    data_repDF["peak_value"] = peak_values
    data_repDF["end_value"] = end_values
    data_repDF["num_frames_up"] = frames_up_list
    data_repDF["num_frames_down"] = frames_down_list
    data_repDF.insert(0, "hand", [hand] * len(data_repDF))
    data_repDF.insert(0, "Participant", [participant] * len(data_repDF))
    data_repDF.insert(0, "Exercise", [e] * len(data_repDF))
    data_repDF.insert(0, "Source", [ds]*len(data_repDF))

# (right[int(rep_temp[2])]-right[int(rep_temp[1])])/frames_up # - Vel calc by frame.
    veldf_down = pd.DataFrame(vel_features_down, columns=['vel_mean_down', 'vel_sd_down',
                                                          'acc_mean_down', 'acc_sd_down'])
    veldf_up = pd.DataFrame(vel_features_up, columns=['vel_mean_up', 'vel_sd_up', 'acc_mean_up', 'acc_sd_up'])
    data_repDF = pd.concat([data_repDF, veldf_down, veldf_up], axis=1)

    return data_repDF


def fft_function(data, fps):
    N = len(data)
    T = 1/fps
    yf = fft(data)
    yf = 2.0/N * np.abs(yf[0:N//2])
    xf = fftfreq(N, T)[:N//2]
    # plt.plot(xf[1:], yf[1:])
    return yf, xf


def fft_features(data, fps):
    """
    input: data, frame per second rate
    Perfom fft on data
    output: 3 DF (freq & mag) - frequencies with the highet magnitude
    """

    magnitude, frequency = fft_function(data, fps)
    # Reomve the DC, index 0 - frequency 0hz
    magnitude = magnitude[1:]
    frequency = frequency[1:]
    freq_num = len(frequency)
    mean_mag = mean(magnitude)
    sd_mag = stdev(magnitude)
    features = [freq_num, mean_mag, sd_mag]

    # 3 DF (with max magnitude)
    DF_ind = np.argpartition(magnitude, -3)[-3:]
    DF_ind = DF_ind[np.argsort(magnitude[DF_ind])]  # sort top 3 DF by magnitude
    DF_mag = magnitude[DF_ind]
    DF_freq = frequency[DF_ind]
    features = np.append(features, [DF_freq, DF_mag])

    # Cycle length and number of cycles, calculated by the main DF (which is the last element in the narray)
    CL = fps/DF_freq[2]  # Cycle length- seconds per cycle = 1/DF ; frames per cycle = fps*(1/DF)
    n = len(data)/CL  # number of cycles
    features = np.append(features, [CL, n])

    # df_cl_plot(data, frequency, magnitude, DF_freq, DF_mag, CL)
    return features


def fft_features_df(right_data, left_data, fps, participant, e, ds):
    right_fft = fft_features(right_data, fps)
    left_fft = fft_features(left_data, fps)

    col_name = ["freq num", "magnitude mean", "magnitude sd", "DF3_freq", "DF2_freq", "DF1_freq", "DF3_mag", "DF2_mag",
                "DF1_mag", "CL", "cycles num"]
    df = pd.DataFrame([right_fft, left_fft], columns=col_name)
    df.insert(0, "hand", ["right", "left"])
    df.insert(0, "Participant", participant)
    df.insert(0, "Exercise", e)
    df.insert(0, "Source", ds)
    return df


def feature_extraction(df, namefile):
    # Settings
    fps = 30  # frame per seconds of Nuitrack
    # For filtering
    cutoff_freq = 6
    filter_order = 2
    y, x = butter(filter_order, cutoff_freq/(fps/2))

    vel_df = pd.DataFrame()
    fft_df = pd.DataFrame()
    for i in range(0, len(df.index), 2):
        right = (df.iloc[i]).dropna().to_numpy()
        # print(right)
        name = right[0]
        ds = right[1]
        ex = right[2]
        # hand = right[3]
        right = filtfilt(y, x, right[4:])
        left = (df.iloc[i+1]).dropna().to_numpy()
        left = filtfilt(y, x, left[4:])

        right_repDF = repetition_features(right, name, 'right', fps, ds, ex)
        left_repDF = repetition_features(left, name, 'left', fps, ds, ex)
        vel_df = pd.concat([vel_df, right_repDF, left_repDF])

        fft_features_temp = fft_features_df(right, left, fps, name, ex, ds)
        fft_df = pd.concat([fft_df, fft_features_temp])

    # Combine repetitions vel and acc features
    vel_df = vel_df.dropna()  # Drop rows with NA - so aggregate values won't be affect
    col_names = vel_df.columns[8:]  # columns to aggregate mean
    # defining the columns' aggregation functions
    col_dict = {'rep': 'count'}
    for c in col_names:
        col_dict[c] = ['mean', 'std']
    vel_df_grouped = vel_df.groupby(['Source', 'Exercise', 'Participant', 'hand']).agg(col_dict).reset_index()
    vel_df_grouped.columns = vel_df_grouped.columns.map('_'.join).str.strip('_')

    features = pd.merge(vel_df_grouped, fft_df, on=["Source", "Exercise", "Participant", "hand"])
    features.to_csv('CSV\\features\\' + namefile + 'featuresbyhand.csv')


# def extraction_main(old_data_form=False, valexp=True):
#     if old_data_form:
#         path = 'CSV/Raw Data/'
#         data_files = os.listdir(path)
#         for file in data_files:
#             if 'ODS' in file:
#                 ex_name = file.split('ODS')[1].split('.csv')[0]
#                 data_source = 'ODS'
#             elif 'YDS' in file:
#                 ex_name = file.split('YDS')[1].split('.csv')[0]
#                 data_source = 'YDS'
#             else:
#                 continue
#             # elif 'naama' not in file:
#             #     ex_name = file.split('maya_')[1].split('.csv')[0]
#             #     data_source = 'maya_'
#             # else:
#             #     if 'pilot' not in file:
#             #         ex_name = file.split('naama_')[1].split('.csv')[0]
#             #         data_source = 'naama_'
#             #     elif not valexp:
#             #         ex_name = file.split('naama_pilot_')[1].split('.csv')[0]
#             #         data_source = 'naama_pilot_'
#             #     else:
#             #         ex_name = file.split('val1')[1].split('.csv')[0]
#             #         data_source = 'val1'
#
#             data_path = 'CSV/Raw Data/' + data_source + ex_name + '.csv'
#             print(data_path)
#             df = pd.read_csv(data_path)
#             feature_extraction(data_source, ex_name, df)
#     else:
#         # extract features for raw_data_scaled
#         path = 'CSV/Raw Data/all_raw_data_scaled.csv'
#         path = 'CSV/Raw Data/val1_raw_data_scaled.csv'
#         path = 'CSV/Raw Data/all_raw_data_scaled_by_local_minmax.csv'
#         path = 'CSV/Raw Data/OY_DS_raw_data_scaled.csv'
#         df = pd.read_csv(path)
#         for ex_name in df["Exercise"].unique():
#             for d in df["Source"].unique():
#                 temp_df = df[(df["Exercise"] == ex_name) & (df["Source"] == d)]
#                 feature_extraction(d, ex_name, temp_df)


def create_graph(datasource, ex_name, savefigure, showfigure, both_hands=False):
    framepersec = 30  # frame per seconds of nuitrack
    # For filtering
    cutoff_freq = 6
    filter_order = 2
    y, x = butter(filter_order, cutoff_freq/(framepersec/2))
    plt.style.use('ggplot')
    df = pd.read_csv('CSV/Raw Data/'+datasource+'_'+ex_name+'.csv')
    if both_hands:
        ids = range(0, len(df.index), 2)
    else:
        ids = range(0, len(df.index))
    for i in ids:  # each participant have 2 rows (1-right hand, 2-left hand)
        right_data = df.iloc[i]
        name = right_data[0]
        if both_hands:
            hand = ""
        else:
            hand = right_data[1]
        right_data = right_data[2:].dropna().to_numpy()
        right_data = filtfilt(y, x, right_data[2:])
        if both_hands:
            left_data = df.iloc[i+1]
            left_data = left_data[2:].dropna().to_numpy()
            left_data = filtfilt(y, x, left_data[2:])
        plt.figure()
        plt.title(name+" "+hand)
        plt.plot(right_data, label="right hand")
        if both_hands:
            plt.plot(left_data, label="left hand")
            plt.legend(loc='lower right')
        plt.xlabel("Frame")
        plt.ylabel("Angle Degree")
        if savefigure:
            plt.savefig('Plots/Signal/'+ex_name+'/'+datasource+'/'+name+hand+'.png')
        if showfigure:
            plt.show()


def signal_plots():
    path = 'CSV/Raw Data/'
    data_files = os.listdir(path)
    for file in data_files:
        print(file)
        if 'naama' not in file:
            ex_name = file.split('maya_')[1].split('.csv')[0]
            data_source = 'maya'
        else:
            if 'pilot' not in file:
                ex_name = file.split('naama_')[1].split('.csv')[0]
                data_source = 'naama'
            else:
                ex_name = file.split('naama_pilot_')[1].split('.csv')[0]
                data_source = 'naama_pilot'
        data_path = 'CSV/Raw Data/' + data_source + ex_name + '.csv'
        create_graph(data_source, ex_name, True, False, True)


def count_participants():
    path = 'CSV/Raw Data/'
    data_files = os.listdir(path)
    d = {'maya_': {}, 'naama_': {}, 'naama_pilot_':{}}
    for file in data_files:
        print(file)
        if 'naama' not in file:
            ex_name = file.split('maya_')[1].split('.csv')[0]
            data_source = 'maya_'
        else:
            if 'pilot' not in file:
                ex_name = file.split('naama_')[1].split('.csv')[0]
                data_source = 'naama_'
            else:
                ex_name = file.split('naama_pilot_')[1].split('.csv')[0]
                data_source = 'naama_pilot_'
        data_path = 'CSV/Raw Data/' + data_source + ex_name + '.csv'
        df = pd.read_csv(data_path)
        participants = df['Participant']
        for p in participants:
            name = p.split('_')[0]
            if ex_name in d[data_source]:
                if name in d[data_source][ex_name]:
                    d[data_source][ex_name][name] += 1
                else:
                    d[data_source][ex_name][name] = 1
            else:
                d[data_source][ex_name] = {}
                d[data_source][ex_name][name] = 1

    # print amount of participants from dict
    for ds in d:
        for ex in d[ds]:
            print(d[ds][ex])
            print('Amount of participants in ', ds, ' ', ex, ' is: ', len(d[ds][ex]))
            print('Amount of performances in ', ds, ' ', ex, ' is: ', sum(d[ds][ex].values())/2)


# def scale_features():
#     # TODO CHECK THE SCALING
#     # Combine all data sources and exercises files to one df
#     datasource = ['maya', 'naama', 'naama_pilot', 'val1']
#     exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows']
#
#     datasource = ['ODS', 'YDS']
#     exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows', 'open_arms_and_forward']
#
#     df = pd.DataFrame()
#     for ds in datasource:
#         for e in exercises:
#             # temp = pd.read_csv('CSV/features/'+ds+'_'+e+"featuresbyhand.csv")
#             temp = pd.read_csv('CSV/features/'+ds+e+"featuresbyhand.csv")
#             temp.insert(1, "Exercise", [e]*len(temp.index), True)
#             temp.insert(1, "Source", [ds]*len(temp.index), True)
#             df = df.append(temp, ignore_index=True)
#
#     df = df.dropna()  # data with only one detected cycle -> have null values in some features.
#     df.to_csv('CSV\\features\\ODS_YDS_allfeatures_nonscaled.csv')  # combined features file Save before scale
#     df = pd.read_csv('CSV\\features\\ODS_YDS_allfeatures_nonscaled.csv')
#
#     feature_col = 6
#     features = df.columns[feature_col:]
#
#     # standardize by 'maya' data source
#     ds_to_stand = 'YDS'
#     df_maya = df[df['Source'] == ds_to_stand]
#     # for extracting the standardize values, saves in list to means and values
#     d_standardize_values = {}
#     for e in exercises:
#         df_maya_ex = df_maya[df_maya['Exercise'] == e]
#         m_list = []
#         s_list = []
#         for f in features:
#             temp = df_maya_ex[f]
#             m = mean(temp)
#             s = np.std(temp)
#             m_list.append("%.4f" % m)
#             s_list.append("%.4f" % s)
#             df.loc[df['Exercise'] == e, f] -= m
#             df.loc[df['Exercise'] == e, f] /= s
#         d_standardize_values[e] = {}
#         d_standardize_values[e]['means'] = m_list
#         d_standardize_values[e]['std'] = s_list
#     print("standardize values:")
#     print(d_standardize_values)
#     # save dict
#     # pickle.dump(d_standardize_values, open('standardize_values_dict', 'wb'))
#
#     # save combined scaled df
#     df.to_csv('CSV\\features\\ODS_YDS_allfeatures_scaledby'+ds_to_stand+'.csv')

def scale_features(df, standby):
    feature_col = 5
    features = df.columns[feature_col:]

    exercises = df["Exercise"].unique()

    # standardize by 'ODS' data source
    df_standby = df[df['Source'] == standby]
    # for extracting the standardize values, saves in list to means and values
    d_standardize_values = {}
    for e in exercises:
        df_standby_ex = df_standby[df_standby['Exercise'] == e]
        m_list = []
        s_list = []
        for f in features:
            temp = df_standby_ex[f]
            m = mean(temp)
            s = np.std(temp)
            m_list.append("%.4f" % m)
            s_list.append("%.4f" % s)
            df.loc[df['Exercise'] == e, f] -= m
            df.loc[df['Exercise'] == e, f] /= s
        d_standardize_values[e] = {}
        d_standardize_values[e]['means'] = m_list
        d_standardize_values[e]['std'] = s_list
    print("standardize values:")
    print(d_standardize_values)
    # save dict
    # pickle.dump(d_standardize_values, open('standardize_values_dict', 'wb'))

    # save combined scaled df
    df.to_csv('CSV\\features\\ODS_YDS_allfeatures_scaledby'+standby+'.csv')


def combine():
    path = 'CSV/features/'
    data_files = os.listdir(path)
    filtered_list = [item for item in data_files if "raw_scaled" in item]
    filtered_list = ['raw_scaledbend_elbowsfeaturesbyhand.csv', 'raw_scaledraise_arms_bend_elbowsfeaturesbyhand.csv', 'raw_scaledraise_arms_horizontallyfeaturesbyhand.csv']

    ds = 'ODS_YDS_raw_scaled'
    data = pd.DataFrame()
    for e in filtered_list:
        ex_name = e.split('raw_scaled')[1].split('.csv')[0].split('featuresbyhand')[0]
        # data_path = 'CSV/Raw Data/' + ds + ex_name + '.csv'
        # df = pd.read_csv(data_path)
        # feature_extraction(ds, ex_name, df)
        temp = pd.read_csv('CSV/features/'+ds+ex_name+"featuresbyhand.csv")
        temp.insert(1, "Exercise", [ex_name]*len(temp.index), True)
        data = data.append(temp, ignore_index=True)

    data.to_csv('CSV/features/ODS_YDS_allfeatures_raw_scaled.csv')

    # add labels
    labels = pd.read_csv('CSV/features/labels.csv')
    data = pd.read_csv('CSV/features/allfeatures_raw_scaled.csv')
    result_left = pd.merge(labels,data, on=['Source', 'Exercise', 'Participant', 'hand'], how='left')
    result_left.to_csv('CSV/features/allfeatures_raw_scaled_label.csv')


if __name__ == "__main__":
    print('main')

    # path = 'CSV/Raw Data/all_raw_data_scaled.csv'
    # path = 'CSV/Raw Data/val1_raw_data_scaled.csv'
    # path = 'CSV/Raw Data/all_raw_data_scaled_by_local_minmax.csv'
    play = False
    if play:
        path = 'CSV/Raw Data/ODS_YDS_all_raw_data_scaled.csv'
        df = pd.read_csv(path)
        feature_extraction(df, 'ODS_YDS_all_raw_data_scaled')

        path = 'CSV/Raw Data/ODS_YDS_all_raw_data.csv'
        df = pd.read_csv(path)
        feature_extraction(df, 'ODS_YDS_all_raw_data')

        path = 'CSV/features/ODS_YDS_all_raw_datafeaturesbyhand.csv'
        df = pd.read_csv(path)
        scale_features(df, 'ODS')




    # extraction_main(path)
    # signal_plots()
    # count_participants()
    # scale_features()

