# from skimage.transform import resize
import datetime
import os
import re
# from sklearn.decomposition import PCA
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

directory = r"G:\2020_10_14"
txt_file = f"{directory}\\2020_10_14_swipe_n20_mm_1.txt"
mat_file = f"{directory}\\2020_10_14_swipe_n20.mat"
filename = "2020_10_14_swipe_n20"
interval = 200


# Initializing y-axis labels (only one time)
p = np.arange(1.75, -2.25, -0.25).tolist()
n = ['{:.2f}'.format(x) for x in p]

# DAtetime transformation for CSI timestamps (only one time)
time_apply = lambda t: np.datetime64(t)
vfunc = np.vectorize(time_apply)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def load_mmwave(path):
    image = []
    timestamps = []
    file_input = open(path, 'r')
    lines = file_input.readlines()
    for line in lines:
        if not line == '\n':
            timestamp = line[3:16]  # Extracting timestamp
            new_line = line[17:].split(',')  # Extracting power values.
            new_line = [int(i) for i in new_line]
            newline = np.array(new_line)
            p = newline.reshape(16, 128)  # Reshaping to 16 velocity bins and 128 range bins
            mean_row = np.mean(p[:, :], axis=1)  # Selecting only 0:105 range bins (0 -4.8m) and average
            image.append(mean_row)
            converted_timestamp = datetime.datetime.utcfromtimestamp(int(timestamp) / 1000.0).strftime(
                '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(np.datetime64(converted_timestamp))
    background = np.mean(image[:2], axis=0)
    # Background removal
    processed = []
    for line in image:
        line = line - background
        processed.append(line)
    return np.array(processed).T, timestamps

mmwave, mmwave_timestamps = load_mmwave(txt_file)

def load_original_mmwave(path):
    image = []
    timestamps = []
    file_input = open(path, 'r')
    lines = file_input.readlines()
    for line in lines:
        if not line == '\n':
            timestamp = line[3:16]  # Extracting timestamp
            new_line = line[17:].split(',')  # Extracting power values.
            new_line = [int(i) for i in new_line]
            newline = np.array(new_line)
            p = newline.reshape(16, 128)  # Reshaping to 16 velocity bins and 128 range bins
            mean_row = np.mean(p[:, :], axis=1)  # Selecting only 0:105 range bins (0 -4.8m) and average
            image.append(mean_row)
            converted_timestamp = datetime.datetime.utcfromtimestamp(int(timestamp) / 1000.0).strftime(
                '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(np.datetime64(converted_timestamp))
    background = np.mean(image[:2], axis=0)
    # Background removal
    processed = []
    for line in image:
        # line = line - background
        processed.append(line)
    return np.array(processed).T, timestamps


def show_mmWave(range_start, range_end, line_start=None, line_end=None):
    plt.tight_layout()
    p = np.arange(45, -55, -1).tolist()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plt.setp(axs, yticks=range(16), yticklabels=n, xlabel='Frame', ylabel='Velocity (m/s)')
    axs.grid(axis='x')
    axs.imshow(mmwave[:, range_start:range_end], aspect='auto', interpolation='sinc')
    if line_start and line_end:
        plt.axvline(x=line_start, color="red")  # Starting gesture index
        plt.axvline(x=line_end, color="red")  # Ending gesture index
    plt.show()


def main(directory,txt_file,mat_file,filename,interval):
    ##
    # directory = r"D:\Projects\COMP9991\16_feb_2021"
    # txt_file = f"{directory}\\2021_02_16_legswing_n10_mm_2.txt"
    # mat_file = f"{directory}\\2021_02_16_legswing_n10.mat"
    # filename = "2021_02_16_legswing_n10"
    # interval = 100

    ## todo change file
    # loc
    mmwave, mmwave_timestamps = load_mmwave(txt_file)
    csi_workspace = loadmat(mat_file)
    csi = csi_workspace['doppler_spectrum'][0]
    csi_timestamps = csi_workspace['time_matlab_string']
    csi_timestamps = vfunc(csi_timestamps)

    file_n = 1  # save file number

    range_start = 0
    range_end = range_start + interval
    max_range = mmwave.shape[1]
    if os.path.exists(f'{filename}.log'):
        # start from last end
        with open(f'{filename}.log') as f:
            file_n, range_start, *_ = f.readlines()[-1].split(',')
            file_n = int(re.search('(\d+).npz', file_n).group(1))
            range_start = int(range_start)
            range_end = range_start + interval
            print(f"Jump to {range_start}, {range_end}, File number {file_n}")

    while True:
        print(f"Current range:{range_start} - {range_end} / {max_range}")

        # todo set X-axis range
        # show the plot of frame in range of 200

        # plot , and ask user to input signal range
        show_mmWave(range_start, range_end)

        # ask  user if need to change the range

        range_confirmed = False
        while not range_confirmed:

            new_range = input("wanna change the range? input x,y :")
            if new_range == 'x':
                # there is no stuff in the vision, skip this range
                range_start += interval
                range_end += interval
                if range_end >= max_range:
                    print('Reached end. exit.')
                    sys.exit()
                continue  # todo fix here

            else:
                try:
                    # if there is empty, means dont change anything
                    if new_range == '':
                        pass

                    # if there only one number input, then consider as (0,n)
                    elif re.match("^\d+$", new_range):
                        range_start = range_start + 0
                        range_end = range_start + int(new_range)

                    # else if it is (x,y), move to range to x,y
                    elif re.match("\d+ \d+", new_range):
                        new_start, new_end = new_range.split(' ')

                        range_end = range_start + int(new_end)
                        range_start = int(range_start) + int(new_start)

                    print(f"Current range:{range_start} - {range_end}")

                    range_confirmed = True
                except ValueError:
                    print("Wrong input.")
                    continue

                show_mmWave(range_start, range_end)

        prove = False

        re_input = None
        while not prove:
            # ask user input range of mmWave
            try:
                if re_input:  # if there is re-input range from last time, then use it
                    line_start, line_end = re_input.split(' ')
                else:
                    line_start, line_end = input("input start range end range:").split(' ')
                line_start = int(line_start)
                line_end = int(line_end)

            except ValueError:
                print("Wrong input, redo")
                continue
            show_mmWave(range_start, range_end, line_start, line_end)
            prove = True

            # ask if need to adjust
            not_confirm = input("Confirm? Type anything to redo.") or None
            if not_confirm:
                prove = False
                re_input = not_confirm

        # We can then get the starting and ending timestamps of a gesture
        starting_ts = mmwave_timestamps[range_start + line_start]
        print(starting_ts)
        ending_ts = mmwave_timestamps[range_start + line_end]
        print(ending_ts)

        # find the matched ts
        #
        # auto find the threshold, and the timestamp

        start_founded = False
        end_founded = False
        for thresh in range(1, 500):
            if all([start_founded, end_founded]):
                break

            if not start_founded:
                for i in csi_timestamps:
                    if abs(starting_ts - i) < np.timedelta64(thresh, 'ms'):
                        start_founded = True
                        start_index = np.where(csi_timestamps == i)[0][0]
                        break

            if not end_founded:
                for i in csi_timestamps:
                    if abs(ending_ts - i) < np.timedelta64(thresh, 'ms'):
                        end_founded = True
                        end_index = np.where(csi_timestamps == i)[0][0]
                        break
        mmWave_index = (range_start + line_start,range_start + line_end)
        print(f"CSI index Start:{start_index}:{end_index}\n mmWave Index: {mmWave_index}")

        plt.tight_layout()
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axs.grid(False)
        axs.imshow(csi[:, 0:end_index + 1000], aspect='auto', interpolation='sinc')
        plt.axvline(x=start_index, c='red')
        plt.axvline(x=end_index, c='red')

        # Preparing for storing
        scaled_csi = scale_minmax(csi[:, start_index:end_index])
        scaled_mmwave = scale_minmax(mmwave[:, range_start + line_start:range_start + line_end])

        # todo setfilename
        save_filename = f'{filename}_{file_n}.npz'
        log_filename = f'{filename}.log'

        np.savez(save_filename, a=scaled_csi, b=scaled_mmwave)
        assert (os.path.exists(f"{save_filename}"))
        print(save_filename)
        print('________________________________________________________________')

        file_n += 1  # file number plus one
        range_start = range_end  # adjust the range
        range_end = max_range if max_range < range_end + interval else range_end + interval

        with open(log_filename, 'a+') as file:
            file.write(f'{save_filename},{range_start},{range_end},({start_index},{end_index}),{mmWave_index}\n')
    # #Use 1,2,3 .... for the file naming for each gesture put them in a directory for each gesture


def find_True_range(arr, tolerance=10) -> list:
    r = []
    start = None
    current_tolerance = 0
    for n, i in enumerate(arr):
        if i == True and start is None:
            start = n
        if not any(arr[n:n + tolerance]) and start:
            r.append((start, n))
            start = None
    return r


def remove_extreme_value(arr, gesture_len_range=(5, 25)) -> list:
    # gesture filter --> remove the gesture that in different size
    ## example: r = [[5,10], [14,19], [20,45], [50,55], [60,61]]
    ##             then we have gesture which length is [5, 5, 25, 5, 1]
    ##              since 25 and 1 are likely not a good gesture, remove them form the array
    # get rid of the extreme value

    length_arr = list(map(lambda x: x[1] - x[0], arr))
    std = np.std(length_arr)
    m = np.median(length_arr)  # we use median instead of mean

    out = []
    for n, i in enumerate(length_arr):
        if (m - 3 * std) < i < (m + 3 * std) and gesture_len_range[0] <= i <= gesture_len_range[1]:
            out.append(arr[n])
    return out




if __name__ == '__main__':



    main(directory,txt_file,mat_file,filename,interval)