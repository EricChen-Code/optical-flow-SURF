import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.fft import fft
from scipy.signal import find_peaks

def distance_to_circle(params, x, y):
    #點與圓心距離
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

def initial_guess(x, y):
    # 擬合圓的初始參數
    xc = np.mean(x)
    yc = np.mean(y)
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))
    return [xc, yc, r]

def vibration_fix(values,fix):
    fix=fix-fix[0]
    values=values-fix
    return values
    
def normalized(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
def standardized(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val
def plot_x_y(df,idex1,times_in_milliseconds,x_values,y_values,frame_rate,axes,normalize,standardize,fft_bool):
    axes[idex1, 0].scatter(x_values, y_values, label=f'{stat}B{blade_number}P {point_index}{fix}', s=1)
    axes[idex1, 0].set_title(f'X,Y Coordinate of {title_pra}')
    axes[idex1, 0].set_xlabel('X Coordinat')
    axes[idex1, 0].set_ylabel('Y Coordinate')
    axes[idex1, 0].legend()
    axes[idex1, 0].grid(True)
    x=times_in_milliseconds
    xlabel='Time (ms)'
    ylabel_2='X Coordinate(pixel)'
    ylabel_3='Y Coordinate(pixel)'
    if normalize== True:
        y1=normalized(x_values)
        y2=normalized(y_values)
        ylabel_2='X Coordinate'
        ylabel_3='Y Coordinate'
    elif standardize== True:
        y1=standardized(x_values)
        y2=standardized(y_values)
        ylabel_2='X Coordinate'
        ylabel_3='Y Coordinate'
    elif fft_bool== True:
        T = 1 / frame_rate
        N = len(x_values)
        x= np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        y1 =2.0 / N * np.abs(fft(x_values)[:N//2]) 
        y2 =2.0 / N * np.abs(fft(y_values)[:N//2])
        xlabel='Frequency (Hz)'
        ylabel_3='Amplitude'
        ylabel_2='Amplitude'
    else:
        y1=x_values
        y2=y_values
    axes[idex1, 1].plot(x, y1, label=f'{stat}B{blade_number}P{point_index}{fix}')
    axes[idex1, 1].set_title(f'X Coordinate of {title_pra}')
    axes[idex1, 1].set_xlabel(xlabel)
    axes[idex1, 1].set_ylabel(ylabel_2)
    axes[idex1, 1].legend()
    axes[idex1, 1].grid(True)
    if fft_bool== True:
        peaks, _ = find_peaks((y1), 20)
        axes[idex1, 1].plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
        for peak in peaks:
            axes[idex1, 1].annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))
    axes[idex1, 2].plot(x, y2, label=f'{stat}B{blade_number}P {point_index}{fix}')
    axes[idex1, 2].set_title(f'Y Coordinate of {title_pra}')
    axes[idex1, 2].set_xlabel(xlabel)
    axes[idex1, 2].set_ylabel(ylabel_3)
    axes[idex1, 2].legend()
    axes[idex1, 2].grid(True)
    if fft_bool== True:
        peaks, _ = find_peaks((y2), 20)
        axes[idex1, 2].plot(x[peaks], y2[peaks], "x", label='_nolegend_', color='red')
        for peak in peaks:
            axes[idex1, 2].annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y2[peak]), xytext=(x[peak], y2[peak]+0.1))

def plots_type(df,idex1,times_in_milliseconds,x_values,y_values,centers,frame_rate,plot_type,axes,normalize,standardize,fft_bool):
    if plot_type == "1":
        plot_x_y(df,idex1,times_in_milliseconds,x_values,y_values,frame_rate,axes,normalize,standardize,fft_bool)

def circle_centers(mode):
    global all_centers,average_centers,all_centers_center,average_centers_center
    all_centers = {}
    all_centers_center={}
    centers = []
    centers_center=[]
    average_centers=[]
    average_centers_center=[]
    if mode == "1":#同一文件上的所有點為一組
        for idx, item in enumerate(files_and_points):
            df = pd.read_csv(f'{file_path}{item["file"]}', delimiter=',')
            df_fix = pd.read_csv('C:/Users/user/Desktop/Python/csv/1_1_1_2.csv', delimiter=',')
            for point in item['points']:
            # 提取每個點的y座標
                x_values = df[df['Point'] == point]['X Coordinate']
                y_values = 1079-df[df['Point'] == point]['Y Coordinate']                       
                x_center = df[df['Point'] == point]['X center']
                y_center = 1079-df[df['Point'] == point]['Y center']
                initial = initial_guess(x_values, y_values)
                result = least_squares(distance_to_circle, initial, args=(x_values, y_values))
                xc, yc, r = result.x
                print(f'{item["file"]} - Point {point},Center: ({xc:.2f},{yc:.2f}),Radius: {r:.2f}')
                if item["file"] not in all_centers:
                    all_centers[item["file"]] = {}
                all_centers[item["file"]][point] = (xc, yc)
                # 添加圓心座標到數列
                centers.append((xc, yc))
                #for center
                initial = initial_guess(x_center, y_center)
                result = least_squares(distance_to_circle, initial, args=(x_center, y_center))
                x_c, y_c, r_ = result.x
                print(f'{item["file"]}(fixed)- Point {point},Center: ({x_c:.2f},{y_c:.2f}),Radius: {r_:.2f}')
                if item["file"] not in all_centers_center:
                    all_centers_center[item["file"]] = {}
                all_centers_center[item["file"]][point] = (x_c, y_c)
                # 添加圓心座標到數列
                centers_center.append((x_c, y_c))
            average_centers = np.mean(centers, axis=0)
            print(average_centers)
            average_centers_center = np.mean(centers_center, axis=0)
            print(average_centers_center)
        pass
    elif mode == "2":#不同文上第i個點為一組
        for point_index in range(len(files_and_points[0]['points'])):
              # 在每個大迴圈中重置圓心列表
            for item in files_and_points:
                point = item['points'][point_index]
                df = pd.read_csv(f'{file_path}{item["file"]}', delimiter=',')
                x_values = df[df['Point'] == point]['X Coordinate']
                y_values = 1079-df[df['Point'] == point]['Y Coordinate']
                x_center = df[df['Point'] == point]['X center']
                y_center = 1079-df[df['Point'] == point]['Y center']
                initial = initial_guess(x_values, y_values)
                result = least_squares(distance_to_circle, initial, args=(x_values, y_values))
                xc, yc, r = result.x
                print(f'{item["file"]}- Point {point},Center: ({xc:.2f},{yc:.2f}),Radius: {r:.2f}')
                if item["file"] not in all_centers:
                    all_centers[item["file"]] = {}
                all_centers[item["file"]][point] = (xc, yc)
                # 添加圓心座標到數列
                centers.append((xc, yc))
                #for center
                initial = initial_guess(x_center, y_center)
                result = least_squares(distance_to_circle, initial, args=(x_center, y_center))
                x_c, y_c, r_ = result.x
                print(f'{item["file"]}(fixed)- Point {point},Center: ({x_c:.2f},{y_c:.2f}),Radius: {r_:.2f}')
                if item["file"] not in all_centers_center:
                    all_centers_center[item["file"]] = {}
                all_centers_center[item["file"]][point] = (x_c, y_c)
                # 添加圓心座標到數列
                centers_center.append((x_c, y_c))
            average_center = np.mean(centers, axis=0)
            print(average_center)
            average_centers_center = np.mean(centers_center, axis=0)
            print(average_centers_center)
        pass
# 定義圖表分組邏輯
def point_group(mode,plot_type,normalize,standardize,fft_bool):
    global nrows,ncols,item,point,point_index,frame_rate,stat,fan_speed,blade_number,video_number,fix,title_pra,radius
    if mode == "1":#同一文件上的所有點為一組
        nrows=len(files_and_points)
        ncols=1
        if plot_type == "1"or plot_type == '5':
            if len(files_and_points)>1:
                fig, axes = plt.subplots(nrows,3, figsize=(16, 9))
            else :
                fig, axes = plt.subplots(2,3, figsize=(16, 9))
        else :
            if len(files_and_points)>1:
                fig, axes = plt.subplots(nrows, figsize=(16, 9))  
            else:
                fig, axes = plt.subplots(2, figsize=(16, 9))
        for idx, item in enumerate(files_and_points):
            df = pd.read_csv(f'{file_path}{item["file"]}', delimiter=',')
            df_fix = pd.read_csv('C:/Users/user/Desktop/Python/csv/1_1_1_2.csv', delimiter=',')
            
            filename= item["file"].split(".")[0]
            # 使用 split() 方法分割文件名
            parts = filename.split("_")
            # 提取所需的信息
            situation = int(parts[0])
            fan_speed = int(parts[1])
            blade_number = int(parts[2])
            video_number = int(parts[3])
            if situation==1:
                stat='Normal'
            elif situation==2:
                stat='1 lighter blade '
            elif situation==3:
                stat='2 Blade'
            title_pra=stat
            for point_index, point in enumerate(item['points']):
            # 提取每個點的y座標
                frame_rate = df.iloc[1]['fps']
                x_values = df[df['Point'] == point]['X Coordinate']
                y_values = 1079-df[df['Point'] == point]['Y Coordinate']
                x_values_series = pd.Series(x_values)
                y_values_series = pd.Series(y_values)
                x_values = x_values_series.reset_index(drop=True)
                y_values = y_values_series.reset_index(drop=True)
                x_fix = df[df['Point'] == 1]['X Coordinate']
                y_fix = 1079-df[df['Point'] == 1]['Y Coordinate'] 
                x_fix_series = pd.Series(x_fix)
                y_fix_series = pd.Series(y_fix)
                x_fix = x_fix_series.reset_index(drop=True)
                y_fix = y_fix_series.reset_index(drop=True)
                x_values = vibration_fix(x_values,x_fix)
                y_values = vibration_fix(y_values,y_fix)

                print(f"Original x_values length: {len(x_values)}, y_values length: {len(y_values)}")

                radius = df[df['Point'] == point]['Radius']
                fix=''
                frames_per_millisecond = frame_rate / 1000  # 每毫秒的帧数
                frame_numbers = df[df['Point'] == item['points'][0]]['Frame']
                times_in_milliseconds = frame_numbers / frames_per_millisecond
                
                plots_type(df,idx,times_in_milliseconds,x_values,y_values,all_centers[item["file"]][point],frame_rate,plot_type,axes,normalize,standardize,fft_bool)
                
                x_center = df[df['Point'] == point]['X center']
                y_center = 1079-df[df['Point'] == point]['Y center']
                x_center_series = pd.Series(x_center)
                y_center_series = pd.Series(y_center)
                x_center = x_center_series.reset_index(drop=True)
                y_center = y_center_series.reset_index(drop=True)
                x_fix1 = df[df['Point'] == 1]['X center']
                y_fix1 = 1079-df[df['Point'] == 1]['Y center']
                x_fix1_series = pd.Series(x_fix1)
                y_fix1_series = pd.Series(y_fix1)
                x_fix1 = x_fix1_series.reset_index(drop=True)
                y_fix1 = y_fix1_series.reset_index(drop=True)
                x_center = vibration_fix(x_center,x_fix1)
                y_center = vibration_fix(y_center,y_fix1)
                fix='(fixed)'
                plots_type(df,idx,times_in_milliseconds,x_center, y_center, all_centers_center[item["file"]][point], frame_rate, plot_type,axes,normalize,standardize,fft_bool)
        plt.tight_layout()
        plt.show()
        pass
    elif mode == "2":#不同文上第i個點為一組
        nrows=len(files_and_points[0]['points'])
        ncols=1
        if plot_type == "1"or plot_type == 5:
            if nrows>1:
                fig, axes = plt.subplots(nrows,3, figsize=(16, 9))
            else:
                fig, axes = plt.subplots(2,3, figsize=(16, 9))
        else :
            if nrows>1:
                fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9))
            else:
                fig, axes = plt.subplots(2,3, figsize=(16, 9))
        for point_index in range(len(files_and_points[0]['points'])):
            title_pra='Point'+f'{point_index}'
            for item_index, item in enumerate(files_and_points):
                filename= item["file"].split(".")[0]
                # 使用 split() 方法分割文件名
                parts = filename.split("_")
                # 提取所需的信息
                situation = int(parts[0])
                fan_speed = int(parts[1])
                blade_number = int(parts[2])
                video_number = int(parts[3])
                if situation==1:
                    stat='Normal'
                elif situation==2:
                    stat='1 lighter blade '
                elif situation==3:
                    stat='2 Blade'
                point = item['points'][point_index]
                df = pd.read_csv(f'{file_path}{item["file"]}', delimiter=',')
                frame_rate = df.iloc[1]['fps']
                x_values = df[df['Point'] == point]['X Coordinate']
                y_values = 1079-df[df['Point'] == point]['Y Coordinate']
                radius = df[df['Point'] == point]['Radius']
                fix=''
                frames_per_millisecond = frame_rate / 1000  # 每毫秒的帧数
                frame_numbers = df[df['Point'] == item['points'][0]]['Frame']
                times_in_milliseconds = frame_numbers / frames_per_millisecond
                plots_type(df, point_index,times_in_milliseconds,x_values, y_values, all_centers[item["file"]][point], frame_rate, plot_type,axes,normalize,standardize,fft_bool)
                x_center = df[df['Point'] == point]['X center']
                y_center = 1079-df[df['Point'] == point]['Y center']
                fix='(fixed)'
                plots_type(df, point_index,times_in_milliseconds,x_center, y_center, all_centers_center[item["file"]][point], frame_rate, plot_type,axes,normalize,standardize,fft_bool)
        plt.tight_layout()
        plt.show()
        pass
    else:
        print("錯誤的資料分組模式選擇。")
def main():
    global files_and_points,file_path,colors,first_points
    colors = plt.cm.viridis(np.linspace(0, 1, 50))
    file_path='C:/Users/user/Desktop/Python/csv/'
    files_and_points = [
    {'file': '1_1_1_1.csv', 'points': [2,3]},
    {'file': '1_1_1_1.csv', 'points': [2,3]}
    ]
    first_points =(1,1,1)
    mode = input("請選擇分組模式：\n1. 同一文件上的所有點為一組\n2. 不同文上第i個點為一組\n請選擇分組模式：")
    plot_type = input("請選擇圖表的內容：\n1. x、y座標\n")
    normalize = False
    standardize = False
    fft_bool = False
    # 循環詢問是否需要資料處理
    while True:
        option = input("要做的處理：\n1. normalize\n2.standardize\n3.FFT\n選擇：")
        if option == '1':
            normalize = True
        elif option == '2':
            standardize = True
        elif option == '3':
            fft_bool = True
        else:
            break
    circle_centers(mode)

    point_group(mode,plot_type,normalize,standardize,fft_bool)     
if __name__ == "__main__":
    main()
