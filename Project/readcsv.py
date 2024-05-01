import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

df = pd.read_csv("csvdata/1/run-1-tag-loss.csv")
df["Step"] /= 3
step = df["Step"].values.tolist()
info = df["Value"].values.tolist()
# df1 = pd.read_csv("csvdata/yolov3/run-.-tag-re_3.csv")
# step1 = df1["Step"].values.tolist()
# info1 = df1["Value"].values.tolist()
# df2 = pd.read_csv("csvdata/yolov3_tiny/run-.-tag-re_2.csv")
# step2 = df2["Step"].values.tolist()
# info2 = df2["Value"].values.tolist()
# df3 = pd.read_csv("csvdata/yolov4_tiny/run-.-tag-re_2.csv")
# step3 = df3["Step"].values.tolist()
# info3 = df3["Value"].values.tolist()
y_smoothed = gaussian_filter1d(info, sigma=10)
# y_smoothed1 = gaussian_filter1d(info1, sigma=10)
# y_smoothed2 = gaussian_filter1d(info2, sigma=10)
# y_smoothed3 = gaussian_filter1d(info3, sigma=10)
plt.plot(step, y_smoothed, 'tab:green',label='yolov4')
# plt.plot(step1, y_smoothed1,'tab:red',label='yolov3')
# plt.plot(step2, y_smoothed2,'tab:gray',label='yolov3_tiny')
# plt.plot(step3, y_smoothed3,'tab:blue',label='yolov4_tiny')
plt.xlabel('Step(batch)')
plt.ylabel('loss')
plt.legend()
plt.title('Total loss in this model')
plt.ylim(0, 5)
plt.show()