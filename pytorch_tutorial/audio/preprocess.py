# Uncomment the following line to run in Google Colab
# !pip install torchaudio
import torch
import torchaudio
import requests
import matplotlib.pyplot as plt

'''
报错 RuntimeError: No audio I/O backend is available.
安装 pip install SoundFile

音频采样，是把声音从模拟信号转换为数字信号。采样率，就是每秒对声音进行采集的次数，同样也是所得的数字信号的每秒样本数
采样越高，声音的还原就越真实越自然，人对频率的识别范围是 20HZ - 20000HZ, 如果每秒钟能对声音做 20000 个采样, 回放时就足可以满足人耳的需求. 
所以 22050 的采样频率是常用的, 44100已是CD音质, 超过48000的采样对人耳已经没有意义。
'''
url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
r = requests.get(url)

with open('steam-train-whistle-daniel_simon-converted-from-mp3.wav', 'wb') as f:
    f.write(r.content)

filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename) # waveform 音频大小 sample_rate 采样率

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

# plt.figure()
# plt.plot(waveform.t().numpy())
# plt.show()

# 对某一个波段重新采样
new_sample_rate = sample_rate/10
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))
print("Shape of transformed waveform: {}".format(transformed.size()))
# plt.figure()
# plt.plot(transformed[0,:].numpy())
# plt.show()

print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))
