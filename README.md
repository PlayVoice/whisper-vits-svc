# SoftVC VITS Singing Voice Conversion
## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，通过icassp2022_vocal_transcription项目提取音频midi note，将两者结合输入VITS替换原本的文本输入达到歌声转换的效果。

## 实现细节
+ midi note（0-127 LongTensor）通过pitch_embedding后与soft-units相加替代vits原本的文本输入
+ 采用了VISinger中的PitchPredictor，引入了PitchLoss修正音高

ps：目前效果不是很好
