import os
import subprocess
import yaml
import sys
import webbrowser
import gradio as gr
from ruamel.yaml import YAML
import shutil
import soundfile
import shlex

class WebUI:
    def __init__(self):
        self.train_config_path = 'configs/train.yaml'
        self.info = Info()
        self.names = []
        self.names2 = []
        self.voice_names = []
        base_config_path = 'configs/base.yaml'
        if not os.path.exists(self.train_config_path):
            shutil.copyfile(base_config_path, self.train_config_path)
            print("初始化成功")
        else:
            print("就绪")
        self.main_ui()

    def main_ui(self):
        with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.green)) as ui:
            gr.Markdown('# so-vits-svc5.0 WebUI')

            with gr.Tab("预处理-训练"):
                with gr.Accordion('训练说明', open=False):
                    gr.Markdown(self.info.train)
                gr.Markdown('### 预处理参数设置')

                with gr.Row():

                    self.model_name = gr.Textbox(value='sovits5.0', label='model', info='模型名称', interactive=True)

                    self.f0_extractor = gr.Dropdown(choices=['dio','crepe'], value='dio', label='f0_extractor', info='f0提取器', interactive=True)

                    self.thread_count = gr.Slider(minimum=1, maximum=os.cpu_count(), step=1, value=2, label='thread_count', info='预处理线程数', interactive=True)

                gr.Markdown('### 训练参数设置')

                with gr.Row():

                    self.learning_rate = gr.Number(value=5e-5, label='learning_rate', info='学习率', interactive=True)

                    self.batch_size = gr.Slider(minimum=1, maximum=50, step=1, value=6, label='batch_size', info='批大小', interactive=True)

                with gr.Row():

                    self.info_interval = gr.Number(value=50, label='info_interval', info='训练日志记录间隔（step）', interactive=True)

                    self.eval_interval = gr.Number(value=1, label='eval_interval', info='验证集验证间隔（epoch）', interactive=True)

                    self.save_interval = gr.Number(value=5, label='save_interval', info='检查点保存间隔（epoch）', interactive=True)

                    self.keep_ckpts = gr.Number(value=0, label='keep_ckpts', info='保留最新的检查点文件(0保存全部)',interactive=True)

                with gr.Row():
                    self.slow_model = gr.Checkbox(label="Whether to add a base mold", info='是否添加底模', value=True, interactive=True)

                gr.Markdown('### 开始训练')

                with gr.Row():

                    self.bt_open_dataset_folder = gr.Button(value='打开数据集文件夹')

                    self.bt_onekey_train = gr.Button('一键训练', variant="primary")

                    self.bt_tb = gr.Button('启动Tensorboard', variant="primary")

                gr.Markdown('### 恢复训练')

                with gr.Row():

                    self.resume_model = gr.Dropdown(choices=sorted(self.names), label='Resume training progress from checkpoints', info='从检查点恢复训练进度', interactive=True)

                    with gr.Column():

                        self.bt_refersh = gr.Button('刷新')

                        self.bt_resume_train = gr.Button('恢复训练', variant="primary")

            with gr.Tab("推理"):

                with gr.Accordion('推理说明', open=False):
                    gr.Markdown(self.info.inference)

                gr.Markdown('### 推理参数设置')


                with gr.Row():
                    with gr.Column():

                        with open("svc_out.wav", "wb") as f:
                            pass
                        self.keychange = gr.Slider(-24, 24, value=0, step=1, label='变调')

                        self.file_list = gr.Markdown(value="", label="文件列表")



                        with gr.Row():

                            self.resume_model2 = gr.Dropdown(choices=sorted(self.names2), label='Select the model you want to export',
                                                             info='选择要导出的模型', interactive=True)
                            with gr.Column():

                                self.bt_refersh2 = gr.Button(value='刷新模型和音色')


                                self.bt_out_model = gr.Button(value='导出模型', variant="primary")

                        with gr.Row():

                            self.resume_voice = gr.Dropdown(choices=sorted(self.voice_names), label='Select the sound file',
                                                            info='选择音色文件', interactive=True)

                        with gr.Row():

                            self.input_wav = gr.Audio(type='filepath', label='选择待转换音频', source='upload',max_size=114514)

                        with gr.Row():

                            self.bt_infer = gr.Button(value='开始转换', variant="primary")

                        with gr.Row():

                            self.output_wav = gr.Audio(label='输出音频', interactive=False)

            self.bt_open_dataset_folder.click(fn=self.openfolder)
            self.bt_onekey_train.click(fn=self.onekey_training,inputs=[self.model_name, self.f0_extractor, self.thread_count,self.learning_rate,self.batch_size, self.info_interval, self.eval_interval,self.save_interval, self.keep_ckpts, self.slow_model])
            self.bt_out_model.click(fn=self.out_model, inputs=[self.model_name, self.resume_model2])
            self.bt_tb.click(fn=self.tensorboard)
            self.bt_refersh.click(fn=self.refresh_model, inputs=[self.model_name], outputs=[self.resume_model])
            self.bt_resume_train.click(fn=self.resume_train, inputs=[self.model_name, self.resume_model])
            self.bt_infer.click(fn=self.inference, inputs=[self.input_wav, self.resume_voice, self.keychange], outputs=[self.output_wav])
            self.bt_refersh2.click(fn=self.refresh_model_and_voice, inputs=[self.model_name],outputs=[self.resume_model2, self.resume_voice])

        ui.launch(inbrowser=True, server_port=2333, share=True)
    def openfolder(self):
        try:
            if sys.platform.startswith('win'):
                os.startfile('dataset_raw')
            elif sys.platform.startswith('linux'):
                subprocess.call(['xdg-open', 'dataset_raw'])
            elif sys.platform.startswith('darwin'):
                subprocess.call(['open', 'dataset_raw'])
            else:
                print('打开文件夹失败！')
        except BaseException:
            print('打开文件夹失败！')



    def preprocessing(self, f0_extractor, thread_count):
        print('开始预处理')
        if str(f0_extractor) == 'crepe':
            train_process = subprocess.Popen('python -u svc_preprocessing.py --crepe -t ' + str(thread_count), stdout=subprocess.PIPE)
        else:
            train_process = subprocess.Popen('python -u svc_preprocessing.py -t ' + str(thread_count), stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def create_config(self, model_name, learning_rate, batch_size, info_interval, eval_interval, save_interval,
                      keep_ckpts, slow_model):
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.width = 1024
        with open("configs/train.yaml", "r") as f:
            config = yaml.load(f)
        config['train']['model'] = model_name
        config['train']['learning_rate'] = learning_rate
        config['train']['batch_size'] = batch_size
        config["log"]["info_interval"] = int(info_interval)
        config["log"]["eval_interval"] = int(eval_interval)
        config["log"]["save_interval"] = int(save_interval)
        config["log"]["keep_ckpts"] = int(keep_ckpts)
        if slow_model:
            config["train"]["pretrain"] = "sovits5.0_bigvgan_mix_v2.pth"
        else:
            config["train"]["pretrain"] = ""
        with open("configs/train.yaml", "w") as f:
            yaml.dump(config, f)
        return f"已将log参数更新为{config['log']}"

    def training(self, model_name):
        print('开始训练')
        train_process = subprocess.Popen('python -u svc_trainer.py -c ' + self.train_config_path + ' -n ' + str(model_name), stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def onekey_training(self, model_name, f0_extractor, thread_count, learning_rate, batch_size, info_interval, eval_interval, save_interval, keep_ckpts, slow_model):
        print(self, model_name, f0_extractor, thread_count, learning_rate, batch_size, info_interval, eval_interval,
              save_interval, keep_ckpts)
        self.create_config(model_name, learning_rate, batch_size, info_interval, eval_interval, save_interval, keep_ckpts, slow_model)
        self.preprocessing(f0_extractor, thread_count)
        self.training(model_name)

    def out_model(self, model_name, resume_model2):
        print('导出模型')
        try:
            subprocess.Popen('python -u svc_export.py -c {} -p "chkpt/{}/{}"'.format(self.train_config_path, model_name, resume_model2),stdout=subprocess.PIPE)
            print('导出模型成功')
        except Exception as e:
            print("出现错误：", e)


    def tensorboard(self):
        if sys.platform.startswith('win'):
            tb_process = subprocess.Popen('tensorboard --logdir=logs --port=6006', stdout=subprocess.PIPE)
            webbrowser.open("http://localhost:6006")
        else:
            p1 = subprocess.Popen(["ps", "-ef"], stdout=subprocess.PIPE) #ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
            p2 = subprocess.Popen(["grep", "tensorboard"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p3 = subprocess.Popen(["awk", "{print $2}"], stdin=p2.stdout, stdout=subprocess.PIPE)
            p4 = subprocess.Popen(["xargs", "kill", "-9"], stdin=p3.stdout)
            p1.stdout.close()
            p2.stdout.close()
            p3.stdout.close()
            p4.communicate()
            tb_process = subprocess.Popen('tensorboard --logdir=logs --port=6007', stdout=subprocess.PIPE)  # AutoDL端口设置为6007
        while tb_process.poll() is None:
            output = tb_process.stdout.readline().decode('utf-8')
            print(output)

    def refresh_model(self, model_name):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, f"chkpt/{model_name}")
        self.names = []
        try:
            for self.name in os.listdir(self.model_root):
                if self.name.endswith(".pt"):
                    self.names.append(self.name)
            return {"choices": sorted(self.names), "__type__": "update"}
        except FileNotFoundError:
            return {"label": "缺少模型文件", "__type__": "update"}

    def refresh_model2(self, model_name):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, f"chkpt/{model_name}")
        self.names2 = []
        try:
            for self.name in os.listdir(self.model_root):
                if self.name.endswith(".pt"):
                    self.names2.append(self.name)
            return {"choices": sorted(self.names2), "__type__": "update"}
        except FileNotFoundError as e:
            return {"label": "缺少模型文件", "__type__": "update"}

    def refresh_voice(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, "data_svc/singer")
        self.voice_names = []
        for self.name in os.listdir(self.model_root):
            if self.name.endswith(".npy"):
                self.voice_names.append(self.name)
        return {"choices": sorted(self.voice_names), "__type__": "update"}

    def refresh_model_and_voice(self, model_name):
        model_update = self.refresh_model2(model_name)
        voice_update = self.refresh_voice()
        return model_update, voice_update

    def resume_train(self, model_name, resume_model):
        print('恢复训练')
        train_process = subprocess.Popen('python -u svc_trainer.py -c {} -n {} -p "chkpt/{}/{}"'.format(self.train_config_path, model_name, model_name, resume_model), stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def inference(self, input, resume_voice, keychange):
        if os.path.exists("test.wav"):
            os.remove("test.wav")
            print("已清理残留文件")
        else:
            print("无需清理残留文件")
        self.train_config_path = 'configs/train.yaml'
        print('开始推理')
        shutil.copy(input, ".")
        input_name = os.path.basename(input)
        os.rename(input_name, "test.wav")
        input_name = "test.wav"
        if not input_name.endswith(".wav"):
            data, samplerate = soundfile.read(input_name)
            input_name = input_name.rsplit(".", 1)[0] + ".wav"
            soundfile.write(input_name, data, samplerate)
        train_config_path = shlex.quote(self.train_config_path)
        keychange = shlex.quote(str(keychange))
        cmd = ["python", "-u", "svc_inference.py", "--config", train_config_path, "--model", "sovits5.0.pth", "--spk",
               f"data_svc/singer/{resume_voice}", "--wave", "test.wav", "--shift", keychange]
        train_process = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        print(train_process.stdout)
        print(train_process.stderr)
        print("推理成功")
        return "svc_out.wav"


class Info:
    def __init__(self) -> None:
        self.train = '''
### 2023.7.11\n
@OOPPEENN(https://github.com/OOPPEENN)第一次编写\n
@thestmitsuk(https://github.com/thestmitsuki)二次补完\n
@OOPPEENN(https://github.com/OOPPEENN)is written for the first time\n
@thestmitsuki(https://github.com/thestmitsuki)Secondary completion

        '''
        self.inference = '''
### 2023.7.11\n
@OOPPEENN(https://github.com/OOPPEENN)第一次编写\n
@thestmitsuk(https://github.com/thestmitsuki)二次补完\n
@OOPPEENN(https://github.com/OOPPEENN)is written for the first time\n
@thestmitsuki(https://github.com/thestmitsuki)Secondary completion

        '''


webui = WebUI()

