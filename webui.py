import os
import subprocess
import yaml
import sys
import platform
import webbrowser
import gradio as gr

class WebUI:
    def __init__(self):
        self.names = []
        self.info = Info()
        self.train_config_path = 'configs/train.yaml'
        self.main_ui()

    def main_ui(self):
        with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.green)) as ui:
            gr.Markdown('# so-vits-svc5.0 WebUI')

            with gr.Tab("训练"):
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
                gr.Markdown('### 开始训练')
                with gr.Row():
                    self.bt_open_dataset_folder = gr.Button('打开数据集文件夹')
                    self.bt_onekey_train = gr.Button('一键训练', variant="primary")
                    self.bt_tb = gr.Button('启动Tensorboard', variant="primary")
                gr.Markdown('### 恢复训练')
                with gr.Row():
                    self.resume_model = gr.Dropdown(choices=sorted(self.names), label='resume_model', info='从检查点恢复训练进度', interactive=True)
                    with gr.Column():
                        self.bt_refersh = gr.Button('刷新')
                        self.bt_resume_train = gr.Button('恢复训练', variant="primary")

            with gr.Tab('推理 (施工中，不可用) '):
                with gr.Accordion('推理说明', open=False):
                    gr.Markdown(self.info.inference)
                gr.Markdown('### 一键推理（不需要手动修改f0）')
                with gr.Row():
                    with gr.Column():
                        #self.choose_model = gr.Dropdown(label='模型文件', choices=sorted(model_file))
                        #self.choose_voice = gr.Dropdown(label='音色文件', choices=sorted(voice_file))
                        self.keychange = gr.Slider(-24, 24, value=0, step=1, label='变调')
                        self.bt_refresh = gr.Button(value='刷新模型和音色')
                        self.bt_infer = gr.Button(value='开始转换', variant="primary")
                    with gr.Column():
                        self.input_wav = gr.Audio(type='filepath', label='选择待转换音频')
                        self.output_wav = gr.Audio(type='filepath', label='输出音频')

            self.bt_open_dataset_folder.click(fn=self.openfolder)
            self.bt_onekey_train.click(fn=self.onekey_training, inputs=[self.model_name, self.f0_extractor, self.thread_count, self.learning_rate,
                                          self.batch_size, self.info_interval, self.eval_interval, self.save_interval])
            self.bt_tb.click(fn=self.tensorboard)
            self.bt_refersh.click(fn=self.refresh_model, inputs=[self.model_name], outputs=[self.resume_model])
            self.bt_resume_train.click(fn=self.resume_train, inputs=[self.model_name, self.resume_model])
            #self.bt_infer.click(fn=self.inference, inputs=[self.input_wav, self.choose_model, self.keychange, self.id], outputs=self.output_wav)
        ui.launch(inbrowser=True, server_port=2333, share=True)

    def openfolder(self):
        try:
            os.startfile('dataset_raw')
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

    def create_config(self, model_name, learning_rate, batch_size, info_interval, eval_interval, save_interval):
        with open('configs/base.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg['train']['model'] = str(model_name)
        cfg['train']['learning_rate'] = float(learning_rate)
        cfg['train']['batch_size'] = int(batch_size)
        cfg['log']['info_interval'] = int(info_interval)
        cfg['log']['eval_interval'] = str(eval_interval)
        cfg['log']['save_interval'] = str(save_interval)
        print('配置文件信息：' + str(cfg))
        with open(self.train_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f)

    def training(self, model_name):
        print('开始训练')
        train_process = subprocess.Popen('python -u svc_trainer.py -c ' + self.train_config_path + ' -n ' + str(model_name), stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def onekey_training(self, model_name, f0_extractor, thread_count, learning_rate, batch_size, info_interval, eval_interval, save_interval):
        self.create_config(model_name, learning_rate, batch_size, info_interval, eval_interval, save_interval)
        self.preprocessing(f0_extractor, thread_count)
        self.training(model_name)

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
        for self.name in os.listdir(self.model_root):
            if self.name.endswith(".pt"):
                self.names.append(self.name)
        return {"choices": sorted(self.names), "__type__": "update"}

    def resume_train(self, model_name, resume_model):
        print('恢复训练')
        train_process = subprocess.Popen('python -u svc_trainer.py -c {} -n {} -p "chkpt/{}/{}"'.format(self.train_config_path, model_name, model_name, resume_model), stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def inference():
        pass


class Info:
    def __init__(self) -> None:
        self.train = '''
### 待补充
        '''
        self.inference = '''
### 待补充
        '''


webui = WebUI()
