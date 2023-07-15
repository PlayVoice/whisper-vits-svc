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
import locale

class WebUI:
    def __init__(self):
        self.train_config_path = 'configs/train.yaml'
        self.info = Info()
        self.names = []
        self.names2 = []
        self.voice_names = []
        self.base_config_path = 'configs/base.yaml'
        if not os.path.exists(self.train_config_path):
            shutil.copyfile(self.base_config_path, self.train_config_path)
            print(i18n("初始化成功"))
        else:
            print(i18n("就绪"))
        self.main_ui()

    def main_ui(self):
        with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.green)) as ui:

            gr.Markdown('# so-vits-svc5.0 WebUI')

            with gr.Tab(i18n("预处理-训练")):

                with gr.Accordion(i18n('训练说明'), open=False):

                    gr.Markdown(self.info.train)

                gr.Markdown(i18n('### 预处理参数设置'))

                with gr.Row():

                    self.model_name = gr.Textbox(value='sovits5.0', label='model', info=i18n('模型名称'), interactive=True) #建议设置为不可修改

                    self.f0_extractor = gr.Textbox(value='crepe', label='f0_extractor', info=i18n('f0提取器'), interactive=False)

                    self.thread_count = gr.Slider(minimum=1, maximum=os.cpu_count(), step=1, value=2, label='thread_count', info=i18n('预处理线程数'), interactive=True)

                gr.Markdown(i18n('### 训练参数设置'))

                with gr.Row():

                    self.learning_rate = gr.Number(value=5e-5, label='learning_rate', info=i18n('学习率'), interactive=True)

                    self.batch_size = gr.Slider(minimum=1, maximum=50, step=1, value=6, label='batch_size', info=i18n('批大小'), interactive=True)

                with gr.Row():

                    self.info_interval = gr.Number(value=50, label='info_interval', info=i18n('训练日志记录间隔（step）'), interactive=True)

                    self.eval_interval = gr.Number(value=1, label='eval_interval', info=i18n('验证集验证间隔（epoch）'), interactive=True)

                    self.save_interval = gr.Number(value=5, label='save_interval', info=i18n('检查点保存间隔（epoch）'), interactive=True)

                    self.keep_ckpts = gr.Number(value=0, label='keep_ckpts', info=i18n('保留最新的检查点文件(0保存全部)'),interactive=True)

                with gr.Row():

                    self.slow_model = gr.Checkbox(label=i18n("是否添加底模"), value=True, interactive=True)

                gr.Markdown(i18n('### 开始训练'))

                with gr.Row():

                    self.bt_open_dataset_folder = gr.Button(value=i18n('打开数据集文件夹'))

                    self.bt_onekey_train = gr.Button(i18n('一键训练'), variant="primary")

                    self.bt_tb = gr.Button(i18n('启动Tensorboard'), variant="primary")

                gr.Markdown(i18n('### 恢复训练'))

                with gr.Row():

                    self.resume_model = gr.Dropdown(choices=sorted(self.names), label='Resume training progress from checkpoints', info=i18n('从检查点恢复训练进度'), interactive=True)

                    with gr.Column():

                        self.bt_refersh = gr.Button(i18n('刷新'))

                        self.bt_resume_train = gr.Button(i18n('恢复训练'), variant="primary")

            with gr.Tab(i18n("推理")):

                with gr.Accordion(i18n('推理说明'), open=False):

                    gr.Markdown(self.info.inference)

                gr.Markdown(i18n('### 推理参数设置'))

                with gr.Row():

                    with gr.Column():

                        self.keychange = gr.Slider(-24, 24, value=0, step=1, label=i18n('变调'))

                        self.file_list = gr.Markdown(value="", label=i18n("文件列表"))

                        with gr.Row():

                            self.resume_model2 = gr.Dropdown(choices=sorted(self.names2), label='Select the model you want to export',
                                                             info=i18n('选择要导出的模型'), interactive=True)
                            with gr.Column():

                                self.bt_refersh2 = gr.Button(value=i18n('刷新模型和音色'))


                                self.bt_out_model = gr.Button(value=i18n('导出模型'), variant="primary")

                        with gr.Row():

                            self.resume_voice = gr.Dropdown(choices=sorted(self.voice_names), label='Select the sound file',
                                                            info=i18n('选择音色文件'), interactive=True)

                        with gr.Row():

                            self.input_wav = gr.Audio(type='filepath', label=i18n('选择待转换音频'), source='upload')

                        with gr.Row():

                            self.bt_infer = gr.Button(value=i18n('开始转换'), variant="primary")

                        with gr.Row():

                            self.output_wav = gr.Audio(label=i18n('输出音频'), interactive=False)

            self.bt_open_dataset_folder.click(fn=self.openfolder)
            self.bt_onekey_train.click(fn=self.onekey_training,inputs=[self.model_name, self.thread_count,self.learning_rate,self.batch_size, self.info_interval, self.eval_interval,self.save_interval, self.keep_ckpts, self.slow_model])
            self.bt_out_model.click(fn=self.out_model, inputs=[self.model_name, self.resume_model2])
            self.bt_tb.click(fn=self.tensorboard)
            self.bt_refersh.click(fn=self.refresh_model, inputs=[self.model_name], outputs=[self.resume_model])
            self.bt_resume_train.click(fn=self.resume_train, inputs=[self.model_name, self.resume_model, self.learning_rate,self.batch_size, self.info_interval, self.eval_interval,self.save_interval, self.keep_ckpts, self.slow_model])
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
                print(i18n('打开文件夹失败！'))
        except BaseException:
            print(i18n('打开文件夹失败！'))

    def preprocessing(self, thread_count):
        print(i18n('开始预处理'))
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
            config["train"]["pretrain"] = "vits_pretrain\sovits5.0.pretrain.pth"
        else:
            config["train"]["pretrain"] = ""
        with open("configs/train.yaml", "w") as f:
            yaml.dump(config, f)
        return f"{config['log']}"

    def training(self, model_name):
        print(i18n('开始训练'))
        train_process = subprocess.Popen('python -u svc_trainer.py -c ' + self.train_config_path + ' -n ' + str(model_name), stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def onekey_training(self, model_name, thread_count, learning_rate, batch_size, info_interval, eval_interval, save_interval, keep_ckpts, slow_model):
        print(self, model_name, thread_count, learning_rate, batch_size, info_interval, eval_interval,
              save_interval, keep_ckpts)
        self.create_config(model_name, learning_rate, batch_size, info_interval, eval_interval, save_interval, keep_ckpts, slow_model)
        self.preprocessing(thread_count)
        self.training(model_name)

    def out_model(self, model_name, resume_model2):
        print(i18n('开始导出模型'))
        try:
            subprocess.Popen('python -u svc_export.py -c {} -p "chkpt/{}/{}"'.format(self.train_config_path, model_name, resume_model2),stdout=subprocess.PIPE)
            print(i18n('导出模型成功'))
        except Exception as e:
            print(i18n("出现错误："), e)


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
            return {"label": i18n("缺少模型文件"), "__type__": "update"}

    def refresh_model2(self, model_name):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, f"chkpt/{model_name}")
        self.names2 = []
        try:
            for self.name in os.listdir(self.model_root):
                if self.name.endswith(".pt"):
                    self.names2.append(self.name)
            return {"choices": sorted(self.names2), "__type__": "update"}
        except FileNotFoundError:
            return {"label": i18n("缺少模型文件"), "__type__": "update"}

    def refresh_voice(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, "data_svc/singer")
        self.voice_names = []
        try:
            for self.name in os.listdir(self.model_root):
                if self.name.endswith(".npy"):
                    self.voice_names.append(self.name)
            return {"choices": sorted(self.voice_names), "__type__": "update"}
        except FileNotFoundError:
            return {"label": i18n("缺少文件"), "__type__": "update"}

    def refresh_model_and_voice(self, model_name):
        model_update = self.refresh_model2(model_name)
        voice_update = self.refresh_voice()
        return model_update, voice_update

    def resume_train(self, model_name, resume_model ,learning_rate, batch_size, info_interval, eval_interval, save_interval, keep_ckpts, slow_model):
        print(i18n('开始恢复训练'))
        self.create_config(model_name, learning_rate, batch_size, info_interval, eval_interval, save_interval,keep_ckpts, slow_model)
        train_process = subprocess.Popen('python -u svc_trainer.py -c {} -n {} -p "chkpt/{}/{}"'.format(self.train_config_path, model_name, model_name, resume_model), stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def inference(self, input, resume_voice, keychange):
        if os.path.exists("test.wav"):
            os.remove("test.wav")
            print(i18n("已清理残留文件"))
        else:
            print(i18n("无需清理残留文件"))
        self.train_config_path = 'configs/train.yaml'
        print(i18n('开始推理'))
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
        print(i18n("推理成功"))
        return "svc_out.wav"

class Info:
    def __init__(self) -> None:
        self.train = i18n('### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完')

        self.inference = i18n('### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完')


LANGUAGE_LIST = ['zh_CN', 'en_US']
LANGUAGE_ALL = {
    'zh_CN': {
        'SUPER': 'END',
        'LANGUAGE': 'zh_CN',
        '初始化成功': '初始化成功',
        '就绪': '就绪',
        '预处理-训练': '预处理-训练',
        '训练说明': '训练说明',
        '### 预处理参数设置': '### 预处理参数设置',
        '模型名称': '模型名称',
        'f0提取器': 'f0提取器',
        '预处理线程数': '预处理线程数',
        '### 训练参数设置': '### 训练参数设置',
        '学习率': '学习率',
        '批大小': '批大小',
        '训练日志记录间隔（step）': '训练日志记录间隔（step）',
        '验证集验证间隔（epoch）': '验证集验证间隔（epoch）',
        '检查点保存间隔（epoch）': '检查点保存间隔（epoch）',
        '保留最新的检查点文件(0保存全部)': '保留最新的检查点文件(0保存全部)',
        '是否添加底模': '是否添加底模',
        '### 开始训练': '### 开始训练',
        '打开数据集文件夹': '打开数据集文件夹',
        '一键训练': '一键训练',
        '启动Tensorboard': '启动Tensorboard',
        '### 恢复训练': '### 恢复训练',
        '从检查点恢复训练进度': '从检查点恢复训练进度',
        '刷新': '刷新',
        '恢复训练': '恢复训练',
        '推理': '推理',
        '推理说明': '推理说明',
        '### 推理参数设置': '### 推理参数设置',
        '变调': '变调',
        '文件列表': '文件列表',
        '选择要导出的模型': '选择要导出的模型',
        '刷新模型和音色': '刷新模型和音色',
        '导出模型': '导出模型',
        '选择音色文件': '选择音色文件',
        '选择待转换音频': '选择待转换音频',
        '开始转换': '开始转换',
        '输出音频': '输出音频',
        '打开文件夹失败！': '打开文件夹失败！',
        '开始预处理': '开始预处理',
        '开始训练': '开始训练',
        '开始导出模型': '开始导出模型',
        '导出模型成功': '导出模型成功',
        '出现错误：': '出现错误：',
        '缺少模型文件': '缺少模型文件',
        '缺少文件': '缺少文件',
        '已清理残留文件': '已清理残留文件',
        '无需清理残留文件': '无需清理残留文件',
        '开始推理': '开始推理',
        '推理成功': '推理成功',
        '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完': '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完'
    },
    'en_US': {
        'SUPER': 'zh_CN',
        'LANGUAGE': 'en_US',
        '初始化成功': 'Initialization successful',
        '就绪': 'Ready',
        '预处理-训练': 'Preprocessing-Training',
        '训练说明': 'Training instructions',
        '### 预处理参数设置': '### Preprocessing parameter settings',
        '模型名称': 'Model name',
        'f0提取器': 'f0 extractor',
        '预处理线程数': 'Preprocessing thread number',
        '### 训练参数设置': '### Training parameter settings',
        '学习率': 'Learning rate',
        '批大小': 'Batch size',
        '训练日志记录间隔（step）': 'Training log recording interval (step)',
        '验证集验证间隔（epoch）': 'Validation set validation interval (epoch)',
        '检查点保存间隔（epoch）': 'Checkpoint save interval (epoch)',
        '保留最新的检查点文件(0保存全部)': 'Keep the latest checkpoint file (0 save all)',
        '是否添加底模': 'Whether to add the base model',
        '### 开始训练': '### Start training',
        '打开数据集文件夹': 'Open the dataset folder',
        '一键训练': 'One-click training',
        '启动Tensorboard': 'Start Tensorboard',
        '### 恢复训练': '### Resume training',
        '从检查点恢复训练进度': 'Restore training progress from checkpoint',
        '刷新': 'Refresh',
        '恢复训练': 'Resume training',
        "推理": "Inference",
        "推理说明": "Inference instructions",
        "### 推理参数设置": "### Inference parameter settings",
        "变调": "Pitch shift",
        "文件列表": "File list",
        "选择要导出的模型": "Select the model to export",
        "刷新模型和音色": "Refresh model and timbre",
        "导出模型": "Export model",
        "选择音色文件": "Select timbre file",
        "选择待转换音频": "Select audio to be converted",
        "开始转换": "Start conversion",
        "输出音频": "Output audio",
        "打开文件夹失败！": "Failed to open folder!",
        "开始预处理": "Start preprocessing",
        "开始训练": "Start training",
        "开始导出模型": "Start exporting model",
        "导出模型成功": "Model exported successfully",
        "出现错误：": "An error occurred:",
        "缺少模型文件": "Missing model file",
        '缺少文件': 'Missing file',
        "已清理残留文件": "Residual files cleaned up",
        "无需清理残留文件": "No need to clean up residual files",
        "开始推理": "Start inference",
        '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完': '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)first writing|[@thestmitsuk](https://github.com/thestmitsuki)second completion'
    }
}

class I18nAuto:
    def __init__(self, language=None):
        self.language_list = LANGUAGE_LIST
        self.language_all = LANGUAGE_ALL
        self.language_map = {}
        self.language = language or locale.getdefaultlocale()[0]
        if self.language not in self.language_list:
            self.language = 'zh_CN'
        self.read_language(self.language_all['zh_CN'])
        while self.language_all[self.language]['SUPER'] != 'END':
            self.read_language(self.language_all[self.language])
            self.language = self.language_all[self.language]['SUPER']

    def read_language(self, lang_dict: dict):
        self.language_map.update(lang_dict)

    def __call__(self, key):
        return self.language_map[key]

if __name__ == "__main__":
    i18n = I18nAuto()
    webui = WebUI()
