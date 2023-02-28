import emotional_vits_onnx_model
import utils
import torch
import commons

hps = utils.get_hparams_from_file("nene.json")
net_g = emotional_vits_onnx_model.SynthesizerTrn(
    40,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()
_ = utils.load_checkpoint("nene.pth", net_g)

stn_tst = torch.LongTensor([0,20,0,21,0,22,0])
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    sid = torch.tensor([0])
    emo = torch.randn(1024)
    o = net_g(x_tst, x_tst_lengths, sid=sid, emo=emo)