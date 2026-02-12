#! python

import json
import sys
import torch
import torch.nn as nn
import torchaudio
import math
import argparse

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import PatchEmbed

sys.path.append('AudioMAE')
import models_vit

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def conv_wave2fbank(config, waveform, sr):
    waveform = waveform - waveform.mean()
    # waveform: [ch, nsample]
    #print('sr: '+str(sr))
    #print('waveform: '+str(waveform.shape))
    fbank = torchaudio.compliance.kaldi.fbank(waveform,
                                              htk_compat=True,
                                              sample_frequency=sr,
                                              use_energy=False,
                                              window_type='hanning',
                                              num_mel_bins=config['num_mel_bins'],
                                              dither=0.0,
                                              frame_shift=10)
    # fbank: [1024, 128]
    #print('fbank(0): '+str(fbank.shape))
    target_length = config['target_length']
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    #print('p: '+str(p))
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # fbank: [1024, 128]
    #print('fbank(1): '+str(fbank.shape))
    fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
    # fbank: [1, 128, 1024]
    #print('fbank(2): '+str(fbank.shape))
    fbank = (fbank - config['mean']) / (config['std'] * 2)
    # fbank: [1, 128, 1024]
    #print('fbank(3): '+str(fbank.shape))
    fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
    # fbank: [1024, 128]
    #print('fbank(4): '+str(fbank.shape))
    return fbank.unsqueeze(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-d_embed_full', help='directory: output embeddings (full layer)')
    parser.add_argument('-layer', help='parameter: index of the embeddings layer', type=int, default=11, required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, default=5000, required=True)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, default=500, required=True)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    parser.add_argument('-model_type', help='parameter: model type(finetuned|pretrained)', required=True)
    parser.add_argument('-hop_mean', help='parameter: apply mean', action='store_true')
    parser.add_argument('-halfA', help='parameter: half PCM input / half PCM output', action='store_true')
    parser.add_argument('-halfB', help='parameter: half PCM output', action='store_true')
    parser.add_argument('-halfC', help='parameter: half PCM input', action='store_true')
    args = parser.parse_args()

    print('** get embeddings (Meta AudioMAE) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print('  output full layer : '+str(args.d_embed_full))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print(' parameter')
    print('  index of layer    : '+str(args.layer))
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  model type        : '+str(args.model_type))
    print('  zero padding      : '+str(args.zero_padding))
    print('  hop_mean          : '+str(args.hop_mean))
    print('  half(A)           : '+str(args.halfA))
    print('  half(B)           : '+str(args.halfB))
    print('  half(C)           : '+str(args.halfC))

    # fixed value
    DEFAULT_FS = 44100
    MODEL_FS = 16000
    L = 12
    Z = 768
    hop_sec = 0.16

    # configure
    config = {
        'num_mel_bins': 128, 
        'target_length': 1024,
        'mean': -4.2677393, 
        'std': 4.5689974,
        'nb_classes': 527,
        'drop_path': 0.1,
        'global_pool': True,
        'mask_2d': True,
        'use_custom_patch': False
    }  

    # model
    model = models_vit.__dict__['vit_base_patch16'](
        num_classes=config['nb_classes'],
        drop_path_rate=config['drop_path'],
        global_pool=config['global_pool'],
        mask_2d=config['mask_2d'],
        use_custom_patch=config['use_custom_patch']
    )

    img_size=(config['target_length'], 128) # 1024, 128
    in_chans=1
    model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=Z, stride=16) # no overlap. stride=img_size=16
    num_patches = model.patch_embed.num_patches
    #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, Z), requires_grad=False)  # fixed sin-cos embedding

    f_cp = 'checkpoint/'+args.model_type+'.pth'
    print("Load pre-trained checkpoint from: %s" % f_cp)
    #checkpoint = torch.load(f_cp, map_location='cpu')
    checkpoint = torch.load(f_cp, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(' model')
    print('  num of parameters : '+str(count_parameters(model)))

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    print(' GPU                : '+str(device))
    model.to(device)
    model.eval()

    # fs converter
    sr = DEFAULT_FS
    tr_fsconv = torchaudio.transforms.Resample(sr, MODEL_FS)

    # chunk processing
    if args.chunk_len_msec == 0:
        frame_shift = int(0.010 * MODEL_FS)#160
        frame_length = int(0.025 * MODEL_FS)#400
        chunk_length = config['target_length'] * frame_shift + (frame_length-frame_shift)#164080
        N = int((config['target_length'] * frame_shift) / (MODEL_FS*hop_sec))#64
        print(' chunk processing')
        print('  frame_shift       : '+str(frame_shift))
        print('  frame_length      : '+str(frame_length))
        print('  chunk_length      : '+str(chunk_length))
        print('  N                 : '+str(N))
    else:
        frame_shift = int(0.010 * MODEL_FS)#160
        frame_length = int(0.025 * MODEL_FS)#400

        chunk_sample_raw = config['target_length'] * frame_shift#163840
        chunk_tail_sample = frame_length - frame_shift
        chunk_sample = chunk_sample_raw + chunk_tail_sample
        N = int(chunk_sample_raw / (hop_sec*MODEL_FS))#64
        chunk_hop_sample = int(args.chunk_hop_msec * (MODEL_FS/1000))
        if args.halfA:
            # half input, half output
            chunk_pcm_sample = int(chunk_sample_raw / 2) + chunk_tail_sample
            zero_top_sample = int(chunk_sample_raw / 4)
            Nout = int(N/2)
        elif args.halfB:
            # full input, half output
            chunk_pcm_sample = chunk_sample_raw + chunk_tail_sample
            zero_top_sample = int(chunk_sample_raw / 4)
            Nout = int(N/2)
        elif args.halfC:
            # half input, full output
            chunk_pcm_sample = int(chunk_sample_raw / 2) + chunk_tail_sample
            zero_top_sample = int(chunk_sample_raw / 4)
            Nout = N
        else:
            chunk_pcm_sample = chunk_sample_raw + chunk_tail_sample
            zero_top_sample = int(chunk_sample_raw / 2)
            Nout = N
        print(' chunk processing')
        print('  chunk_sample(raw) : '+str(chunk_sample_raw))
        print('  chunk_sample      : '+str(chunk_sample))
        print('  chunk_pcm_sample  : '+str(chunk_pcm_sample))
        print('  chunk_hop_sample  : '+str(chunk_hop_sample))
        print('  zero_top_sample   : '+str(zero_top_sample))
        print('  N                 : '+str(N))
        print('  Nout              : '+str(Nout))

    # get embeddings
    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_fname_tmp = json.load(f)
    a_fname = []
    for split in a_fname_tmp:
        for fname in a_fname_tmp[split]:
            if fname.startswith('#'):
                continue
            a_fname.append(fname)
    del a_fname_tmp
    a_fname.sort()
    for fname in a_fname:
        print(fname)
        wave, sr = torchaudio.load(args.d_audio.rstrip('/')+'/'+fname+'.wav')
        wave_mono = torch.mean(wave, dim=0)
        del wave
        if sr != MODEL_FS:
            if sr != DEFAULT_FS:
                print(sr)
                tr_fsconv_tmp = torchaudio.transforms.Resample(sr, MODEL_FS)
                wave_mono = tr_fsconv_tmp(wave_mono)
            else:
                wave_mono = tr_fsconv(wave_mono)

        if args.chunk_len_msec == 0:
            #print('wave_mono: '+str(wave_mono.shape))
            n_chunk = math.ceil(wave_mono.shape[0] / (config['target_length'] * frame_shift))
            #print('n_chunk: '+str(n_chunk))
            a_embeddings = torch.zeros((n_chunk, N, L, Z), dtype=torch.float32)
            a_cls_tokens = torch.zeros((n_chunk, L, Z), dtype=torch.float32)
            for i in range(n_chunk):
                input_audio = torch.zeros((1, chunk_length), dtype=torch.float32)
                idx_s = i * config['target_length'] * frame_shift
                idx_e = min(idx_s + chunk_length, wave_mono.shape[0])
                input_audio[0, :(idx_e-idx_s)] = wave_mono[idx_s:idx_e]

                fbank = conv_wave2fbank(config, input_audio.to(device), MODEL_FS)
                # fbank: [1, 1024, 128]

                with torch.no_grad():
                    output, embeddings, cls_tokens = model(fbank.unsqueeze(0))
                    #print('output: '+str(output.shape))
                    #print('embeddings: '+str(embeddings.shape))
                    #print('cls_tokens: '+str(cls_tokens.shape))
                    # embeddings: [B(1), N(64), L(12), Z(768)]
                    # cls_tokens: [B(1), L(12), Z(768)]
                a_embeddings[i] = embeddings.squeeze(0).detach().cpu()
                a_cls_tokens[i] = cls_tokens.squeeze(0).detach().cpu()

            a_embeddings = a_embeddings.reshape(n_chunk*N, L, Z)
            if args.d_embed_full is not None:
                torch.save(a_embeddings, args.d_embed_full.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')
                torch.save(a_cls_tokens, args.d_embed_full.rstrip('/')+'/'+fname+'_cls_tokens_'+args.model_type+'.dat')
            torch.save(a_embeddings[:,args.layer,:], args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')
            torch.save(a_cls_tokens[:,args.layer,:], args.d_embed.rstrip('/')+'/'+fname+'_cls_tokens_'+args.model_type+'.dat')

        else:
            # add half chunk to top
            if args.zero_padding:
                zero_top = torch.zeros(zero_top_sample, dtype=wave_mono.dtype)
                wave_mono = torch.cat([zero_top, wave_mono])

            n_chunk = math.ceil(wave_mono.shape[0] / chunk_hop_sample)
            if args.hop_mean:
                a_embeddings = torch.zeros([n_chunk, L, Z], dtype=torch.float32)
            else:
                a_embeddings = torch.zeros([n_chunk, Nout, L, Z], dtype=torch.float32)
            a_cls_tokens = torch.zeros((n_chunk, L, Z), dtype=torch.float32)

            for i in range(n_chunk):
                input_audio = torch.zeros(1, chunk_sample, dtype=torch.float32)
                idx_s = i * chunk_hop_sample
                idx_e = min(idx_s + chunk_pcm_sample, wave_mono.shape[0])
                input_audio[0, :(idx_e-idx_s)] = wave_mono[idx_s:idx_e]
                fbank = conv_wave2fbank(config, input_audio.to(device), MODEL_FS)
                # fbank: [1, 1024, 128]

                with torch.no_grad():
                    output, embeddings, cls_tokens = model(fbank.unsqueeze(0))
                # output: [1, 527]
                # embeddings: [B(1), N(64), L(12), Z(768)]
                # cls_tokens: [B(1), L(12), Z(768)]

                if args.hop_mean:
                    a_embeddings[i] = embeddings.squeeze(0).detach().cpu()[:Nout,:,:].mean(dim=0)
                else:
                    a_embeddings[i] = embeddings.squeeze(0).detach().cpu()[:Nout,:,:]
                a_cls_tokens[i] = cls_tokens.squeeze(0).detach().cpu()

            if args.hop_mean is False:
                a_embeddings = a_embeddings.reshape(n_chunk*Nout, L, Z)

            if args.d_embed_full is not None:
                torch.save(a_embeddings, args.d_embed_full.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')
                torch.save(a_cls_tokens, args.d_embed_full.rstrip('/')+'/'+fname+'_cls_tokens_'+args.model_type+'.dat')
            torch.save(a_embeddings[:,args.layer,:], args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')
            torch.save(a_cls_tokens[:,args.layer,:], args.d_embed.rstrip('/')+'/'+fname+'_cls_tokens_'+args.model_type+'.dat')

    print('** done **')
