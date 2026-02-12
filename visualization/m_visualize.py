#! python

import os
import argparse
import json
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def show_embed(time, ana, label, title, axes, idx, final_flag=False):
    for i in range(len(time)):
        color=ana[i]
        axes[idx].fill_between([time[i], time[i+1] if i < len(time)-1 else time[i]], 0, 1, facecolor=color)
    axes[idx].set_ylim(0, 1)
    axes[idx].set_yticks([])
    axes[idx].set_ylabel(label, rotation=0, fontsize=args.fontsize, labelpad=4, horizontalalignment='right', verticalalignment='center')
    if title != '':
        axes[idx].set_title(title, fontsize=args.fontsize, pad=1, fontstyle='italic', loc='left')
    if final_flag is False:
        axes[idx].tick_params(axis='x', length=2, labelbottom=False)

    return axes
    

def graph_embeddings():
    # 0: spec
    # 1: annotated label

    # (2): space

    ## Self-supervised Learning: Masked Language Modeling (MLM)
    # 3-4: MusicFM(msd) (no_pooling, pooling)
    # 5-6: MERT(330M) (no_pooling, pooling)
    # 7-8: MetaAudioMAE(pretrained) (no_pooling, pooling)

    # (9): space

    ## Self-supervised Learning: Contrastive Learning
    # 10: MULE (no_pooling)

    # (11): space

    ## Self-supervised Learning: Tokenization(Codec)
    # 12-13: EnCodec(24kHz_24kbps) (no_pooling, pooling)
    # 14-15: DAC (no_pooling, pooling)

    # (16): space

    ## Supervised Fine-tuning (Audio Tagging) after MLM
    # 17-18: MetaAudioMAE(finetuned) (no_pooling, pooling)

    # (19): space

    ## Supervised Learning (Audio Tagging)
    # 20: PANNs(Cnn14) (pooling)
    # 21-22: PaSST (no_pooling, pooling)

    # (23): space

    ## Supervised Learning & Fine-tuning (Sound Event Detection)
    # 24-25: PANNs(Cnn14_dlm) (no_pooling, pooling)

    # (26): space

    ## Cross-modal Contrastive Learning (Audio-Text)
    # 27: CLAP(music_speech) (pooling)

    # (28): space
    ## Cross-modal Contrastive Learning (Audio-Visual)
    # 29-30: OpenL3 (no_pooling, pooling)

    n_graph = 31

    height_ratios = [1 for i in range(n_graph)]
    height_ratios[0] = 2

    tick_length = 2

    # space
    space_flag = True
    space_height = 0.35

    height_ratios[2] = 0.5
    height_ratios[9] = space_height
    height_ratios[11] = space_height
    height_ratios[16] = space_height
    height_ratios[19] = space_height
    height_ratios[23] = space_height
    height_ratios[26] = space_height
    height_ratios[28] = space_height

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = args.fontsize
    fix, axes = plt.subplots(n_graph, 1, figsize=(5,10), sharex=True, gridspec_kw={'height_ratios': height_ratios})

    # spectrogram
    idx = 0
    axes[idx].imshow(a_spec.T, aspect='auto', origin='lower', extent=[time[0], time[-1], freq[0], freq[-1]])
    axes[idx].set_yticks([])
    axes[idx].set_ylabel('Spectrogram', rotation=0, fontsize=args.fontsize, labelpad=5, horizontalalignment='right', verticalalignment='center')
    axes[idx].tick_params(axis='x', length=2, labelbottom=False)

    # label
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    handles = []
    unique_labels = np.unique(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8))
    for label in unique_labels:
        color = colors[label % len(colors)]
        handles.append(plt.Rectangle((0,0), 1, 1, fc=color))
    idx += 1
    for i in range(len(time)):
        color = colors[a_label[i] % len(colors)]
        axes[idx].fill_between([time[i], time[i+1] if i < len(time) - 1 else time[i]], 0, 1, facecolor=color)
    axes[idx].set_ylim(0, 1)
    axes[idx].set_yticks([])
    axes[idx].set_ylabel('Annotation', rotation=0, fontsize=args.fontsize, labelpad=5, horizontalalignment='right', verticalalignment='center')
    axes[idx].tick_params(axis='x', length=2, labelbottom=False)

    legend_labels = [a_label_names.get(label, label) for label in unique_labels]
    axes[idx].legend(handles[1:], legend_labels[1:], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.1, labelspacing=0.3)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # MusicFM(msd)
    idx += 1
    axes = show_embed(time, ana_data_musicfm_msd_A, 'MusicFM (MSD)', 'Self-supervised Learning: Masked Language Modeling (MLM)', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_musicfm_msd_B, 'MusicFM (MSD)*', '', axes, idx)

    # MERT(330M)
    idx += 1
    axes = show_embed(time, ana_data_mert_330m_A, 'MERT (330M)', '', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_mert_330m_B, 'MERT (330M)*', '', axes, idx)

    # MetaAudioMAE(pretrained)
    idx += 1
    axes = show_embed(time, ana_data_meta_pretrained_A, 'AudioMAE (Huang)', '', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_meta_pretrained_B, 'AudioMAE (Huang)*', '', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # MULE
    idx += 1
    axes = show_embed(time, ana_data_mule_A, 'MULE', 'Self-supervised Learning: Contrastive Learning', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # EnCodec(24kHz_24kbps)
    idx += 1
    axes = show_embed(time, ana_data_encodec_24khz_A, 'EnCodec (24kHz/24kbps)', 'Self-supervised Learning: Tokenization (Codec)', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_encodec_24khz_B, 'EnCodec (24kHz/24kbps)*', '', axes, idx)

    # DAC
    idx += 1
    axes = show_embed(time, ana_data_dac_A, 'DAC', '', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_dac_B, 'DAC*', '', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # MetaAudioMAE(finetuned)
    idx += 1
    axes = show_embed(time, ana_data_meta_finetuned_A, 'AudioMAE (Huang)', 'Supervised Fine-tuning (Audio Tagging) after MLM', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_meta_finetuned_B, 'AudioMAE (Huang)*', '', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # PANNs(Cnn14)
    idx += 1
    axes = show_embed(time, ana_data_panns_cnn14_B, 'PANNs*', 'Supervised Learning (Audio Tagging)', axes, idx)

    # PaSST
    idx += 1
    axes = show_embed(time, ana_data_passt_A, 'PaSST', '', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_passt_B, 'PaSST*', '', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # PANNs(Cnn14_DecisionLevelMax)
    idx += 1
    axes = show_embed(time, ana_data_panns_cnn14_dlm_A, 'PANNs', 'Supervised Learning & Fine-tuning (Sound Event Detection)', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_panns_cnn14_dlm_B, 'PANNs*', '', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # CLAP(music_speech)
    idx += 1
    axes = show_embed(time, ana_data_clap_music_speech_B, 'CLAP*', 'Cross-modal Contrastive Learning (Audio-text)', axes, idx)

    # space
    if space_flag:
        idx += 1
        axes[idx].axis('off')

    # OpenL3
    idx += 1
    axes = show_embed(time, ana_data_openl3_A, 'OpenL3', 'Cross-modal Contrastive Learning (Audio-video)', axes, idx)
    idx += 1
    axes = show_embed(time, ana_data_openl3_B, 'OpenL3*', '', axes, idx, final_flag=True)

    axes[idx].set_xlabel('Time [sec]', fontsize=args.fontsize)
    plt.tight_layout()
    plt.subplots_adjust(left=0.26, right=0.85, top=0.95, bottom=0.1)
    plt.savefig('FAE_'+args.method+'_'+args.data+'.png', bbox_inches='tight')
    plt.savefig('FAE_'+args.method+'_'+args.data+'.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: audio')
    parser.add_argument('-d_embed', help='directory: embeddings')
    parser.add_argument('-d_label', help='direcotry: label')
    parser.add_argument('-f_config', help='file: config')
    parser.add_argument('-data', help='data name')
    parser.add_argument('-method', help='parameter: analysis method(PCA|UMAP|TSNE)', default=UMAP)
    parser.add_argument('-fontsize', help='parameter: font size', type=int, default=10)
    args = parser.parse_args()

    print('** visualise embeddings using PCA/UMAP/tSNE **')
    print(' directory')
    print('  audio           : '+str(args.d_audio))
    print('  embeddings      : '+str(args.d_embed))
    print('  label           : '+str(args.d_label))
    print(' file')
    print('  config          : '+str(args.f_config))
    print(' parameter')
    print('  analysis method : '+str(args.method))
    print('  data to convert : '+str(args.data))
    print('  font size       : '+str(args.fontsize))

    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    ##
    ## (1) calculate "nframe" and read the results
    ##
    ## hop size: 500ms
    hop_in_sec = 30 / config['model_parameters']['no_pooling']['To']
    nframe_in_sec = 1.0 / hop_in_sec

    ## spectrogram [nframe(500ms), F]
    tr_feature = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['feature']['sampling_frequency'],
        n_fft=config['feature']['window_length'],
        win_length=config['feature']['window_length'],
        hop_length=config['feature']['hop_sample'],
        pad_mode=config['feature']['pad_mode'],
        n_mels=config['feature']['mel']['n_bins'],
        f_min=config['feature']['mel']['f_min'],
        f_max=config['feature']['mel']['f_max'],
        norm=config['feature']['mel']['norm']
    )
    wave, sr = torchaudio.load(args.d_audio.rstrip('/')+'/'+args.data+'.wav')
    wave_mono = torch.mean(wave, dim=0)
    if sr != config['feature']['sampling_frequency']:
        tr_fsconv = torchaudio.transforms.Resample(sr, config['feature']['sampling_frequency'])
        wave_mono = tr_fsconv(wave_mono)
    spec = torch.log(tr_feature(wave_mono).T + config['feature']['log_offset'])
    hop_spec = config['feature']['hop_sample'] / config['feature']['sampling_frequency']
    ratio_spec = hop_in_sec/hop_spec
    nframe_spec, F = spec.shape
    print(spec.shape)
    nframe = int(nframe_spec/ratio_spec)
    print('nframe_spec: '+str(nframe_spec)+' -> '+str(nframe))

    ## label['structure']['name']: [nframe] (500ms)
    hop_label = config['label']['hop_sec']
    ratio_label = hop_in_sec/hop_label
    label = torch.load(args.d_label.rstrip('/')+'/'+args.data+'.dat')
    nframe_label = label['structure']['name'].shape[0]
    nframe = min(nframe, int(nframe_label/ratio_label))
    print('nframe_label: '+str(nframe_label)+' -> '+str(nframe))

    ## MusicFM
    # msd [nframe(40ms), 1024]
    hop_musicfm = 30 / config['model_parameters']['no_pooling']['MusicFM']['msd']['Ti']
    ratio_musicfm = hop_in_sec/hop_musicfm
    embed_musicfm_msd_A = torch.load(args.d_embed.rstrip('/')+'/MusicFM/no_pooling/'+args.data+'_msd.dat')
    nframe_musicfm_msd_A, Z_musicfm_msd_A = embed_musicfm_msd_A.shape
    nframe = min(nframe, int(nframe_musicfm_msd_A/ratio_musicfm))
    print('nframe_musicfm_msd_A: '+str(nframe_musicfm_msd_A/ratio_musicfm)+' -> '+str(nframe))

    embed_musicfm_msd_B = torch.load(args.d_embed.rstrip('/')+'/MusicFM/pooling/'+args.data+'_msd.dat')
    nframe_musicfm_msd_B, Z_musicfm_msd_B = embed_musicfm_msd_B.shape
    nframe = min(nframe, nframe_musicfm_msd_B)
    print('nframe_musicfm_msd_B: '+str(nframe_musicfm_msd_B)+' -> '+str(nframe))

    ## MERT
    # 330M [nframe(13.3333msec), 1024]
    hop_mert = 30 / config['model_parameters']['no_pooling']['MERT']['330M']['Ti']
    ratio_mert = hop_in_sec/hop_mert
    embed_mert_330m_A = torch.load(args.d_embed.rstrip('/')+'/MERT/no_pooling/'+args.data+'_330M.dat')
    nframe_mert_330m_A, Z_mert_330m_A = embed_mert_330m_A.shape
    nframe = min(nframe, int(nframe_mert_330m_A/ratio_mert))
    print('nframe_mert_330m_A: '+str(nframe_mert_330m_A/ratio_mert)+' -> '+str(nframe))

    embed_mert_330m_B = torch.load(args.d_embed.rstrip('/')+'/MERT/pooling/'+args.data+'_330M.dat')
    nframe_mert_330m_B, Z_mert_330m_B = embed_mert_330m_B.shape
    nframe = min(nframe, nframe_mert_330m_B)
    print('nframe_mert_330m_B: '+str(nframe_mert_330m_B)+' -> '+str(nframe))

    ## MULE [1728, nframe(2000ms)]
    hop_mule = 30 / config['model_parameters']['no_pooling']['MULE']['Ti']
    ratio_mule = hop_in_sec/hop_mule
    embed_mule_A = torch.load(args.d_embed.rstrip('/')+'/MULE/no_pooling/'+args.data+'.dat')
    nframe_mule_A, Z_mule_A = embed_mule_A.shape
    nframe = min(nframe, int(nframe_mule_A/ratio_mule))
    print('nframe_mule_A: '+str(nframe_mule_A/ratio_mule)+' -> '+str(nframe))

    ## PANNs
    # Cnn14 [2048]
    embed_panns_cnn14_B = torch.load(args.d_embed.rstrip('/')+'/PANNs/pooling/'+args.data+'_Cnn14.dat')
    nframe_panns_cnn14_B, Z_panns_cnn14_B = embed_panns_cnn14_B.shape
    nframe = min(nframe, nframe_panns_cnn14_B)
    print('nframe_panns_cnn14_B: '+str(nframe_panns_cnn14_B)+' -> '+str(nframe))

    # Cnn14_DecisionLevelMax [nframe(320ms), 2048]
    hop_panns_cnn14_dlm = 30 / config['model_parameters']['no_pooling']['PANNs']['Cnn14_DecisionLevelMax']['Ti']
    ratio_panns_cnn14_dlm = hop_in_sec/hop_panns_cnn14_dlm
    embed_panns_cnn14_dlm_A = torch.load(args.d_embed.rstrip('/')+'/PANNs/no_pooling/'+args.data+'_Cnn14_DecisionLevelMax.dat')
    nframe_panns_cnn14_dlm_A, Z_panns_cnn14_dlm_A = embed_panns_cnn14_dlm_A.shape
    nframe = min(nframe, int(nframe_panns_cnn14_dlm_A/ratio_panns_cnn14_dlm))
    print('nframe_panns_cnn14_dlm_A: '+str(nframe_panns_cnn14_dlm_A/ratio_panns_cnn14_dlm)+' -> '+str(nframe))

    embed_panns_cnn14_dlm_B = torch.load(args.d_embed.rstrip('/')+'/PANNs/pooling/'+args.data+'_Cnn14_DecisionLevelMax.dat')
    nframe_panns_cnn14_dlm_B, Z_panns_cnn14_dlm_B = embed_panns_cnn14_dlm_B.shape
    nframe = min(nframe, nframe_panns_cnn14_dlm_B)
    print('nframe_panns_cnn14_dlm_B: '+str(nframe_panns_cnn14_dlm_B)+' -> '+str(nframe))

    ## PaSST [nframe(50ms), 768]
    hop_passt = 30 / config['model_parameters']['no_pooling']['PaSST']['Ti']
    ratio_passt = hop_in_sec/hop_passt
    embed_passt_A = torch.load(args.d_embed.rstrip('/')+'/PaSST/no_pooling/'+args.data+'.dat')
    nframe_passt_A, Z_passt_A = embed_passt_A.shape
    nframe = min(nframe, int(nframe_passt_A/ratio_passt))
    print('nframe_passt_A: '+str(nframe_passt_A/ratio_passt)+' -> '+str(nframe))

    embed_passt_B = torch.load(args.d_embed.rstrip('/')+'/PaSST/pooling/'+args.data+'.dat')
    nframe_passt_B, Z_passt_B = embed_passt_B.shape
    nframe = min(nframe, nframe_passt_B)
    print('nframe_passt_B: '+str(nframe_passt_B)+' -> '+str(nframe))

    ## Meta AudioMAE
    # pretrained [nframe(160ms), 768]
    hop_meta = 30 / config['model_parameters']['no_pooling']['MetaAudioMAE']['pretrained']['Ti']
    ratio_meta = hop_in_sec/hop_meta
    embed_meta_pretrained_A = torch.load(args.d_embed.rstrip('/')+'/MetaAudioMAE/no_pooling/'+args.data+'_pretrained.dat')
    nframe_meta_pretrained_A, Z_meta_pretrained_A = embed_meta_pretrained_A.shape
    nframe = min(nframe, int(nframe_meta_pretrained_A/ratio_meta))
    print('nframe_meta_pretrained_A: '+str(nframe_meta_pretrained_A/ratio_meta)+' -> '+str(nframe))

    embed_meta_pretrained_B = torch.load(args.d_embed.rstrip('/')+'/MetaAudioMAE/pooling/'+args.data+'_pretrained.dat')
    nframe_meta_pretrained_B, Z_meta_pretrained_B = embed_meta_pretrained_B.shape
    nframe = min(nframe, nframe_meta_pretrained_B)
    print('nframe_meta_pretrained_B: '+str(nframe_meta_pretrained_B)+' -> '+str(nframe))

    # finetuned [nframe(160ms), 768]
    embed_meta_finetuned_A = torch.load(args.d_embed.rstrip('/')+'/MetaAudioMAE/no_pooling/'+args.data+'_finetuned.dat')
    nframe_meta_finetuned_A, Z_meta_finetuned_A = embed_meta_finetuned_A.shape
    nframe = min(nframe, int(nframe_meta_finetuned_A/ratio_meta))
    print('nframe_meta_finetuned_A: '+str(nframe_meta_finetuned_A/ratio_meta)+' -> '+str(nframe))

    embed_meta_finetuned_B = torch.load(args.d_embed.rstrip('/')+'/MetaAudioMAE/pooling/'+args.data+'_finetuned.dat')
    nframe_meta_finetuned_B, Z_meta_finetuned_B = embed_meta_finetuned_B.shape
    nframe = min(nframe, nframe_meta_finetuned_B)
    print('nframe_meta_finetuned_B: '+str(nframe_meta_finetuned_B)+' -> '+str(nframe))

    ## CLAP
    # music_audioset [nframe(160ms), 512]
    embed_clap_music_B = torch.load(args.d_embed.rstrip('/')+'/CLAP/pooling/'+args.data+'_music_audioset.dat')
    nframe_clap_music_B, Z_clap_music_B = embed_clap_music_B.shape
    nframe = min(nframe, nframe_clap_music_B)
    print('nframe_clap_music_B: '+str(nframe_clap_music_B)+' -> '+str(nframe))

    # music_speech_audioset [nframe(160ms), 512]
    embed_clap_music_speech_B = torch.load(args.d_embed.rstrip('/')+'/CLAP/pooling/'+args.data+'_music_speech_audioset.dat')
    nframe_clap_music_speech_B, Z_clap_music_speech_B = embed_clap_music_speech_B.shape
    nframe = min(nframe, nframe_clap_music_speech_B)
    print('nframe_clap_music_speech_B: '+str(nframe_clap_music_speech_B)+' -> '+str(nframe))

    ## OpenL3 [nframe(100ms), 6144]
    hop_openl3 = 30 / config['model_parameters']['no_pooling']['OpenL3']['Ti']
    ratio_openl3 = hop_in_sec/hop_openl3
    embed_openl3_A = torch.load(args.d_embed.rstrip('/')+'/OpenL3/no_pooling/'+args.data+'.dat')
    nframe_openl3_A, Z_openl3_A = embed_openl3_A.shape
    nframe = min(nframe, int(nframe_openl3_A/ratio_openl3))
    print('nframe_openl3_A: '+str(nframe_openl3_A/ratio_openl3)+' -> '+str(nframe))

    embed_openl3_B = torch.load(args.d_embed.rstrip('/')+'/OpenL3/pooling/'+args.data+'.dat')
    nframe_openl3_B, Z_openl3_B = embed_openl3_B.shape
    nframe = min(nframe, nframe_openl3_B)
    print('nframe_openl3_B: '+str(nframe_openl3_B)+' -> '+str(nframe))

    ## DAC [nframe(11.06ms), 1024]
    hop_dac = 30 / config['model_parameters']['no_pooling']['DAC']['Ti']
    ratio_dac = hop_in_sec/hop_dac
    embed_dac_A = torch.load(args.d_embed.rstrip('/')+'/DAC/no_pooling/'+args.data+'.dat')
    nframe_dac_A, Z_dac_A = embed_dac_A.shape
    nframe = min(nframe, int(nframe_dac_A/ratio_dac))
    print('nframe_dac_A: '+str(nframe_dac_A/ratio_dac)+' -> '+str(nframe))

    embed_dac_B = torch.load(args.d_embed.rstrip('/')+'/DAC/pooling/'+args.data+'.dat')
    nframe_dac_B, Z_dac_B = embed_dac_B.shape
    nframe = min(nframe, nframe_dac_B)
    print('nframe_dac_B: '+str(nframe_dac_B)+' -> '+str(nframe))

    ## EnCodec
    # 24kHz_24kbps [nframe(13.333ms), 128]
    hop_encodec_24khz = 30 / config['model_parameters']['no_pooling']['EnCodec']['24kHz_24kbps']['Ti']
    ratio_encodec_24khz = hop_in_sec/hop_encodec_24khz
    embed_encodec_24khz_A = torch.load(args.d_embed.rstrip('/')+'/EnCodec/no_pooling/'+args.data+'_24kHz_24kbps.dat')
    nframe_encodec_24khz_A, Z_encodec_24khz_A = embed_encodec_24khz_A.shape
    nframe = min(nframe, int(nframe_encodec_24khz_A/ratio_encodec_24khz))
    print('nframe_encodec_24khz_A: '+str(nframe_encodec_24khz_A/ratio_encodec_24khz)+' -> '+str(nframe))

    embed_encodec_24khz_B = torch.load(args.d_embed.rstrip('/')+'/EnCodec/pooling/'+args.data+'_24kHz_24kbps.dat')
    nframe_encodec_24khz_B, Z_encodec_24khz_B = embed_encodec_24khz_B.shape
    nframe = min(nframe, nframe_encodec_24khz_B)
    print('nframe_encodec_24khz_B: '+str(nframe_encodec_24khz_B)+' -> '+str(nframe))

    ##
    ## (2) get the vectors for visualization
    ##
    print('nframe: '+str(nframe))
    func_average = nn.AdaptiveAvgPool1d(nframe)
    scaler = StandardScaler()
    if args.method.lower() == 'umap':
        analysis = UMAP(n_components=3)
    elif args.method.lower() == 'pca':
        analysis = PCA(n_components=3)
    elif args.method.lower() == 'tsne':
        analysis = TSNE(n_components=3)

    ## spec
    a_spec = np.zeros([nframe, F], dtype=np.float32)
    a_spec[:nframe] = spec[::int(ratio_spec)][:nframe].cpu().numpy()

    ## label
    a_label = np.zeros([nframe], dtype=np.int8)
    a_label[:nframe] = label['structure']['name'][:nframe].numpy() + 1
    for i in range(nframe):
        if label['structure']['boundary'][i] == 1:
            a_label[i] = 0

    ## MusicFM
    # msd [nframe(40ms), 1024]
    a_embed_musicfm_msd_A = np.zeros([nframe, Z_musicfm_msd_A], dtype=np.float32)
    a_embed_musicfm_msd_A[:nframe] = func_average(embed_musicfm_msd_A[:int(nframe*ratio_musicfm)].T).T[:nframe].detach().cpu().numpy()
    a_embed_musicfm_msd_A = scaler.fit_transform(a_embed_musicfm_msd_A)
    ana_data_musicfm_msd_A = analysis.fit_transform(a_embed_musicfm_msd_A)
    ana_data_musicfm_msd_A = scaler.fit_transform(ana_data_musicfm_msd_A)
    max_val = ana_data_musicfm_msd_A.max()
    min_val = ana_data_musicfm_msd_A.min()
    for i in range(len(ana_data_musicfm_msd_A)):
        ana_data_musicfm_msd_A[i] = (ana_data_musicfm_msd_A[i] - min_val) / (max_val - min_val)
    del embed_musicfm_msd_A

    a_embed_musicfm_msd_B = np.zeros([nframe, Z_musicfm_msd_B], dtype=np.float32)
    a_embed_musicfm_msd_B[:nframe] = embed_musicfm_msd_B[:nframe].detach().cpu().numpy()
    a_embed_musicfm_msd_B = scaler.fit_transform(a_embed_musicfm_msd_B)
    ana_data_musicfm_msd_B = analysis.fit_transform(a_embed_musicfm_msd_B)
    ana_data_musicfm_msd_B = scaler.fit_transform(ana_data_musicfm_msd_B)
    max_val = ana_data_musicfm_msd_B.max()
    min_val = ana_data_musicfm_msd_B.min()
    for i in range(len(ana_data_musicfm_msd_B)):
        ana_data_musicfm_msd_B[i] = (ana_data_musicfm_msd_B[i] - min_val) / (max_val - min_val)
    del embed_musicfm_msd_B

    # MERT
    a_embed_mert_330m_A = np.zeros([nframe, Z_mert_330m_A], dtype=np.float32)
    a_embed_mert_330m_A[:nframe] = func_average(embed_mert_330m_A[:int(nframe*ratio_mert)].T).T[:nframe].detach().cpu().numpy()
    a_embed_mert_330m_A = scaler.fit_transform(a_embed_mert_330m_A)
    ana_data_mert_330m_A = analysis.fit_transform(a_embed_mert_330m_A)
    ana_data_mert_330m_A = scaler.fit_transform(ana_data_mert_330m_A)
    max_val = ana_data_mert_330m_A.max()
    min_val = ana_data_mert_330m_A.min()
    for i in range(len(ana_data_mert_330m_A)):
        ana_data_mert_330m_A[i] = (ana_data_mert_330m_A[i] - min_val) / (max_val - min_val)
    del embed_mert_330m_A

    a_embed_mert_330m_B = np.zeros([nframe, Z_mert_330m_B], dtype=np.float32)
    a_embed_mert_330m_B[:nframe] = embed_mert_330m_B[:nframe].detach().cpu().numpy()
    a_embed_mert_330m_B = scaler.fit_transform(a_embed_mert_330m_B)
    ana_data_mert_330m_B = analysis.fit_transform(a_embed_mert_330m_B)
    ana_data_mert_330m_B = scaler.fit_transform(ana_data_mert_330m_B)
    max_val = ana_data_mert_330m_B.max()
    min_val = ana_data_mert_330m_B.min()
    for i in range(len(ana_data_mert_330m_B)):
        ana_data_mert_330m_B[i] = (ana_data_mert_330m_B[i] - min_val) / (max_val - min_val)
    del embed_mert_330m_B

    # MULE
    a_embed_mule_A = np.zeros([nframe, Z_mule_A], dtype=np.float32)
    a_embed_mule_A[:nframe] = func_average(embed_mule_A[:int(nframe*ratio_mule)].T).T[:nframe].detach().cpu().numpy()
    a_embed_mule_A = scaler.fit_transform(a_embed_mule_A)
    ana_data_mule_A = analysis.fit_transform(a_embed_mule_A)
    ana_data_mule_A = scaler.fit_transform(ana_data_mule_A)
    max_val = ana_data_mule_A.max()
    min_val = ana_data_mule_A.min()
    for i in range(len(ana_data_mule_A)):
        ana_data_mule_A[i] = (ana_data_mule_A[i] - min_val) / (max_val - min_val)
    del embed_mule_A

    # PANNs
    a_embed_panns_cnn14_B = np.zeros([nframe, Z_panns_cnn14_B], dtype=np.float32)
    a_embed_panns_cnn14_B[:nframe] = embed_panns_cnn14_B[:nframe].detach().cpu().numpy()
    a_embed_panns_cnn14_B = scaler.fit_transform(a_embed_panns_cnn14_B)
    ana_data_panns_cnn14_B = analysis.fit_transform(a_embed_panns_cnn14_B)
    ana_data_panns_cnn14_B = scaler.fit_transform(ana_data_panns_cnn14_B)
    max_val = ana_data_panns_cnn14_B.max()
    min_val = ana_data_panns_cnn14_B.min()
    for i in range(len(ana_data_panns_cnn14_B)):
        ana_data_panns_cnn14_B[i] = (ana_data_panns_cnn14_B[i] - min_val) / (max_val - min_val)
    del embed_panns_cnn14_B

    a_embed_panns_cnn14_dlm_A = np.zeros([nframe, Z_panns_cnn14_dlm_A], dtype=np.float32)
    a_embed_panns_cnn14_dlm_A[:nframe] = func_average(embed_panns_cnn14_dlm_A[:int(nframe*ratio_panns_cnn14_dlm)].T).T[:nframe].detach().cpu().numpy()
    a_embed_panns_cnn14_dlm_A = scaler.fit_transform(a_embed_panns_cnn14_dlm_A)
    ana_data_panns_cnn14_dlm_A = analysis.fit_transform(a_embed_panns_cnn14_dlm_A)
    ana_data_panns_cnn14_dlm_A = scaler.fit_transform(ana_data_panns_cnn14_dlm_A)
    max_val = ana_data_panns_cnn14_dlm_A.max()
    min_val = ana_data_panns_cnn14_dlm_A.min()
    for i in range(len(ana_data_panns_cnn14_dlm_A)):
        ana_data_panns_cnn14_dlm_A[i] = (ana_data_panns_cnn14_dlm_A[i] - min_val) / (max_val - min_val)
    del embed_panns_cnn14_dlm_A

    a_embed_panns_cnn14_dlm_B = np.zeros([nframe, Z_panns_cnn14_dlm_B], dtype=np.float32)
    a_embed_panns_cnn14_dlm_B[:nframe] = embed_panns_cnn14_dlm_B[:nframe].detach().cpu().numpy()
    a_embed_panns_cnn14_dlm_B = scaler.fit_transform(a_embed_panns_cnn14_dlm_B)
    ana_data_panns_cnn14_dlm_B = analysis.fit_transform(a_embed_panns_cnn14_dlm_B)
    ana_data_panns_cnn14_dlm_B = scaler.fit_transform(ana_data_panns_cnn14_dlm_B)
    max_val = ana_data_panns_cnn14_dlm_B.max()
    min_val = ana_data_panns_cnn14_dlm_B.min()
    for i in range(len(ana_data_panns_cnn14_dlm_B)):
        ana_data_panns_cnn14_dlm_B[i] = (ana_data_panns_cnn14_dlm_B[i] - min_val) / (max_val - min_val)
    del embed_panns_cnn14_dlm_B

    # PaSST
    a_embed_passt_A = np.zeros([nframe, Z_passt_A], dtype=np.float32)
    a_embed_passt_A[:nframe] = func_average(embed_passt_A[:int(nframe*ratio_passt)].T).T[:nframe].detach().cpu().numpy()
    a_embed_passt_A = scaler.fit_transform(a_embed_passt_A)
    ana_data_passt_A = analysis.fit_transform(a_embed_passt_A)
    ana_data_passt_A = scaler.fit_transform(ana_data_passt_A)
    max_val = ana_data_passt_A.max()
    min_val = ana_data_passt_A.min()
    for i in range(len(ana_data_passt_A)):
        ana_data_passt_A[i] = (ana_data_passt_A[i] - min_val) / (max_val - min_val)
    del embed_passt_A

    a_embed_passt_B = np.zeros([nframe, Z_passt_B], dtype=np.float32)
    a_embed_passt_B[:nframe] = embed_passt_B[:nframe].detach().cpu().numpy()
    a_embed_passt_B = scaler.fit_transform(a_embed_passt_B)
    ana_data_passt_B = analysis.fit_transform(a_embed_passt_B)
    ana_data_passt_B = scaler.fit_transform(ana_data_passt_B)
    max_val = ana_data_passt_B.max()
    min_val = ana_data_passt_B.min()
    for i in range(len(ana_data_passt_B)):
        ana_data_passt_B[i] = (ana_data_passt_B[i] - min_val) / (max_val - min_val)
    del embed_passt_B

    # Meta AudioMAE
    a_embed_meta_pretrained_A = np.zeros([nframe, Z_meta_pretrained_A], dtype=np.float32)
    a_embed_meta_pretrained_A[:nframe] = func_average(embed_meta_pretrained_A[:int(nframe*ratio_meta)].T).T[:nframe].detach().cpu().numpy()
    a_embed_meta_pretrained_A = scaler.fit_transform(a_embed_meta_pretrained_A)
    ana_data_meta_pretrained_A = analysis.fit_transform(a_embed_meta_pretrained_A)
    ana_data_meta_pretrained_A = scaler.fit_transform(ana_data_meta_pretrained_A)
    max_val = ana_data_meta_pretrained_A.max()
    min_val = ana_data_meta_pretrained_A.min()
    for i in range(len(ana_data_meta_pretrained_A)):
        ana_data_meta_pretrained_A[i] = (ana_data_meta_pretrained_A[i] - min_val) / (max_val - min_val)
    del embed_meta_pretrained_A

    a_embed_meta_pretrained_B = np.zeros([nframe, Z_meta_pretrained_B], dtype=np.float32)
    a_embed_meta_pretrained_B[:nframe] = embed_meta_pretrained_B[:nframe].detach().cpu().numpy()
    a_embed_meta_pretrained_B = scaler.fit_transform(a_embed_meta_pretrained_B)
    ana_data_meta_pretrained_B = analysis.fit_transform(a_embed_meta_pretrained_B)
    ana_data_meta_pretrained_B = scaler.fit_transform(ana_data_meta_pretrained_B)
    max_val = ana_data_meta_pretrained_B.max()
    min_val = ana_data_meta_pretrained_B.min()
    for i in range(len(ana_data_meta_pretrained_B)):
        ana_data_meta_pretrained_B[i] = (ana_data_meta_pretrained_B[i] - min_val) / (max_val - min_val)
    del embed_meta_pretrained_B

    a_embed_meta_finetuned_A = np.zeros([nframe, Z_meta_finetuned_A], dtype=np.float32)
    a_embed_meta_finetuned_A[:nframe] = func_average(embed_meta_finetuned_A[:int(nframe*ratio_meta)].T).T[:nframe].detach().cpu().numpy()
    a_embed_meta_finetuned_A = scaler.fit_transform(a_embed_meta_finetuned_A)
    ana_data_meta_finetuned_A = analysis.fit_transform(a_embed_meta_finetuned_A)
    ana_data_meta_finetuned_A = scaler.fit_transform(ana_data_meta_finetuned_A)
    max_val = ana_data_meta_finetuned_A.max()
    min_val = ana_data_meta_finetuned_A.min()
    for i in range(len(ana_data_meta_finetuned_A)):
        ana_data_meta_finetuned_A[i] = (ana_data_meta_finetuned_A[i] - min_val) / (max_val - min_val)
    del embed_meta_finetuned_A

    a_embed_meta_finetuned_B = np.zeros([nframe, Z_meta_finetuned_B], dtype=np.float32)
    a_embed_meta_finetuned_B[:nframe] = embed_meta_finetuned_B[:nframe].detach().cpu().numpy()
    a_embed_meta_finetuned_B = scaler.fit_transform(a_embed_meta_finetuned_B)
    ana_data_meta_finetuned_B = analysis.fit_transform(a_embed_meta_finetuned_B)
    ana_data_meta_finetuned_B = scaler.fit_transform(ana_data_meta_finetuned_B)
    max_val = ana_data_meta_finetuned_B.max()
    min_val = ana_data_meta_finetuned_B.min()
    for i in range(len(ana_data_meta_finetuned_B)):
        ana_data_meta_finetuned_B[i] = (ana_data_meta_finetuned_B[i] - min_val) / (max_val - min_val)
    del embed_meta_finetuned_B

    # CLAP
    a_embed_clap_music_B = np.zeros([nframe, Z_clap_music_B], dtype=np.float32)
    a_embed_clap_music_B[:nframe] = embed_clap_music_B[:nframe].detach().cpu().numpy()
    a_embed_clap_music_B = scaler.fit_transform(a_embed_clap_music_B)
    ana_data_clap_music_B = analysis.fit_transform(a_embed_clap_music_B)
    ana_data_clap_music_B = scaler.fit_transform(ana_data_clap_music_B)
    max_val = ana_data_clap_music_B.max()
    min_val = ana_data_clap_music_B.min()
    for i in range(len(ana_data_clap_music_B)):
        ana_data_clap_music_B[i] = (ana_data_clap_music_B[i] - min_val) / (max_val - min_val)
    del embed_clap_music_B

    a_embed_clap_music_speech_B = np.zeros([nframe, Z_clap_music_speech_B], dtype=np.float32)
    a_embed_clap_music_speech_B[:nframe] = embed_clap_music_speech_B[:nframe].detach().cpu().numpy()
    a_embed_clap_music_speech_B = scaler.fit_transform(a_embed_clap_music_speech_B)
    ana_data_clap_music_speech_B = analysis.fit_transform(a_embed_clap_music_speech_B)
    ana_data_clap_music_speech_B = scaler.fit_transform(ana_data_clap_music_speech_B)
    max_val = ana_data_clap_music_speech_B.max()
    min_val = ana_data_clap_music_speech_B.min()
    for i in range(len(ana_data_clap_music_speech_B)):
        ana_data_clap_music_speech_B[i] = (ana_data_clap_music_speech_B[i] - min_val) / (max_val - min_val)
    del embed_clap_music_speech_B

    # OpenL3
    a_embed_openl3_A = np.zeros([nframe, Z_openl3_A], dtype=np.float32)
    a_embed_openl3_A[:nframe] = func_average(embed_openl3_A[:int(nframe*ratio_openl3)].T).T[:nframe].detach().cpu().numpy()
    a_embed_openl3_A = scaler.fit_transform(a_embed_openl3_A)
    ana_data_openl3_A = analysis.fit_transform(a_embed_openl3_A)
    ana_data_openl3_A = scaler.fit_transform(ana_data_openl3_A)
    max_val = ana_data_openl3_A.max()
    min_val = ana_data_openl3_A.min()
    for i in range(len(ana_data_openl3_A)):
        ana_data_openl3_A[i] = (ana_data_openl3_A[i] - min_val) / (max_val - min_val)
    del embed_openl3_A

    a_embed_openl3_B = np.zeros([nframe, Z_openl3_B], dtype=np.float32)
    a_embed_openl3_B[:nframe] = embed_openl3_B[:nframe].detach().cpu().numpy()
    a_embed_openl3_B = scaler.fit_transform(a_embed_openl3_B)
    ana_data_openl3_B = analysis.fit_transform(a_embed_openl3_B)
    ana_data_openl3_B = scaler.fit_transform(ana_data_openl3_B)
    max_val = ana_data_openl3_B.max()
    min_val = ana_data_openl3_B.min()
    for i in range(len(ana_data_openl3_B)):
        ana_data_openl3_B[i] = (ana_data_openl3_B[i] - min_val) / (max_val - min_val)
    del embed_openl3_B

    # DAC
    a_embed_dac_A = np.zeros([nframe, Z_dac_A], dtype=np.float32)
    a_embed_dac_A[:nframe] = func_average(embed_dac_A[:int(nframe*ratio_dac)].T).T[:nframe].detach().cpu().numpy()
    a_embed_dac_A = scaler.fit_transform(a_embed_dac_A)
    ana_data_dac_A = analysis.fit_transform(a_embed_dac_A)
    ana_data_dac_A = scaler.fit_transform(ana_data_dac_A)
    max_val = ana_data_dac_A.max()
    min_val = ana_data_dac_A.min()
    for i in range(len(ana_data_dac_A)):
        ana_data_dac_A[i] = (ana_data_dac_A[i] - min_val) / (max_val - min_val)
    del embed_dac_A

    a_embed_dac_B = np.zeros([nframe, Z_dac_B], dtype=np.float32)
    a_embed_dac_B[:nframe] = embed_dac_B[:nframe].detach().cpu().numpy()
    a_embed_dac_B = scaler.fit_transform(a_embed_dac_B)
    ana_data_dac_B = analysis.fit_transform(a_embed_dac_B)
    ana_data_dac_B = scaler.fit_transform(ana_data_dac_B)
    max_val = ana_data_dac_B.max()
    min_val = ana_data_dac_B.min()
    for i in range(len(ana_data_dac_B)):
        ana_data_dac_B[i] = (ana_data_dac_B[i] - min_val) / (max_val - min_val)
    del embed_dac_B

    # EnCodec
    a_embed_encodec_24khz_A = np.zeros([nframe, Z_encodec_24khz_A], dtype=np.float32)
    a_embed_encodec_24khz_A[:nframe] = func_average(embed_encodec_24khz_A[:int(nframe*ratio_encodec_24khz)].T).T[:nframe].detach().cpu().numpy()
    a_embed_encodec_24khz_A = scaler.fit_transform(a_embed_encodec_24khz_A)
    ana_data_encodec_24khz_A = analysis.fit_transform(a_embed_encodec_24khz_A)
    ana_data_encodec_24khz_A = scaler.fit_transform(ana_data_encodec_24khz_A)
    max_val = ana_data_encodec_24khz_A.max()
    min_val = ana_data_encodec_24khz_A.min()
    for i in range(len(ana_data_encodec_24khz_A)):
        ana_data_encodec_24khz_A[i] = (ana_data_encodec_24khz_A[i] - min_val) / (max_val - min_val)
    del embed_encodec_24khz_A

    a_embed_encodec_24khz_B = np.zeros([nframe, Z_encodec_24khz_B], dtype=np.float32)
    a_embed_encodec_24khz_B[:nframe] = embed_encodec_24khz_B[:nframe].detach().cpu().numpy()
    a_embed_encodec_24khz_B = scaler.fit_transform(a_embed_encodec_24khz_B)
    ana_data_encodec_24khz_B = analysis.fit_transform(a_embed_encodec_24khz_B)
    ana_data_encodec_24khz_B = scaler.fit_transform(ana_data_encodec_24khz_B)
    max_val = ana_data_encodec_24khz_B.max()
    min_val = ana_data_encodec_24khz_B.min()
    for i in range(len(ana_data_encodec_24khz_B)):
        ana_data_encodec_24khz_B[i] = (ana_data_encodec_24khz_B[i] - min_val) / (max_val - min_val)
    del embed_encodec_24khz_B

    ##
    ## (3) visualisation
    ##
    #time = np.arange(nframe)
    time = np.arange(0, nframe*0.5, 0.5)
    freq = np.arange(F)
    a_label_names = {
        0: 'boundary',
        1: 'intro',
        2: 'verse',
        3: 'chorus',
        4: 'bridge',
        5: 'inst',
        6: 'outro',
        7: 'silence'
    }
    graph_embeddings()
