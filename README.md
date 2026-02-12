# Do Foundational Audio Encoders Understand Music Structure?

This repository contains the official Python implementation of **"Do Foundational Audio Encoders Understand Music Structure?"** presented in ICASSP 2026 ([arXiv 2512.17209](https://arxiv.org/abs/2512.17209)).

## Usage
1. prepare corpus (Harmonix/RWC)
    * see [corpus/README.md](corpus/README.md)

2. calculate FAE features
    * see [FAE/README.md](FAE/README.md)

3. visualize FEA features
    * see [visualization/README.md](visualization/README.md)

4. train/evaluate linear probing models
    * see [MSA/README.md](MSA/README.md)


## Foundational Audio Encoders (FAEs)
| FAE | github repository |
| :--- | :--- |
| MusicFM | https://github.com/minzwon/musicfm |
| MERT | https://github.com/yizhilll/MERT |
| AudioMAE (Huang et al.) | https://github.com/facebookresearch/AudioMAE |
| AudioMAE (Zhong et al.) | This has not been publicly opened. |
| MULE | https://github.com/PandoraMedia/music-audio-representations |
| EnCodec | https://github.com/facebookresearch/encodec |
| DAC | https://github.com/descriptinc/descript-audio-codec |
| PANNs | https://github.com/qiuqiangkong/audioset_tagging_cnn |
| PaSST | https://github.com/kkoutini/PaSST |
| CLAP | https://github.com/LAION-AI/CLAP |
| OpenL3 | https://github.com/torchopenl3/torchopenl3 |


## Supplimental Results
  <head>
    <style>
      .bold-text {
        font-weight: bold;
      }
      .underline-text {
        text-decoration: underline;
      }
      table.manual-border {
        border-collapse: collapse;
        padding: 8px;
      }
      th.right-border, td.right-border {
        border-right: 1px solid #ccc;
      }
      th {
       border-bottom: 1px solid #ccc !important;
      }
      table {
        margin-bottom: 50px;
      }
    </style>
    </head>
<body>
<table class="manual-border">
<caption><span class="bold-text">Table A.</span> 8-fold validaiton of linear probing on Harmonix dataset. Values are mean&plusmn;standard deviation.</caption>
<thead>
<tr>
<th rowspan=3 style="text-align:center; vertical-align:middle;" class="right-border">FAE</th><th colspan=4 style="text-align:center; vertical-align:middle;" class="right-border">Boundary detection</th><th colspan=4 style="text-align:center; vertical-align:middle;">Function prediction</th>
</tr>
<tr>
<th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">HR.5F</th><th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">HR3F</th><th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">PW</th><th style="text-align:center; vertical-align:middle;" colspan=2>ACC</th>
</tr>
<tr>
<th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;">Pooling</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Masked Language Modeling (MLM)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MusicFM (FMA)</td><td>51.22&plusmn;1.12</td><td class="right-border">41.39&plusmn;1.58</td><td>58.80&plusmn;0.84</td><td class="right-border">59.19&plusmn;1.23</td><td><span class="underline-text">66.35</span>&plusmn;1.99</td><td class="right-border">63.78&plusmn;1.29</td><td><span class="underline-text">67.65</span>&plusmn;1.59</td><td>63.14&plusmn;0.97</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MusicFM (MSD)</td><td><span class="bold-text">54.19</span>&plusmn;0.94</td><td class="right-border">49.76&plusmn;0.64</td><td>60.58&plusmn;0.76</td><td class="right-border"><span class="underline-text">63.91</span>&plusmn;1.18</td><td><span class="bold-text">66.89</span>&plusmn;1.52</td><td class="right-border">64.66&plusmn;1.14</td><td><span class="bold-text">68.13</span>&plusmn;1.84</td><td>64.44&plusmn;1.55</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MERT (95M)</td><td>42.94&plusmn;0.85</td><td class="right-border">42.23&plusmn;1.93</td><td>52.25&plusmn;0.56</td><td class="right-border">60.99&plusmn;1.27</td><td>62.44&plusmn;1.22</td><td class="right-border">63.40&plusmn;1.33</td><td>62.58&plusmn;1.16</td><td>62.01&plusmn;1.27</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MERT (330M)</td><td>45.39&plusmn;1.01</td><td class="right-border">40.63&plusmn;1.88</td><td>54.46&plusmn;0.99</td><td class="right-border">57.72&plusmn;1.96</td><td>64.16&plusmn;1.30</td><td class="right-border">64.17&plusmn;1.37</td><td>63.77&plusmn;1.37</td><td>62.30&plusmn;1.46</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Huang)</td><td>36.57&plusmn;1.17</td><td class="right-border">36.95&plusmn;1.18</td><td>51.09&plusmn;1.54</td><td class="right-border">58.11&plusmn;1.09</td><td>60.33&plusmn;0.88</td><td class="right-border">64.58&plusmn;1.49</td><td>59.65&plusmn;1.24</td><td>63.07&plusmn;1.93</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Zhong)</td><td>43.92&plusmn;0.49</td><td class="right-border"><span class="underline-text">53.86</span>&plusmn;1.07</td><td>59.26&plusmn;0.80</td><td class="right-border"><span class="bold-text">64.87</span>&plusmn;0.98</td><td>62.85&plusmn;1.24</td><td class="right-border">64.06&plusmn;1.71</td><td>60.99&plusmn;1.23</td><td>61.33&plusmn;2.02</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Constrastive Learning</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MULE</td><td>20.40&plusmn;0.66</td><td class="right-border">n/a</td><td>43.61&plusmn;0.89</td><td class="right-border">n/a</td><td>57.67&plusmn;1.43</td><td class="right-border">n/a</td><td>57.38&plusmn;1.85</td><td>n/a</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Tokenization (Codec)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (24kHz/3kbps)</td><td>23.49&plusmn;0.89</td><td class="right-border">19.39&plusmn;1.15</td><td>42.63&plusmn;0.75</td><td class="right-border">31.88&plusmn;0.74</td><td>53.86&plusmn;0.98</td><td class="right-border">52.87&plusmn;1.07</td><td>48.95&plusmn;1.56</td><td>45.87&plusmn;1.18</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (24kHz/6kbps)</td><td>23.47&plusmn;0.68</td><td class="right-border">19.15&plusmn;1.38</td><td>42.78&plusmn;0.65</td><td class="right-border">31.90&plusmn;0.80</td><td>53.72&plusmn;1.16</td><td class="right-border">52.71&plusmn;1.12</td><td>48.73&plusmn;1.42</td><td>46.09&plusmn;1.84</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (24kHz/12kbps)</td><td>23.77&plusmn;0.84</td><td class="right-border">18.94&plusmn;1.47</td><td>42.99&plusmn;1.05</td><td class="right-border">31.88&plusmn;0.79</td><td>54.02&plusmn;1.17</td><td class="right-border">52.75&plusmn;1.11</td><td>48.88&plusmn;1.41</td><td>45.94&plusmn;1.78</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (24kHz/24kbps)</td><td>23.98&plusmn;0.73</td><td class="right-border">19.25&plusmn;1.47</td><td>43.00&plusmn;0.72</td><td class="right-border">31.81&plusmn;0.85</td><td>53.94&plusmn;1.13</td><td class="right-border">52.87&plusmn;1.14</td><td>48.52&plusmn;1.39</td><td>45.77&plusmn;2.14</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (48kHz/3kbps)</td><td>24.00&plusmn;1.03</td><td class="right-border">19.64&plusmn;0.60</td><td>43.10&plusmn;0.98</td><td class="right-border">37.27&plusmn;0.76</td><td>55.25&plusmn;1.47</td><td class="right-border">53.94&plusmn;1.11</td><td>51.82&plusmn;1.78</td><td>47.50&plusmn;1.98</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (48kHz/6kbps)</td><td>23.89&plusmn;1.15</td><td class="right-border">19.04&plusmn;0.79</td><td>42.80&plusmn;1.08</td><td class="right-border">36.08&plusmn;0.57</td><td>55.34&plusmn;1.09</td><td class="right-border">54.30&plusmn;1.14</td><td>52.74&plusmn;1.59</td><td>47.99&plusmn;1.52</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (48kHz/12kbps)</td><td>23.27&plusmn;1.13</td><td class="right-border">19.06&plusmn;1.54</td><td>42.55&plusmn;1.02</td><td class="right-border">34.78&plusmn;0.85</td><td>54.98&plusmn;1.12</td><td class="right-border">53.84&plusmn;0.80</td><td>52.57&plusmn;1.40</td><td>46.72&plusmn;1.84</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">EnCodec (48kHz/24kbps)</td><td>23.42&plusmn;1.09</td><td class="right-border">19.67&plusmn;1.40</td><td>42.60&plusmn;0.94</td><td class="right-border">34.77&plusmn;1.33</td><td>54.42&plusmn;0.96</td><td class="right-border">53.44&plusmn;0.96</td><td>52.40&plusmn;1.59</td><td>44.63&plusmn;1.75</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">DAC</td><td>23.33&plusmn;1.06</td><td class="right-border">19.10&plusmn;0.93</td><td>42.73&plusmn;0.81</td><td class="right-border">39.63&plusmn;0.96</td><td>54.79&plusmn;1.35</td><td class="right-border">55.06&plusmn;0.83</td><td>50.34&plusmn;1.76</td><td>50.21&plusmn;1.42</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Fine-tuning (Audio Tagging) after MLM</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE(Huang)</td><td>44.26&plusmn;0.70</td><td class="right-border">38.41&plusmn;1.34</td><td>57.23&plusmn;0.89</td><td class="right-border">59.14&plusmn;0.42</td><td>63.30&plusmn;1.61</td><td class="right-border">63.95&plusmn;1.92</td><td>63.25&plusmn;1.70</td><td>63.14&plusmn;1.64</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE(Zhong)</td><td>37.74&plusmn;1.10</td><td class="right-border">36.50&plusmn;1.25</td><td>53.82&plusmn;0.91</td><td class="right-border">54.31&plusmn;1.14</td><td>62.61&plusmn;1.69</td><td class="right-border">61.53&plusmn;1.39</td><td>61.09&plusmn;2.28</td><td>58.64&plusmn;1.21</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Learning (Audio Tagging)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PANNs</td><td>n/a</td><td class="right-border">26.12&plusmn;0.76</td><td>n/a</td><td class="right-border">46.37&plusmn;0.89</td><td>n/a</td><td class="right-border">59.29&plusmn;0.94</td><td>n/a</td><td>57.55&plusmn;1.59</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PaSST</td><td>28.94&plusmn;1.08</td><td class="right-border">22.00&plusmn;0.96</td><td>45.52&plusmn;0.87</td><td class="right-border">44.06&plusmn;1.20</td><td>59.28&plusmn;1.08</td><td class="right-border">58.39&plusmn;1.56</td><td>57.61&plusmn;1.10</td><td>55.80&plusmn;1.94</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Learning & Fine-tuning (Sound Event Detection)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PANNs</td><td>28.73&plusmn;0.83</td><td class="right-border">23.89&plusmn;0.72</td><td>53.22&plusmn;0.72</td><td class="right-border">46.73&plusmn;0.79</td><td>60.01&plusmn;1.29</td><td class="right-border">57.60&plusmn;1.23</td><td>58.45&plusmn;1.24</td><td>54.90&plusmn;1.06</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Cross-modal Contrastive Learning (Audio-text)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">CLAP (music-audioset)</td><td>n/a</td><td class="right-border">29.21&plusmn;0.96</td><td>n/a</td><td class="right-border">46.60&plusmn;1.30</td><td>n/a</td><td class="right-border">60.36&plusmn;1.08</td><td>n/a</td><td>58.56&plusmn;1.21</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">CLAP (music-speech-audioset)</td><td>n/a</td><td class="right-border">29.29&plusmn;0.92</td><td>n/a</td><td class="right-border">46.50&plusmn;1.17</td><td>n/a</td><td class="right-border">60.46&plusmn;1.19</td><td>n/a</td><td>59.03&plusmn;0.96</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Cross-modal Contrastive Learning (Audio-visual)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">OpenL3</td><td>38.33&plusmn;1.24</td><td class="right-border">22.65&plusmn;0.86</td><td>50.24&plusmn;0.95</td><td class="right-border">44.48&plusmn;1.20</td><td>60.30&plusmn;1.88</td><td class="right-border">60.15&plusmn;1.05</td><td>58.09&plusmn;2.40</td><td>58.45&plusmn;1.23</td>
</tr>
</tbody>
</table>


<table class="manual-border">
<caption><span class="bold-text">Table B.</span> Cross-dataset validaiton of linear probing between Harmonix(test) and RWC(train) datasets.</caption>
<thead>
<tr>
<th rowspan=3 style="text-align:center; vertical-align:middle;" class="right-border">FAE</th><th colspan=4 style="text-align:center; vertical-align:middle;" class="right-border">Boundary detection</th><th style="text-align:center; vertical-align:middle;" colspan=4>Function prediction</th>
</tr>
<tr>
<th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">HR.5F</th><th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">HR3F</th><th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">PW</th><th style="text-align:center; vertical-align:middle;" colspan=2>ACC</th>
</tr>
<tr>
<th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;">Pooling</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Masked Language Modeling (MLM)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MusicFM (FMA)</td><td><span class="underline-text">48.0</span></td><td class="right-border">36.0</td><td>55.8</td><td class="right-border">55.2</td><td><span class="bold-text">62.4</span></td><td class="right-border">60.1</td><td><span class="bold-text">56.4</span></td><td>51.9</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MusicFM (MSD)</td><td><span class="bold-text">49.5</span></td><td class="right-border">45.0</td><td>55.6</td><td class="right-border"><span class="bold-text">59.1</span></td><td><span class="underline-text">61.3</span></td><td class="right-border">59.5</td><td><span class="underline-text">54.9</span></td><td>50.6</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MERT (95M)</td><td>37.5</td><td class="right-border">38.7</td><td>49.6</td><td class="right-border"><span class="underline-text">57.6</span></td><td>57.3</td><td class="right-border">57.9</td><td>49.4</td><td>47.7</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MERT (330M)</td><td>36.6</td><td class="right-border">32.2</td><td>48.3</td><td class="right-border">49.6</td><td>58.7</td><td class="right-border">58.1</td><td>50.7</td><td>47.9</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Huang)</td><td>30.6</td><td class="right-border">30.0</td><td>46.4</td><td class="right-border">50.9</td><td>54.9</td><td class="right-border">59.0</td><td>49.2</td><td>48.8</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Zhong)</td><td>38.6</td><td class="right-border">44.7</td><td>54.2</td><td class="right-border">57.5</td><td>59.5</td><td class="right-border">59.0</td><td>50.4</td><td>48.2</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Constrastive Learning</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MULE</td><td>17.3</td><td class="right-border">n/a</td><td>41.3</td><td class="right-border">n/a</td><td>55.4</td><td class="right-border">n/a</td><td>48.5</td><td>n/a</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Tokenization (Codec)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/3kbps)</td><td>23.5</td><td class="right-border">16.8</td><td>42.6</td><td class="right-border">30.9</td><td>54.2</td><td class="right-border">53.9</td><td>47.0</td><td>44.7</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/6kbps)</td><td>23.3</td><td class="right-border">17.4</td><td>42.7</td><td class="right-border">31.3</td><td>54.1</td><td class="right-border">53.9</td><td>46.9</td><td>43.3</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/12kbps)</td><td>23.0</td><td class="right-border">16.5</td><td>42.6</td><td class="right-border">31.0</td><td>53.9</td><td class="right-border">53.7</td><td>46.0</td><td>43.7</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/24kbps)</td><td>23.3</td><td class="right-border">17.1</td><td>42.6</td><td class="right-border">31.0</td><td>54.1</td><td class="right-border">53.5</td><td>46.2</td><td>43.3</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/3kbps)</td><td>24.5</td><td class="right-border">17.8</td><td>43.0</td><td class="right-border">36.2</td><td>55.6</td><td class="right-border">54.3</td><td>51.3</td><td>46.7</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/6kbps)</td><td>23.5</td><td class="right-border">16.2</td><td>42.6</td><td class="right-border">35.8</td><td>55.7</td><td class="right-border">54.4</td><td>51.3</td><td>46.9</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/12kbps)</td><td>22.5</td><td class="right-border">15.9</td><td>42.0</td><td class="right-border">35.1</td><td>55.5</td><td class="right-border">54.1</td><td>50.7</td><td>47.1</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/24kbps)</td><td>22.0</td><td class="right-border">16.0</td><td>41.7</td><td class="right-border">34.7</td><td>55.4</td><td class="right-border">54.2</td><td>51.2</td><td>47.3</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">DAC</td><td>21.2</td><td class="right-border">17.1</td><td>41.0</td><td class="right-border">38.9</td><td>53.8</td><td class="right-border">53.5</td><td>44.1</td><td>41.7</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Fine-tuning (Audio Tagging) after MLM</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Huang)</td><td>34.9</td><td class="right-border">24.3</td><td>49.0</td><td class="right-border">45.6</td><td>55.3</td><td class="right-border">56.2</td><td>44.6</td><td>45.2</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Zhong)</td><td>31.1</td><td class="right-border">26.7</td><td>48.1</td><td class="right-border">45.3</td><td>58.2</td><td class="right-border">57.4</td><td>50.1</td><td>46.5</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Learning (Audio Tagging)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PANNs</td><td>n/a</td><td class="right-border">25.2</td><td>n/a</td><td class="right-border">43.0</td><td>n/a</td><td class="right-border">56.3</td><td>n/a</td><td>44.5</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PaSST</td><td>25.4</td><td class="right-border">19.6</td><td>43.5</td><td class="right-border">40.9</td><td>57.2</td><td class="right-border">55.8</td><td>51.6</td><td>48.7</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Learning & Fine-tuning (Sound Event Detection)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PANNs</td><td>26.8</td><td class="right-border">20.8</td><td>50.9</td><td class="right-border">44.8</td><td>58.2</td><td class="right-border">57.2</td><td>47.1</td><td>46.2</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Cross-modal Contrastive Learning (Audio-text)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">CLAP (music-audioset)</td><td>n/a</td><td class="right-border">27.0</td><td>n/a</td><td class="right-border">43.8</td><td>n/a</td><td class="right-border">56.2</td><td>n/a</td><td>44.4</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">CLAP (music-speech-audioset)</td><td>n/a</td><td class="right-border">26.0</td><td>n/a</td><td class="right-border">43.5</td><td>n/a</td><td class="right-border">57.3</td><td>n/a</td><td>48.8</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Cross-modal Contrastive Learning (Audio-visual)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">OpenL3</td><td>29.4</td><td class="right-border">19.3</td><td>43.5</td><td class="right-border">41.6</td><td>55.0</td><td class="right-border">54.8</td><td>44.8</td><td>42.6</td>
</tr>
</tbody>
</table>


<table class="manual-border">
<caption><span class="bold-text">Table C.</span> Cross-dataset validaiton of linear probing between Harmonix(train) and RWC(test) datasets.</caption>
<thead>
<tr>
<th rowspan=3 style="text-align:center; vertical-align:middle;" class="right-border">FAE</th><th colspan=4 style="text-align:center; vertical-align:middle;" class="right-border">Boundary detection</th><th style="text-align:center; vertical-align:middle;" colspan=4>Function prediction</th>
</tr>
<tr>
<th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">HR.5F</th><th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">HR3F</th><th colspan=2 style="text-align:center; vertical-align:middle;" class="right-border">PW</th><th style="text-align:center; vertical-align:middle;" colspan=2>ACC</th>
</tr>
<tr>
<th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;" class="right-border">Pooling</th><th style="text-align:center; vertical-align:middle;">No pooling</th><th style="text-align:center; vertical-align:middle;">Pooling</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Masked Language Modeling (MLM)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MusicFM (FMA)</td><td><span class="underline-text">55.4</span></td><td class="right-border">41.7</td><td>67.3</td><td class="right-border">64.8</td><td>63.3</td><td class="right-border">60.7</td><td><span class="underline-text">60.3</span></td><td>55.4</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MusicFM (MSD)</td><td><span class="bold-text">59.3</span></td><td class="right-border">53.1</td><td><span class="bold-text">69.5</span></td><td class="right-border"><span class="underline-text">68.9</span></td><td><span class="bold-text">66.8</span></td><td class="right-border">64.5</td><td><span class="bold-text">61.1</span></td><td>57.4</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MERT (95M)</td><td>46.8</td><td class="right-border">42.3</td><td>60.5</td><td class="right-border">66.3</td><td>60.6</td><td class="right-border">62.8</td><td>52.9</td><td>54.0</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MERT (330M)</td><td>48.0</td><td class="right-border">36.2</td><td>61.2</td><td class="right-border">60.6</td><td>62.3</td><td class="right-border">61.2</td><td>53.2</td><td>52.9</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Huang)</td><td>38.6</td><td class="right-border">36.4</td><td>59.5</td><td class="right-border">61.7</td><td>56.5</td><td class="right-border"><span class="underline-text">64.5</span></td><td>53.4</td><td>53.9</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Zhong)</td><td>42.8</td><td class="right-border">50.6</td><td>64.7</td><td class="right-border">65.3</td><td>61.8</td><td class="right-border">62.5</td><td>56.1</td><td>49.2</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Constrastive Learning</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">MULE</td><td>19.7</td><td class="right-border">n/a</td><td>46.8</td><td class="right-border">n/a</td><td>55.4</td><td class="right-border">n/a</td><td>51.4</td><td>n/a</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Self-supervised Learning: Tokenization (Codec)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/3kbps)</td><td>27.6</td><td class="right-border">17.8</td><td>50.2</td><td class="right-border">29.6</td><td>50.6</td><td class="right-border">50.1</td><td>47.2</td><td>43.9</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/6kbps)</td><td>28.4</td><td class="right-border">17.5</td><td>51.9</td><td class="right-border">34.3</td><td>51.0</td><td class="right-border">50.6</td><td>47.8</td><td>45.6</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/12kbps)</td><td>28.2</td><td class="right-border">17.7</td><td>51.5</td><td class="right-border">33.9</td><td>50.4</td><td class="right-border">49.9</td><td>47.3</td><td>44.5</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (24kHz/24kbps)</td><td>28.0</td><td class="right-border">18.1</td><td>51.8</td><td class="right-border">34.1</td><td>50.1</td><td class="right-border">50.4</td><td>47.1</td><td>44.5</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/3kbps)</td><td>28.4</td><td class="right-border">19.5</td><td>51.1</td><td class="right-border">37.1</td><td>50.0</td><td class="right-border">49.1</td><td>48.0</td><td>40.4</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/6kbps)</td><td>28.0</td><td class="right-border">17.9</td><td>52.6</td><td class="right-border">32.2</td><td>49.9</td><td class="right-border">49.3</td><td>47.8</td><td>38.6</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/12kbps)</td><td>25.9</td><td class="right-border">16.6</td><td>51.7</td><td class="right-border">32.9</td><td>50.2</td><td class="right-border">50.4</td><td>47.7</td><td>40.1</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">Codec (48kHz/24kbps)</td><td>26.1</td><td class="right-border">17.8</td><td>51.5</td><td class="right-border">32.3</td><td>49.4</td><td class="right-border">49.6</td><td>47.8</td><td>38.6</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">DAC</td><td>26.9</td><td class="right-border">20.5</td><td>53.0</td><td class="right-border">39.3</td><td>51.0</td><td class="right-border">50.8</td><td>48.6</td><td>46.1</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Fine-tuning (Audio Tagging) after MLM</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Huang)</td><td>45.4</td><td class="right-border">35.1</td><td>64.5</td><td class="right-border">63.8</td><td>62.8</td><td class="right-border">63.5</td><td>53.8</td><td>51.2</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">AudioMAE (Zhong)</td><td>38.7</td><td class="right-border">34.7</td><td>60.5</td><td class="right-border">56.1</td><td>61.3</td><td class="right-border">58.8</td><td>55.3</td><td>49.1</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Learning (Audio Tagging)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PANNs</td><td>n/a</td><td class="right-border">27.1</td><td>n/a</td><td class="right-border">55.9</td><td>n/a</td><td class="right-border">64.0</td><td>n/a</td><td>54.7</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PaSST</td><td>34.9</td><td class="right-border">22.9</td><td>52.7</td><td class="right-border">47.6</td><td>58.0</td><td class="right-border">56.7</td><td>49.9</td><td>50.7</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Supervised Learning & Fine-tuning (Sound Event Detection)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">PANNs</td><td>29.5</td><td class="right-border">25.2</td><td>60.1</td><td class="right-border">47.4</td><td>59.4</td><td class="right-border">56.7</td><td>51.7</td><td>48.1</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Cross-modal Contrastive Learning (Audio-text)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">CLAP (music-audioset)</td><td>n/a</td><td class="right-border">28.4</td><td>n/a</td><td class="right-border">51.5</td><td>n/a</td><td class="right-border">63.1</td><td>n/a</td><td>52.3</td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">CLAP (music-speech-audioset)</td><td>n/a</td><td class="right-border">29.0</td><td>n/a</td><td class="right-border">52.7</td><td>n/a</td><td class="right-border">61.6</td><td>n/a</td><td>52.3</td>
</tr>
<tr>
<td colspan=9 style="background-color: lightgray;"><i>Cross-modal Contrastive Learning (Audio-visual)</i></td>
</tr>
<tr style="text-align:center; vertical-align:middle;">
<td class="right-border">OpenL3</td><td>42.4</td><td class="right-border">21.5</td><td>53.4</td><td class="right-border">28.9</td><td>52.6</td><td class="right-border">47.8</td><td>45.0</td><td>37.7</td>
</tr>
</tbody>
</table>
</body>


## Reference for the supplimental experiments
* Goto et al., "RWC Music Database: Popular, Classical and Jazz Music Databases," in Proceedings of ISMIR, 2002
* Goto, "AIST Annotation for the RWC Music Database," in Proceedings of ISMIR, 2006


## Citation
```
@inproceedings{toyama2026icassp,
    author={Keisuke Toyama and Zhi Zhong and Akira Takahashi and Shusuke Takahashi and Yuki Mitsufuji},
    title={Do Foundational Audio Encoders Understand Music Structure?},
    booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing},
    year={2026}
}
```

## Contact
* Keisuke Toyama (keisuke.toyama@sony.com)
