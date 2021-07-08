# Multi-Task Audio Source Separation (MTASS)

=======================================================================================

## 1. Task Description

This task aims to separate the three fixed types of sound sources from the monaural mixture into three tracks, which are **speech**, **music**, and **background noise**. In detail, the output of the speech track is a normal speaking voice, and the music track signal defined here is a broad category, which may be full songs, vocals, and different accompaniments. Except for music and speech, any other possible background sounds, such as closing doors, animal calls, and some annoying noises, are classified as noise track signals.

------------------------------------------------------------------------------------

## 2. MTASS Dataset Preparation

In this project, we prepare three types of datasets for the generation of the mixed dataset. We also release a python script to preprocess these datasets and generate the MTASS dataset. You can download the needed source datasets from the below links.

### 2.1. Speech Datasets
Aishell-1 and Didi Speech, are used to build the speech source dataset of MTASS.

**[Aishell-1]:** (http://www.openslr.org/33/)

Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng, “Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline,” in *20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment*, 2017, pp. 1–5.

**[Didi Speech]:** (https://outreach.didichuxing.com/research/opendata/)

Tingwei Guo, Cheng Wen, Dongwei Jiang, Ne Luo, Ruixiong Zhang, Shuaijiang Zhao, Wubo Li, Cheng Gong, Wei Zou, Kun Han, et al., “Didispeech: A large scale mandarin speech corpus,” in *ICASSP*, 2021, pp. 6968–6972.

### 2.2. Music Datasets
The demixing secrets dataset (DSD100) of the Signal Separation Evaluation Campaign (SISEC) is used as the music source dataset of MTASS.

**[DSD100]:** (https://sisec.inria.fr/)

N. Ono, Z. Rafii, D. Kitamura, N. Ito, and A. Liutkus, “The 2015 Signal Separation Evaluation Campaign,” in *International Conference on Latent Variable Analysis and Signal Separation*, 2015, pp. 186–190.

### 2.3. Noise Datasets
The noise dataset of the Deep Noise Suppression (DNS) Challenge is used as the noise source dataset of MTASS.

**[DNS-noise]:** (https://github.com/microsoft/DNS-Challenge/)

Chandan K. A. Reddy, Harishchandra Dubey, Vishak Gopal, and et al, “ICASSP 2021 deep noise suppression challenge,” in *ICASSP*, 2021, pp. 6623–6627.

----------------------------------------------------------------------------------------------

## 3. MTASS Baseline Model

To tackle this challenging multi-task separation problem, we also proposed a baseline model to separate different tracks. Since this model works in the complex frequency domain for multi-task audio source separation, we call it “**Complex-MTASSNet**”. The **Complex-MTASSNet** separates the signal of each audio track in the complex domain, and further compensates the leaked residual signal for each track. The framework of this baseline model is shown in Figure 1.


![Fig 1. Complex-MTASSNet](img_url)



### 3.1. Comparison with other models
In this multi-task separation, we have compared the proposed **Complex-MTASSNet** with several well-known baselines in speech enhancement, speech separation, and music source separation, which are [GCRN](https://github.com/JupiterEthan/GCRN-complex), [Conv-TasNet](https://github.com/naplab/Conv-TasNet), [Demucs](https://github.com/facebookresearch/demucs), and [D3Net](https://github.com/sony/ai-research-code/tree/master/d3net).


<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td rowspan="2" style="border:solid windowtext 1.0pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Methods</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-font-kerning:0pt;mso-fareast-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td rowspan="2" style="border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Para.<o:p></o:p></span></p>
  <p class="MsoNormal"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-font-kerning:0pt">(millions)<o:p></o:p></span></p>
  </td>
  <td rowspan="2" style="border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">MAC</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-font-kerning:0pt;mso-fareast-language:EN-US">/S<o:p></o:p></span></p>
  </td>
  <td colspan="4" valign="top" style="border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-font-kerning:0pt">SDRi</span></span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:0pt"> (dB)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Speech<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Music<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Noise<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Ave<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td valign="top" style="border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">GCRN<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">9.88 M<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">2.5 G<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">9.11<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">5.76<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">5.51<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">6.79<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3">
  <td valign="top" style="border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-font-kerning:0pt">Demucs</span></span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">243.32 M<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">5.6 G<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">9.93<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">6.38<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">6.29<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">7.53<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4">
  <td valign="top" style="border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">D3Net<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">7.93 M<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">3.5 G<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">10.55<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">7.64<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">7.79<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">8.66<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5">
  <td valign="top" style="border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Conv-<span class="SpellE">TasNet</span><o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">5.14 M<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">5.2 G<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">11.80<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">8.35<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">8.07<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">9.41<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;mso-yfti-lastrow:yes">
  <td valign="top" style="border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">Complex-<span class="SpellE">MTASSNet</span><o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">28.18 M<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:
  0pt">1.8 G<o:p></o:p></span></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:0pt">12.57<o:p></o:p></span></b></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:0pt">9.86<o:p></o:p></span></b></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:0pt">8.42<o:p></o:p></span></b></p>
  </td>
  <td valign="top" style="border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman&quot;,serif;mso-font-kerning:0pt">10.28<o:p></o:p></span></b></p>
  </td>
 </tr>
</tbody></table>

### 3.2. Listening Demos (will be updated soon...)


-----------------------------------------------------------------------------------

## 4. Usage (code and usage will be updated soon...)





------------------------------------------------------------------------------------

## 5. Copyright and Authors
All rights reserved by Dr. Wind (zhanglu_wind@163.com).
People shall use the source code here only for non-commercial research and educational purposes.













