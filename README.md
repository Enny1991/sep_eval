# SEP_EVAL: a collection of speech enhancement measures [![Build Status](https://travis-ci.com/Enny1991/sep_eval.svg?branch=master)](https://travis-ci.com/Enny1991/sep_eval.svg?branch=master)

This small library collects some measures for source separation and speech enhancement.
The library contains 
    - SDR
    - STOI
    - Extended STOI
    - PESQ (see below for details on installation)
    - BSS_Eval measures 

Most of them are already present as python packages. This library only collects them and allows to 
compute all of them in one single step even on batched data.  

## Installation
Simply
```bash
pip install sep_eval
```
or directly from here
```bash
git clone https://github.com/Enny1991/sep_eval
cd sep_eval
python setup.py install
``` 

## PESQ
Unfortunately, to get pesq scores, there is not other way than using the binary files from [ITU](https://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en).
Follow the next steps:

Download the zip file from [here](https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200511-I!Amd2!SOFT-ZST-E&type=items)
```bash
unzip T-REC-P.862-200511-I\!Amd2\!SOFT-ZST-E
cd Software
unzip P862_annex_A_2005_CD\ \ wav\ final.zip 
cd P862_annex_A_2005_CD/source/
gcc -o PESQ *.c -lm
```     
The compilation will produce a unique file called ```pesq``` in this folder.
Add this location to ```$PATH```

## Usage
Simply call the desired measure (or full_eval) with degraded and reference signals. 
Degraded and reference can also be arrays or lists and you can choose to get a full result or an average across all the given samples (see [example1](examples/example1.py) and [example2](examples/example2.py) for more details) 
```python
import sep_eval
import soundfile as sf

degraded, fs = sf.read('../wavs/degraded.wav')
reference, _ = sf.read('../wavs/reference1.wav')


eval_deg = sep_eval.full_eval(degraded, reference)  # returns a dictionary will all the measures

pesq_deg = sep_eval.pesq(degraded, reference)

stoi_deg = sep_eval.stoi(degraded, reference)
```


