#!/usr/bin/env python3
"""Genera 108 progresiones sintéticas (12 tónicas x mayor/menor x 4-5 progresiones, con bajo) para
entrenar el modelo de tonalidad. Las progresiones de los tests (I-vi-IV-V, i-iv-V-i) NO están aquí.
Uso: tools/gen_synth_training.py CARPETA   (escribe wavs y expected.csv)"""
import numpy as np, wave, random, csv
import sys
S=sys.argv[1]  # carpeta de salida
random.seed(5); rng=np.random.default_rng(5); SR=44100
NAMES=['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']
def note(out,start,length,midi,amp,detune):
    f0=440*2**((midi-69+detune/100)/12); n=min(length,len(out)-start); fade=int(0.02*SR)
    i=np.arange(n); env=np.ones(n); env[:fade]=i[:fade]/fade; env[-fade:]=np.minimum(env[-fade:],(n-i[-fade:])/fade)
    s=sum(np.sin(2*np.pi*f0*h*i/SR)/h for h in range(1,7) if f0*h<SR/2.2)
    out[start:start+n]+=amp*env*s
def triad(t,deg,minor): return (48+t+deg,[0,3,7] if minor else [0,4,7])
MAJ={'I':(0,0),'ii':(2,1),'iii':(4,1),'IV':(5,0),'V':(7,0),'vi':(9,1)}
MIN={'i':(0,1),'III':(3,0),'iv':(5,1),'V':(7,0),'VI':(8,0),'VII':(10,0)}
majp=[['I','V','vi','IV'],['I','IV','V','I'],['ii','V','I','I'],['vi','IV','I','V'],['I','IV','I','V']]
minp=[['i','VI','VII','i'],['i','iv','i','V'],['i','VII','VI','V'],['i','III','iv','V']]
rows=[]
for t in range(12):
    for mode,progs,tab in (('major',majp,MAJ),('minor',minp,MIN)):
        for pi,prog in enumerate(progs):
            dur=[random.uniform(1.5,2.5) for _ in prog]; det=random.uniform(-15,15)
            song=np.zeros(int(sum(dur)*2*SR)+SR)
            pos=0
            for loop in range(2):
                for ch,d in zip(prog,dur):
                    deg,mn=tab[ch]; root,iv=triad(t,deg,mn); L=int(d*SR)
                    for k in iv: note(song,pos,L,root+k,0.12,det)
                    note(song,pos,L,root-12,0.18,det)
                    pos+=L
            song+=rng.normal(0,0.003,len(song))
            fn=f'{NAMES[t]}_{mode}_{pi}.wav'
            w=wave.open(f'{S}/{fn}','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(song,-1,1)*32767).astype(np.int16).tobytes()); w.close()
            rows.append((fn,f'{NAMES[t]} {mode}'))
open(S+'/expected.csv','w').write(''.join(f'{a},{b}\n' for a,b in rows)); print(len(rows))
