import sys,os,glob,numpy as np,librosa,essentia.standard as es
S=sys.argv[1]; d=S+'/set4'
NAMES=['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']
PC={'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
KS={'major':np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]),
    'minor':np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])}
rows=[]
for f in sorted(glob.glob(d+'/*.mp3')):
    try:
        a=es.MonoLoader(filename=f,sampleRate=44100)()
        est=[]
        for prof in ('edma','bgate'):
            k,sc,st=es.KeyExtractor(profileType=prof)(a); est.append((PC[k],sc))
        ch=librosa.feature.chroma_cqt(y=a[::2],sr=22050).mean(axis=1)
        best=max(((np.corrcoef(ch,np.roll(KS[m],t))[0,1],t,m) for t in range(12) for m in KS))
        est.append((best[1],best[2]))
    except Exception as e:
        print(os.path.basename(f),'ERR',repr(e)[:80],flush=True); continue
    agree=est[0]==est[1]==est[2]
    name=f'{NAMES[est[0][0]]} {est[0][1]}'
    print(os.path.basename(f),est,'CONSENSO' if agree else '-',flush=True)
    if agree: rows.append((os.path.basename(f),name))
open(d+'/expected.csv','w').write(''.join(f'{a},{b}\n' for a,b in rows)); print('consenso',len(rows))
