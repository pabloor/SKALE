import random,os,subprocess,sys,collections
from music21 import corpus, converter, key as m21key
import pretty_midi
random.seed(7)
S=sys.argv[1]; out=S+'/set3'
NAMES=['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']
paths=[p for p in corpus.getCorePaths() if any(x in str(p) for x in ('ryansMammoth','oneills1850','essenFolksong','airdsAirs','miscFolk'))]
random.shuffle(paths)
want=collections.Counter(); rows=[]; PROGS=[40,21,24,0,73,46,25,71]
for p in paths:
    if len(rows)>=60: break
    try:
        sc=converter.parse(str(p))
        if hasattr(sc,'scores') and len(sc.scores): sc=sc.scores[0]
        ks=sc.flatten().getElementsByClass('KeySignature')
        if not len(ks): continue
        n=ks[0].sharps
        major=ks[0].asKey('major'); minor=ks[0].asKey('minor')
        notes=[x for x in sc.flatten().notes if x.isNote]
        if len(notes)<25: continue
        last=notes[-1].pitch.pitchClass
        if last==major.tonic.pitchClass: k=(last,'major')
        elif last==minor.tonic.pitchClass: k=(last,'minor')
        else: continue
        if want[k]>=2: continue
        mid=f'{out}/tmp.mid'; sc.write('midi',fp=mid)
        pm=pretty_midi.PrettyMIDI(mid)
        dur=pm.get_end_time()
        if dur<8 or dur>150: continue
        reps=max(1,int(45//dur)+1)
        prog=PROGS[len(rows)%len(PROGS)]
        for inst in pm.instruments:
            inst.program=prog
            orig=list(inst.notes)
            for r in range(1,reps):
                for nt in orig: inst.notes.append(pretty_midi.Note(nt.velocity,nt.pitch,nt.start+r*dur,nt.end+r*dur))
        i=len(rows); pm.write(f'{out}/{i:02d}.mid')
        subprocess.run(['fluidsynth','-ni','-r','44100','-F',f'{out}/{i:02d}.wav','/usr/share/sounds/sf2/FluidR3_GM.sf2',f'{out}/{i:02d}.mid'],capture_output=True,timeout=120)
        os.remove(f'{out}/{i:02d}.mid')
        want[k]+=1
        rows.append((f'{i:02d}.wav',f'{NAMES[k[0]]} {k[1]}',str(p).split('/corpus/')[1],prog))
        print(rows[-1],flush=True)
    except Exception as e:
        continue
open(out+'/expected.csv','w').write(''.join(f'{a},{b}\n' for a,b,_,_ in rows))
open(out+'/sources.tsv','w').write(''.join('\t'.join(map(str,r))+'\n' for r in rows))
print('done',len(rows))
