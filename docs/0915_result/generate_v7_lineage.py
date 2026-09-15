"""Plot recorded lineage metrics; no training or inference is performed."""
from pathlib import Path
import csv,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
D=ROOT/'work_dirs'
prefix='controlled_degree2_tail_ms1mv3_'
def ijbc(run,sub):
 p=next((D/(prefix+run)/sub).rglob('ijbc_tar_at_far.csv'))
 rows=list(csv.reader(p.open()));tar=float(rows[1][rows[0].index('0.0001')])
 logs=list((D/(prefix+run)/sub).glob('*.out'))
 matches=[(log,int(x)) for log in logs for x in re.findall(r'Non-finite augmented embedding rows: (\d+)',log.read_text())]
 assert len(matches)==1
 return dict(tar=tar,nonfinite=matches[0][1],csv=str(p.relative_to(ROOT)),log=str(matches[0][0].relative_to(ROOT)))
def ms(sub):
 p=D/(prefix+'deployment_v7_20260908')/sub/'manifest.json';d=json.load(p.open())
 assert d['dataset_source_rows']==5179510 and d['both_orientations']
 return dict(nonfinite=len(d['output_nonfinite']),manifest=str(p.relative_to(ROOT)),checkpoint=d['checkpoint'])
polish=ijbc('20260907','ijbc_controlled');v4=ijbc('narrow_guard_v4_20260908','ijbc_epoch1');v7=ijbc('deployment_v7_20260908','ijbc_final')
v5e=ijbc('causal_fixed_v5_20260908','ijbc_epoch1');v5f=ijbc('causal_fixed_v5_20260908','ijbc_final')
rows=[
 ['PReLU teacher','Reference','--','--','0','96.56'],
 ['Progressive final','8','99.7000','--','--','--'],
 ['Polish final','3','99.7500','--',str(polish['nonfinite']),f"{polish['tar']:.2f}*"],
 ['v4 best (= epoch1)','1 of 3','99.8000','--',str(v4['nonfinite']),f"{v4['tar']:.2f}*"],
 ['v5 best (actual v7 source)','2','99.8000',str(ms('mine')['nonfinite']),'Not completed','Not completed'],
 ['v7 final','1','99.7333',str(ms('remine')['nonfinite']),str(v7['nonfinite']),f"{v7['tar']:.2f}*"],
]
evidence=dict(rows=rows,polish=polish,v4=v4,v7=v7,v5_other_evaluations=dict(epoch1=v5e,final=v5f),v5_ms=ms('mine'),v7_ms=ms('remine'),notes=['IJBC nearest ROC TAR at requested FAR=1e-4; historical convention.','Nonfinite counts are augmented embedding rows, not scalar values.','v5 best evaluation job 358339 was cancelled; final and epoch1 differ from current best.','LFW final-stage numbers are from respective epoch logs, not best_canary metadata.','Checkpoint comparisons: v4 best equals epoch1; v5 best differs from final in 370 tensors and epoch1 in 449 tensors.'])
(OUT/'v7_lineage.json').write_text(json.dumps(evidence,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(15,5.4));ax.axis('off')
t=ax.table(cellText=rows,colLabels=['Actual source-chain checkpoint','Stage epochs','LFW (%)','MS1MV3 NF','IJB-C NF','IJB-C TAR (%)'],loc='center',cellLoc='center',colWidths=[.32,.12,.13,.13,.15,.15]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.2)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor('white');cell.set_facecolor('#163c51' if r==0 else ('#e9f2f5' if r%2 else '#f4f6f8'))
 if r==0:cell.get_text().set_color('white')
ax.set_title('PReLU teacher -> quadratic conversion -> deployment v7',fontsize=18,pad=18)
fig.text(.07,.04,'NF = non-finite original/flip embedding rows. -- = no matching full audit/result found.\n* Diagnostic TAR after non-finite feature sanitization; not a qualifying finite result. FAR=1e-4, nearest ROC convention.\nv5 best is NOT v5 final/epoch1; their IJB-C scores must not be inserted into this source chain.',fontsize=10)
fig.savefig(OUT/'v7_lineage.png',dpi=170,bbox_inches='tight');plt.close(fig)
assert [polish['nonfinite'],v4['nonfinite'],v7['nonfinite']]==[8,9,11]
assert [ms('mine')['nonfinite'],ms('remine')['nonfinite']]==[6,5]
print('PASS: lineage table agrees with saved IJBC CSV/logs and MS1MV3 manifests.')
