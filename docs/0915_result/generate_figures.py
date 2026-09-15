"""Rebuild report evidence and PNG figures on CPU; never runs model inference."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from controlled_degree2.compare_quadratic_checkpoints import extract_state_dict, activation_data
OUT = Path(__file__).resolve().parent
FARS = np.logspace(-6, -1, 6)
MODELS = {
 'ijbc_calibrated': ('IJB-C calibrated', 'work_dirs/channelwise_template_supervised01_20260915/ijbc_full/evaluated_checkpoint.pt', 'work_dirs/channelwise_template_supervised01_20260915/ijbc_full/pair_geometry'),
 'no_ijbc_head': ('MS1MV3 head-only v3', 'work_dirs/controlled_degree2_accuracy_recovery_head_only_v3_20260908/train/student_best.pt', 'work_dirs/controlled_degree2_accuracy_recovery_head_only_v3_20260908/ijbc_epoch1/result/controlled_d2'),
 'prelu_baseline': ('Original PReLU reference', 'work_dirs/ms1mv3_r50/model.pt', 'work_dirs/channelwise_prelu_verified_20260915/ijbc_full/original_prelu'),
}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
labels_path=ROOT/'ijb/IJBC/meta/ijbc_template_pair_label.txt'
labels=np.loadtxt(labels_path,usecols=2,dtype=np.int8)
assert len(labels)==15658489 and labels.sum()==19557
metrics={}; coeffrows=[]; summaries={}
for key,(label,checkpoint,result) in MODELS.items():
 print('Processing',key,flush=True)
 p=ROOT/result; scores=np.load(p/'ijbc.npy',mmap_mode='r')
 assert scores.shape==labels.shape and np.isfinite(scores).all()
 fpr,tpr,thr=roc_curve(labels,scores)
 fpr,tpr,thr=fpr[::-1],tpr[::-1],thr[::-1]
 points=[]
 original=list(csv.reader((p/'ijbc_tar_at_far.csv').open()))[1][1:]
 for far,rounded in zip(FARS,original):
  n=int(np.argmin(abs(fpr-far))); valid=np.where(fpr<=far)[0]; c=valid[np.argmax(tpr[valid])]
  assert f'{tpr[n]*100:.2f}'==rounded
  points.append(dict(requested_far=float(far),nearest_tar_percent=float(tpr[n]*100),nearest_actual_far=float(fpr[n]),tar_percent=float(tpr[c]*100),actual_far=float(fpr[c]),threshold=float(thr[c])))
 metrics[key]=dict(label=label,checkpoint=checkpoint,checkpoint_sha256=sha(ROOT/checkpoint),scores=str((p/'ijbc.npy').relative_to(ROOT)),scores_sha256=sha(p/'ijbc.npy'),points=points)
 if key=='prelu_baseline': continue
 payload=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
 state=extract_state_dict(payload); acts=activation_data(state)
 assert len(acts)==25 and sum(a.channels for a in acts.values())==5888
 assert all(np.all(a.coeffs[:,2]!=0) for a in acts.values())
 # Fourth storage column must be EXACTLY zero, rather than approximately zero.
 assert all(v.shape[1]==3 or torch.count_nonzero(v[:,3])==0 for k,v in state.items() if k.endswith('.coeffs'))
 for name,a in acts.items():
  for i in range(a.channels):
   coeffrows.append([key,name,i,*map(float,a.coeffs[i]),float(a.lam_fit[i]),float(a.lam_reg[i]),float(a.slope[i])])
 summaries[key]=dict(sites=25,channels=5888,coefficients=17664,nonzero_curvatures=5888,lam_fit_min=float(min(a.lam_fit.min() for a in acts.values())),lam_fit_max=float(max(a.lam_fit.max() for a in acts.values())))
 # One actual channel per site, chosen by median approximation radius.
 fig,axs=plt.subplots(5,5,figsize=(20,17),constrained_layout=True)
 for ax,(name,a) in zip(axs.flat,acts.items()):
  i=int(np.argsort(a.lam_fit,kind='stable')[a.channels//2]); r=float(a.lam_fit[i]); c0,c1,c2=map(float,a.coeffs[i]); slope=float(a.slope[i]); x=np.linspace(-2*r,2*r,401)
  ax.axvspan(-r,r,color='#deecf5',alpha=.7)
  ax.plot(x,c0+c1*x+c2*x*x,color='#007c91',label='Saved quadratic')
  ax.plot(x,np.where(x>=0,x,slope*x),'--',color='#b74f32',label='Original PReLU target')
  ax.axhline(0,color='gray',lw=.4); ax.axvline(0,color='gray',lw=.4)
  ax.set_title(f'{name} | channel {i}',fontsize=10)
  ax.text(.03,.96,f'q = {c0:.4g} {c1:+.4g}x {c2:+.4g}x²\nR = {r:.4g}; slope = {slope:.4g}',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
  ax.ticklabel_format(axis='both',style='sci',scilimits=(-3,3));ax.tick_params(labelsize=8)
 axs[0,0].legend(loc='lower right',fontsize=8)
 fig.suptitle(label+' | actual quadratic at all 25 sites\nShaded: fitting interval [-R, R]; curves extend to [-2R, 2R] without clipping',fontsize=17)
 fig.savefig(OUT/f'quadratics_{key}.png');plt.close(fig)
 # All channels in interval-normalized coordinates, no misleading shared raw x.
 fig,axs=plt.subplots(5,5,figsize=(17,14),constrained_layout=True);u=np.linspace(-2,2,301)
 for ax,(name,a) in zip(axs.flat,acts.items()):
  r=a.lam_fit[:,None];x=r*u;y=(a.coeffs[:,0,None]+a.coeffs[:,1,None]*x+a.coeffs[:,2,None]*x*x)/r
  q=np.quantile(y,[.1,.5,.9],axis=0)
  ax.fill_between(u,q[0],q[2],color='#007c91',alpha=.2);ax.plot(u,q[1],color='#007c91')
  ax.axvline(-1,ls=':',color='gray');ax.axvline(1,ls=':',color='gray');ax.set_title(name,fontsize=10)
 fig.suptitle(label+' | all-channel shapes: u=x/R, v=q(x)/R\nMedian and P10-P90 band (not a single shared activation)',fontsize=15)
 fig.savefig(OUT/f'channel_bands_{key}.png');plt.close(fig)
 if key=='no_ijbc_head': dump('head_training_config.json',payload['train_config'])
 del payload,state,acts
with (OUT/'coefficients.csv').open('w') as f:
 w=csv.writer(f, lineterminator="\n");w.writerow(['model','site','channel','c0','c1','c2','lam_fit','lam_reg','original_prelu_slope']);w.writerows(coeffrows)
dump('metrics.json',dict(label_path=str(labels_path.relative_to(ROOT)),label_sha256=sha(labels_path),models=metrics,coefficient_summary=summaries))
with (OUT/'tar_vs_far.csv').open('w') as f:
 w=csv.writer(f, lineterminator="\n");w.writerow(['model','requested_far','tar_percent','actual_far','threshold','nearest_tar_percent','nearest_actual_far'])
 for key,m in metrics.items():
  for p in m['points']: w.writerow([key,*[p[n] for n in ['requested_far','tar_percent','actual_far','threshold','nearest_tar_percent','nearest_actual_far']]])
for convention in ['strict','nearest']:
 fig,ax=plt.subplots(figsize=(13,3.6));ax.axis('off');field='tar_percent' if convention=='strict' else 'nearest_tar_percent'
 data=[[m['label'],*[f'{p[field]:.4f}' for p in m['points']]] for m in metrics.values()]
 table=ax.table(cellText=data,colLabels=['Model / TAR (%)',*[f'{x:.0e}' for x in FARS]],loc='center',cellLoc='center',colWidths=[.36]+[.105]*6)
 table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.15)
 for (r,c),cell in table.get_celld().items():
  cell.set_edgecolor('white');cell.set_facecolor('#163c51' if r==0 else ('#e9f2f5' if r%2 else '#f4f6f8'))
  if r==0:cell.get_text().set_color('white')
 ax.set_title('IJB-C TAR vs FAR | '+('actual FAR <= requested FAR' if convention=='strict' else 'nearest ROC point (historical CSV convention)'),pad=22)
 fig.text(.04,.03,'Calibrated: fitting-set performance. MS1MV3 v3: documented no IJB-C fitting.\nBaseline is a reference only. Models are selected by TAR at FAR=1e-4, not independently at every FAR.',fontsize=9)
 fig.savefig(OUT/f'tar_vs_far_{convention}.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
for key,m in metrics.items():ax.semilogx(FARS,[p['tar_percent'] for p in m['points']],marker='o',label=m['label'])
ax.set(xlabel='Requested FAR (actual FAR <= requested)',ylabel='TAR (%)',title='IJB-C verification operating points');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'tar_vs_far_curve.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(14,8));ax.set(xlim=(0,14),ylim=(0,10));ax.axis('off')
def box(x,y,text,color='#e4f0f4'):
 ax.text(x,y,text,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.65',facecolor=color,edgecolor='#597b8c'))
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.7,color='#597b8c'))
box(7,9.2,'Original MS1MV3 PReLU iResNet50 teacher');arrow(6,8.8,3.4,8);arrow(8,8.8,10.3,8)
box(3.4,7.4,'No IJB-C fitting: controlled degree-2\nMS1MV3 distillation + run10 interval buffers');box(10.3,7.4,'Fresh MS1MV3 histogram fit\nPReLU -> 25 channelwise quadratics')
arrow(3.4,6.8,3.4,6);arrow(10.3,6.8,10.3,6)
box(3.4,5.4,'Progressive conversion / tail refinements\nTeacher cosine + hints + range penalty');box(10.3,5.4,'ArcFace + KD + range / prefix repair\nMain training snapshot: epoch14')
arrow(3.4,4.8,3.4,4);arrow(10.3,4.8,10.3,4)
box(3.4,3.4,'Head-only v3: freeze stem + layer1-4\n5 x teacher cosine loss; selected epoch0');box(10.3,3.4,'IJB-C numerical repair + exact head KD\nImage-pair geometry -> supervised templates')
arrow(3.4,2.8,3.4,2);arrow(10.3,2.8,10.3,2)
box(3.4,1.4,'Unclipped IJB-C: zero bad embeddings\nTAR 93.42% (nearest ROC)', '#d9eedf');box(10.3,1.4,'Unclipped exported graph: full finite audit\nTAR 96.0628% (calibration set)', '#d9eedf')
ax.set_title('Training paths and IJB-C evaluation',fontsize=19);fig.savefig(OUT/'training_flow.png',bbox_inches='tight');plt.close(fig)
print('All score/CSV, pair-count, coefficient and plotting checks passed.',flush=True)
