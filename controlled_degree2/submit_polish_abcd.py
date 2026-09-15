"""Submit a frozen A/B/C/D campaign after a completed real-batch smoke.

This command submits once, records IDs, and exits. It does not monitor jobs.
"""
import argparse,json,os,shutil,subprocess,tarfile
from pathlib import Path
from controlled_degree2.polish_abcd import digest,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--smoke-root',required=True);p.add_argument('--smoke-job',required=True)
    p.add_argument('--record',required=True);a=p.parse_args()
    repo=Path.cwd();smoke=Path(a.smoke_root)
    for arm in 'ABCD':
        c=json.loads((smoke/arm/'completed.json').read_text())
        config=json.loads((smoke/arm/'config.json').read_text())
        assert c['steps']==2 and c['finite_updates'] and config['batch_size']==128 and config['global_batch']==2048
    assert json.loads((smoke/'D/routing_smoke.json').read_text())['finite_gradients']
    state=subprocess.check_output(['sacct','-j',a.smoke_job,'-X','-n','-P','--format=State,ExitCode'],text=True).strip()
    if state!='COMPLETED|0:0':raise RuntimeError('smoke has not completed successfully: '+state)
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    code=output/'source';code.mkdir();inputs=output/'inputs';inputs.mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    archive=output/'source.tar'
    with archive.open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    with tarfile.open(archive) as f:f.extractall(code)
    archive.unlink()
    for name in ['ms1m-retinaface-t1','ijb']:
        # Replace only the newly extracted metadata directory, never source data.
        if (code/name).exists():shutil.rmtree(code/name)
        (code/name).symlink_to(repo/name,target_is_directory=True)
    for source,name in [('work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_final.pt','progressive_final.pt'),('work_dirs/ms1mv3_r50/model.pt','teacher.pt')]:
        shutil.copyfile(repo/source,inputs/name);(inputs/name).chmod(0o444)
    record=dict(output=str(output),code_commit=commit,code_snapshot=str(code),smoke_job=a.smoke_job,
                smoke_root=str(smoke),smoke_state=state,epochs=2,gpus_per_arm=4,max_training_gpus=16,
                checkpoint_sha256=digest(inputs/'progressive_final.pt'),teacher_sha256=digest(inputs/'teacher.pt'),jobs={},monitoring=False)
    def submit(name,script,options,mode):
        env=f'ALL,ABCD_OUTPUT={output},ABCD_CODE_ROOT={code},ABCD_MODE={mode}'
        cmd=['sbatch','--parsable',f'--job-name=polish-abcd-{name}',
             f'--output={output}/{name}-%A_%a.out',f'--error={output}/{name}-%A_%a.err',
             '--export='+env,*options,str(code/'controlled_degree2'/script)]
        result=subprocess.run(cmd,text=True,capture_output=True,check=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('invalid submission response: '+result.stdout)
        record['jobs'][name]=dict(id=job,command=cmd,submission_stderr=result.stderr)
        write(output/'submission.json',record);write(a.record,record)
        print(name,job,flush=True);return job
    baseline=submit('baseline','polish_abcd_eval.slurm',[],'baseline')
    train=submit('train','polish_abcd.slurm',['--array=0-3%4','--dependency=afterok:'+baseline,'--kill-on-invalid-dep=yes'],'train')
    submit('eval','polish_abcd_eval.slurm',['--array=0-3%4','--dependency=aftercorr:'+train,'--kill-on-invalid-dep=yes'],'eval')

if __name__=='__main__':main()
