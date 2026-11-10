"""Standalone 3-D fusion t-SNE experiment.
Does not modify the existing 2-D pipeline or its outputs.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from visualization.t_sne.stage_pipeline import (
    STAGES, SPACE_LABELS, PathsConfig, load_feature_set, feature_cache_path,
    get_task, l2, predict_logits, spaces, make_jobs, load_public_splits,
)
from linear_probe.head_io import load_head
from linear_probe.config import TASK_ORDER

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--config-dir', default='configs')
    ap.add_argument('--task', default='all', choices=['all']+list(TASK_ORDER))
    ap.add_argument('--out', default='visualization/t_sne/outputs/seed42cls_3d_fusion')
    ap.add_argument('--perplexity', type=float, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    paths=PathsConfig.from_env(); tasks=list(TASK_ORDER) if args.task=='all' else [args.task]
    cohorts={t:load_public_splits(get_task(t)) for t in tasks}
    # Reuse the exact cohorts and samples recorded by the existing run.
    root2=Path('visualization/t_sne/outputs/seed42cls')
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for stage in list(STAGES.keys())[1:]:
      for task in tasks:
        man=json.loads((root2/stage/task/'manifest.json').read_text(encoding='utf-8'))
        ids=np.asarray(man['ids']); y=np.asarray(man['labels']); names=man['class_names']
        cohort=cohorts[task]; jobs=[j for j in make_jobs(argparse.Namespace(weights_dir=args.weights_dir, config_dir=args.config_dir, amp='bf16', ratio=1.0, probe_budget='melp', probe_seed=42, dataset='all'),paths,{task:cohort}) if j['stage']==stage and j['mode']=='fusion']
        if len(jobs)!=1: raise RuntimeError(f'fusion job missing: {stage}/{task}')
        job=jobs[0]
        cache=feature_cache_path(paths.cache_dir,get_task(task).family,'fusion',job['identity']['encoder_tag'],job['identity']['preprocess_tag'],'cls')
        fs=load_feature_set(cache); X=fs.rows_for(ids).numpy().astype(np.float64)
        head,payload=load_head(job['head'],job['identity'],X.shape[1]); auc=float(payload['test_metrics']['macro_auroc'])
        transformed,_=spaces(X,fs,cohort,head,job,out/stage/task/'ridge'/'fusion')
        for space, F in transformed.items():
          Z=TSNE(n_components=3, perplexity=args.perplexity, random_state=args.seed, init='pca', learning_rate='auto', max_iter=1000).fit_transform(F)
          sil=float(silhouette_score(Z,y)); d=out/stage/task/space; d.mkdir(parents=True,exist_ok=True)
          np.savez_compressed(d/'fusion_3d.npz',coordinates=Z,labels=y,sample_ids=ids)
          fig=plt.figure(figsize=(11,9)); ax=fig.add_subplot(111,projection='3d')
          cmap=plt.get_cmap('tab20',len(names)); present=np.unique(y)
          for i in present:
            m=y==i; ax.scatter(Z[m,0],Z[m,1],Z[m,2],s=12,alpha=.72,color=cmap(int(i)%20),label=names[int(i)])
          ax.set_title(f'{stage} | {task} | fusion {space} | 3D t-SNE\\nBCE AUROC {auc*100:.2f}% | 3D silhouette {sil:.3f} | perplexity {args.perplexity:g} | seed {args.seed}')
          ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.set_zlabel('t-SNE 3')
          ax.legend(fontsize=7,ncol=3,loc='upper left',bbox_to_anchor=(1.02,1)); fig.tight_layout(); fig.savefig(d/'fusion_3d.png',dpi=200,bbox_inches='tight'); plt.close(fig)
          try:
            import plotly.express as px
            figi=px.scatter_3d(x=Z[:,0],y=Z[:,1],z=Z[:,2],color=[names[int(i)] for i in y],hover_data={'sample_id':ids},title=f'{stage} | {task} | fusion {space} | 3D t-SNE')
            figi.write_html(str(d/'fusion_3d.html'),include_plotlyjs='cdn')
          except ImportError: pass
          print(stage,task,space,'n=',len(y),'silhouette=',sil,'auc=',auc,flush=True)
if __name__=='__main__': main()
