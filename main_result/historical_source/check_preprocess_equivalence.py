import sys, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, "revision/code")
import run_selection_bootstrap as bs
from threadpoolctl import threadpool_limits

D = pd.read_csv("private_data/development.csv")
E = pd.read_csv("private_data/evaluation.csv")
a, b, c = bs.prepare(D, E)
with threadpool_limits(limits=1):
    for algo, cfg, blocks in [
        ("ridge", 0.1, ["H", "D", "B", "C"]),
        ("hgb", {"max_leaf_nodes": 3, "learning_rate": 0.05}, ["H", "D", "B", "C"]),
        ("ridge", 0.01, bs.BLOCK_ORDER),
        ("hgb", {"max_leaf_nodes": 7, "learning_rate": 0.05}, ["B", "A"]),
    ]:
        cc = bs.select_columns(blocks, c)
        p = bs.fit_array(
            algo, cfg, bs.eng.SEED, a[:, cc], D.outcome_any_use.to_numpy(int), b[:, cc]
        )
        q, _ = bs.eng.fit_predict(D, E, blocks, algo, cfg)
        print(algo, blocks, np.max(abs(p - q)))
        assert np.max(abs(p - q)) < 1e-10
