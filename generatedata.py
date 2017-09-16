from scipy.stats import truncnorm
import numpy as np
import pandas as pd
import bestresponse as br

def entry(entry_prob):
    rv = np.random.rand()
    if rv < entry_prob:
        return 1
    else:
        return 0

# equi_selectionは3つの均衡がモデルから得られた時に左側の均衡を選択する確率のパラメータ。defaultは0.5
def generate(alpha, beta, delta, equi_selection = 0.5, mean_pop = 1, mean_dist = [1.2, 0.8], sample_size = 10000, tane = 12345):
    sample_size = 10000
    np.random.seed(seed=tane)
    
    # pop
    pop = truncnorm.rvs(0, 5, loc = mean_pop, size = sample_size)

    # dist
    dist = np.zeros((sample_size, 2))
    dist[:, 0] = truncnorm.rvs(0, 5, loc = mean_dist[0], size = sample_size)
    dist[:, 1] = truncnorm.rvs(0, 5, loc = mean_dist[1], size = sample_size)

    # 乱数の箱
    RV = np.random.rand(sample_size)
    
    # 均衡として得られた参入確率をしまう箱
    probs = np.empty((sample_size, 2))
    
    # 実現した均衡の種類を示すインデックスをしまう箱
    # 0: そもそも均衡が一つしかない
    # 1: 3つの均衡が存在して、左側が選ばれた
    # 2: 3つの均衡が存在して、右側が選ばれた
    equi_type = np.empty(sample_size)
    
    for i in range(sample_size):
        equilibria = br.findequi(pop[i], dist[i, :], alpha, beta, delta)

        # 均衡の数で場合分け
        if len(equilibria) == 3:
            if RV[i] < equi_selection:
                probs[i, :] = equilibria[0]
                equi_type[i] = 1
            else:
                probs[i, :] = equilibria[2]
                equi_type[i] = 2

        elif len(equilibria) == 1:
            probs[i, :] = equilibria[0]
            equi_type[i] = 0

        else:
            probs[i, :] = [-1, -1]
            equi_type[i] = 3

    equi_type = equi_type.astype(int)
    realized_size = len([i for i in equi_type if i != 3])
    
    # 全体の箱
    param1_data = np.empty((realized_size, 6))
    param1_data[:, 0] = pop[equi_type != 3]
    param1_data[:, 1:3] = dist[equi_type != 3, :]
    param1_data[:, 3:5] = probs[equi_type != 3, :]
    param1_data[:, 5] = equi_type[equi_type != 3]
    
    # 出力
    data = pd.DataFrame(param1_data)
    data.columns = ["pop", "dist1", "dist2", "entryprob1", "entryprob2", "equitype"]
    data["entry1"] = data["entryprob1"].apply(entry)
    data["entry2"] = data["entryprob2"].apply(entry)
    data["single"] = data["equitype"].apply(lambda x:1 if x == 0.0 else 0)
    data.to_csv(str(alpha)+"-"+str(beta)+"-"+str(delta)+".csv")
    