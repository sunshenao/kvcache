import json
import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    data = []
    t = "final_async"
    for name in ['1:1disagg_prefill', '1:2disagg_prefill','2:1disagg_prefill','default_prefill','chunked_prefill']:
        for qps in range(0,100): # 2,4,
            if not os.path.exists(f"{t}/{name}-qps-{qps}.json"):
                continue
            with open(f"{t}/{name}-qps-{qps}.json") as f:
                x = json.load(f)
                x['name'] = name
                x['qps'] = qps
                data.append(x)

    df = pd.DataFrame.from_dict(data)
    # print(df)
    dis_df1_1 = df[df['name'] == '1:1disagg_prefill']
    dis_df1_2 = df[df['name'] == '1:2disagg_prefill']
    dis_df2_1 = df[df['name'] == '2:1disagg_prefill']

    def_df = df[df['name'] == 'default_prefill']
    chu_df = df[df['name'] == 'chunked_prefill']

    plt.style.use('bmh')
    plt.rcParams['font.size'] = 20

    for key in [
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_itl_ms',
            'median_itl_ms', 'p99_itl_ms','mean_tpot_ms','median_tpot_ms','p99_tpot_ms'
    ]:

        fig, ax = plt.subplots(figsize=(11, 7))
        plt.plot(dis_df1_1['qps'],
                 dis_df1_1[key],
                 label='1:1disagg_prefill',
                 marker='o',
                 linewidth=4)
        
        plt.plot(dis_df1_2['qps'],
                 dis_df1_2[key],
                 label='1:2disagg_prefill',
                 marker='o',
                 linewidth=4)
        
        plt.plot(dis_df2_1['qps'],
                 dis_df2_1[key],
                 label='2:1disagg_prefill',
                 marker='o',
                 linewidth=4)
        
        plt.plot(def_df['qps'],
                 def_df[key],
                 label='default_prefill',
                 marker='o',
                 linewidth=4)
        
        plt.plot(chu_df['qps'],
                 chu_df[key],
                 label='chunked_prefill',
                 marker='o',
                 linewidth=4)
        
        ax.legend(fontsize=16)  # 设置为8号字体


        ax.set_xlabel('QPS')
        ax.set_ylabel(key)
        ax.set_ylim(bottom=0)
        fig.savefig(f'{t}/{key}.png')
        plt.close(fig)
