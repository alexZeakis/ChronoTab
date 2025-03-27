import json
import pandas as pd
import os
from math import sqrt


def calc_absolute_percentage_error(true, predicted):
    if true == 0:
        true = 0.000000001
    return abs((true-predicted)/true) * 100

def calc_squared_error(true, predicted):
    return (true-predicted)**2

def simple_predict(j, metric=calc_squared_error, normalize=False):
    scores = j['settings']
    cols = j['results'][0]['true']
    
    for col in cols:
        scores[col] = 0
    
    if normalize:
        min_vals, max_vals = {}, {}
        for col in cols:
            min_vals[col] = 100000000000
            max_vals[col] = -100000000000
        for entry in j['results']:
            for col in entry['true']:
                if entry['true'][col] > max_vals[col]:
                    max_vals[col] = entry['true'][col]
                if entry['true'][col] < min_vals[col]:
                    min_vals[col] = entry['true'][col]
        denom = {col: max_vals[col]-min_vals[col] for col in cols}
            
    for entry in j['results']:
        for col in entry['true']:
            # scores[col] += (entry['true'][col] - entry['avg'][col])**2
            # scores[col] += metric(entry['true'][col], entry['avg'][col])
            x, y = entry['true'][col], entry['avg'][col]
            if normalize:
                x = (x-min_vals[col]) / denom[col]
                y = (y-min_vals[col]) / denom[col]
            scores[col] += metric(x, y)
            
    for col in cols:
        scores[col] = scores[col]/len(j['results'])
        if metric==calc_squared_error:
            scores[col] = sqrt(scores[col])
        
    return scores

dataset_cols = {'Gas': 2, 'Electricity': 3, 'Weather': 4, 'ILI': 8}

log_dir = '../logs/baselines/'
metrics = {'RMSE': calc_squared_error, 'MAPE': calc_absolute_percentage_error}
for method_dir in os.listdir(log_dir):
    print(method_dir)
    if method_dir != 'multicast':
        continue
    total_df = pd.DataFrame()
    for file in os.listdir(log_dir+method_dir):
        path = log_dir + method_dir + "/" + file
        if not path.endswith('jsonl'):
            continue
        
        local_df = pd.DataFrame()
        for metric, metric_func in metrics.items():
            total_scores = []
            with open(path) as f:
                for line in f:
                    j = json.loads(line)
                    
                    scores = simple_predict(j, metric_func, False)
                    total_scores.append(scores)
                
    
            df = pd.DataFrame(total_scores)
            # df = df.loc[:, df.nunique() > 1]
            df = df.iloc[:, -dataset_cols[j['settings']['dataset']]:]
            df['Mean'] = df.mean(axis=1)
            cols = df.columns
            
            if metric == 'RMSE':
                df = df.round(2)
                df = df.applymap(lambda x: x if isinstance(x, str) else (">100" if x > 100 else str(float(x))))
            elif metric == 'MAPE':
                df = df.applymap(lambda x: x if isinstance(x, str) else (">100" if x > 100 else str(int(x))))
            df[0] = [f'{method_dir}-{metric}']
            df = df[[0]+list(cols)]
            
            local_df = pd.concat([local_df, df], axis=0)
        total_df = pd.concat([total_df, local_df.T])
        # print(total_df.to_latex(index=False))
    
    total_df = total_df[list(set(total_df.columns))]
    print(total_df.to_latex())
    