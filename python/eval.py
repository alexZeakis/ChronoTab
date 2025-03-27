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

def simple_predict(j, metric=calc_squared_error):
    scores = j['settings']
    cols = j['results'][0]['true']
    for col in cols:
        scores[col] = 0
    for entry in j['results']:
        for col in entry['true']:
            # scores[col] += (entry['true'][col] - entry['avg'][col])**2
            scores[col] += metric(entry['true'][col], entry['avg'][col])
    for col in cols:
        scores[col] = scores[col]/len(j['results'])
        if metric==calc_squared_error:
            scores[col] = sqrt(scores[col])
        
    return scores

def multiple_predict(j, metric=calc_squared_error):
    scores = j['settings']
    cols = j['results'][0]['true']
    counter = {}
    for col in cols:
        scores[col] = 0
        counter[col] = 0
    for entry in j['results']:
        for col in entry['true']:
            for yi, y2i in zip(entry['true'][col], entry['avg'][col]):
                # scores[col] += (yi - y2i)**2
                scores[col] += metric(yi, y2i)
                counter[col] += 1
    for col in cols:
        scores[col] = scores[col]/counter[col]
        if metric==calc_squared_error:
            scores[col] = sqrt(scores[col])
    return scores
    

log_dir = '../logs/'
metrics = {'RMSE': calc_squared_error, 'MAPE': calc_absolute_percentage_error}
for exp_dir in os.listdir(log_dir):
    total_df = pd.DataFrame()
    
    if exp_dir == 'baselines':
        continue
    
    print(exp_dir)
    for file in os.listdir(log_dir+exp_dir):
        path = log_dir + exp_dir + "/" + file
        
        local_df = pd.DataFrame()
        for metric, metric_func in metrics.items():
            total_scores = []
            with open(path) as f:
                for line in f:
                    j = json.loads(line)
                    
                    if j['settings']['no_predictions'] == 1:
                        scores = simple_predict(j, metric_func)
                    else:
                        scores = multiple_predict(j, metric_func)
                    total_scores.append(scores)
                

            df = pd.DataFrame(total_scores)
            df = df.loc[:, df.nunique() > 1]
            df['Mean'] = df.iloc[:, 1:].mean(axis=1)

            if metric == 'RMSE':
                df = df.round(2)
                df = df.applymap(lambda x: x if isinstance(x, str) else (">100" if x > 100 else str(float(x))))
            elif metric == 'MAPE':
                df = df.applymap(lambda x: x if isinstance(x, str) else (">100" if x > 100 else str(int(x))))
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: f'{x}-{metric}')
            local_df = pd.concat([local_df, df], axis=0)
        total_df = pd.concat([total_df, local_df.T])
        # print(total_df.to_latex(index=False))

    total_df = total_df[list(set(total_df.columns))]
    # print(total_df)
    # if exp_dir == 'baselines':
    #     break
    # df2 = total_df.copy()
    # index = [x for x in df2.index if x !='no_predictions']
    # df2 = df2.loc[index]
    # print(df2.apply(lambda x: (x[0]-x[1])/x[0], axis=1).mean())
    # # print(df2.apply(lambda x: (x[0]-x[2])/x[0], axis=1).mean())
    # # total_df.iloc[:,[0,2]].apply(lambda x: (x[1]-x[0])/x[1], axis=1).mean()
    
    
    # df2 = total_df.iloc[:,[0,2,4]]
    # df2 = df2[df2.applymap(lambda x: isinstance(x, float)).all(axis=1)]
    # df2.apply(lambda x: (x[0]-x[1])/x[1], axis=1).mean()
        
    print(total_df.to_latex())