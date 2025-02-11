import json
import pandas as pd
import os

def simple_predict(j):
    scores = j['settings']
    cols = j['results'][0]['true']
    for col in cols:
        scores[col] = 0
    for entry in j['results']:
        for col in entry['true']:
            scores[col] += (entry['true'][col] - entry['avg'][col])**2
    for col in cols:
        scores[col] = scores[col]/len(j['results'])
    return scores

def multiple_predict(j):
    scores = j['settings']
    cols = j['results'][0]['true']
    counter = {}
    for col in cols:
        scores[col] = 0
        counter[col] = 0
    for entry in j['results']:
        for col in entry['true']:
            for yi, y2i in zip(entry['true'][col], entry['avg'][col]):
                scores[col] += (yi - y2i)**2
                counter[col] += 1
    for col in cols:
        scores[col] = scores[col]/counter[col]
    return scores
    

log_dir = '../logs/'
for exp_dir in os.listdir(log_dir):
    total_df = pd.DataFrame()
    print(exp_dir)
    for file in os.listdir(log_dir+exp_dir):
        path = log_dir + exp_dir + "/" + file
        
        total_scores = []
        with open(path) as f:
            for line in f:
                j = json.loads(line)
                
                if j['settings']['no_predictions'] == 1:
                    scores = simple_predict(j)
                else:
                    scores = multiple_predict(j)
                total_scores.append(scores)
                

        df = pd.DataFrame(total_scores)
        df = df.loc[:, df.nunique() > 1]
        # df2['dataset'] = df.dataset.values
        df = df.round(2)
        total_df = pd.concat([total_df, df.T])
        # print(total_df.to_latex(index=False))
        
    # print(total_df.to_latex())
    # if exp_dir == 'horizon':
    #     df2 = total_df.copy()
    #     index = [x for x in df2.index if x !='no_predictions']
    #     df2 = df2.loc[index]
    #     print(df2.apply(lambda x: (x[0]-x[1])/x[0], axis=1).mean())
    #     print(df2.apply(lambda x: (x[0]-x[2])/x[0], axis=1).mean())