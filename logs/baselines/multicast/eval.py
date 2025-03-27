import pickle
import json

def transform_col(col):
    names = {'GasRate(ft3/min)': 'GasRate',
             'CO2%': 'CO2',
             'HUFL': 'HUFL',
             'HULL': 'HULL',
             'OT': 'OT',
            'Tlog (degC)': 'Tlog',
            'H2OC (mmol/mol)': 'H2OC',
            'VPmax (mbar)': 'VPmax',
            'Tpot (K)': 'Tpot',
            '% WEIGHTED ILI': 'W. ILI', 
            '%UNWEIGHTED ILI': 'UNW. ILI', 
            'AGE 0-4': 'AGE 0-4',
            'AGE 5-24': 'AGE 5-24',
            'AGE 25-49': 'AGE 25-49',
            'AGE 50-64': 'AGE 50-64', 
            'AGE 65': 'AGE 65',
            'TOTAL PATIENTS': 'TOTAL'
    }
    return names.get(col, '')

datasets = ['Gas', 'Electricity', 'Weather', 'ILI']

for nod, dataset in enumerate(datasets):
    with open(f'results/data_MultiCast llama-7b_{nod}_10.pkl', 'rb') as f:
        data = pickle.load(f)
        
    out_model = data[0]
    train_init = data[1]
    test_init = data[2]    
    
    print(len(out_model['samples']))
    
    logs = {'settings': {"dataset": dataset, "no_predictions": 1, 
                         "num_repeats": int(out_model['samples'][0].shape[0])}}
    
    i_logs = {}
    for nov, pred in enumerate(out_model['samples']):
        col = test_init.columns[nov]
        col2 = transform_col(col)
    
        preds = pred.mean()
        
        for qid, avg in preds.items():
            if qid not in i_logs:
                i_logs[qid] = {'qid': str(qid), 'avg': {}, 'true': {}}
            i_logs[qid]['avg'][col2] = float(avg)
            i_logs[qid]['true'][col2] = float(test_init.loc[qid, col])
    logs['results'] = list(i_logs.values())
    
    with open(f'{dataset}_logs.jsonl', 'w') as f:
        f.write(json.dumps(logs)+'\n')
# for pred in 