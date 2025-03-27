import pandas as pd
import argparse
import json
from time import time
import os

from utils import grid_iter
from darts_utils import get_arima_predictions_data
from validation_likelihood_tuning import get_autotuned_predictions_data

def run_predictions(df, timestamp_col, corr_cols, date_range, no_predictions, 
                    context_window, num_repeats):
    
    logs = []
    
    model_hypers = {'model': ['arima'], 'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}
    hypers = list(grid_iter(model_hypers))
    
    date_range_set = set(date_range)
    test_index = df[timestamp_col].apply(lambda x: x in date_range_set)
    train_df = df.loc[~test_index]
    test_df = df.loc[test_index]
    
    logs = {date: {'qid': str(date),
                   'all': {k: [] for k in df.columns},
                   'avg': {k: 0 for k in corr_cols+['times']},
                   'true': {}
                   } for date in date_range}
    
    for corr_col in corr_cols:
        t = time()
        results = get_autotuned_predictions_data(train_df[corr_col], 
                                                   test_df[corr_col], 
                                                   hypers, 
                                                   num_samples=num_repeats, 
                                                   get_predictions_fn=get_arima_predictions_data, 
                                                   verbose=False, 
                                                   parallel=False, 
                                                   device="cuda:1")
        t = time() - t 
        avg_t = t / len(date_range)
        
        for index, row in results['samples'].T.iterrows():
            qid = date_range[index]
            logs[qid]['avg']['times'] += avg_t
            logs[qid]['all'][corr_col] = list(row.values)
            logs[qid]['avg'][corr_col] = float(row.values.mean())
            logs[qid]['true'][corr_col] = float(test_df.loc[index][corr_col])
        
    logs = list(logs.values())
    return logs

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ivalue


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    parser.add_argument("--dataset", type=str, choices=['Gas', 'Electricity', 'Weather', 'ILI'], help="Dataset to use")
    parser.add_argument("--no_predictions", default=1, type=positive_int, help="Number of predictions (must be positive)")
    parser.add_argument("--context_window", default=10, type=positive_int, help="Size of the context window (must be positive)")
    parser.add_argument("--num_repeats", default=5, type=positive_int, help="Number of repeats per prompt(must be positive)")
    
    parser.add_argument("--data_dir", type=str, help="Path to the datasets")
    parser.add_argument("--out_file", type=str, help="Path to output file")
    
    args = parser.parse_args()
    print("\n", args)
    
    train = pd.read_csv(f'{args.data_dir}{args.dataset}_train.csv')
    test = pd.read_csv(f'{args.data_dir}{args.dataset}_test.csv')
    # test = test.head(2)
    df = pd.concat([train,test])
    
    timestamp_cols = {'Gas': 'time', 'Electricity': 'date', 'Weather': 'Date Time', 'ILI': 'time'}
    timestamp_col = timestamp_cols[args.dataset]
    
    columns = {'Gas': {'time': 'time',
                       'GasRate(ft3/min)': 'GasRate',
                       'CO2%': 'CO2'},
                'Electricity': {'date': 'date',
                                'HUFL': 'HUFL',
                                'HULL': 'HULL',
                                'OT': 'OT'
                                },
                'Weather': {'Date Time':'Date Time',
                            'Tlog (degC)': 'Tlog',
                            'H2OC (mmol/mol)': 'H2OC',
                            'VPmax (mbar)': 'VPmax',
                            'Tpot (K)': 'Tpot'
                            },
                'ILI': {'time': 'time',
                        '% WEIGHTED ILI': 'W. ILI', 
                        '%UNWEIGHTED ILI': 'UNW. ILI', 
                        'AGE 0-4': 'AGE 0-4',
                        'AGE 5-24': 'AGE 5-24',
                        'AGE 25-49': 'AGE 25-49',
                        'AGE 50-64': 'AGE 50-64', 
                        'AGE 65': 'AGE 65',
                        'TOTAL PATIENTS': 'TOTAL'
                    }
    }
    
    df = df.rename(columns=columns[args.dataset])
    
    logs = run_predictions(df, 
                        timestamp_col = timestamp_col, 
                        corr_cols = [c for c in df.columns if c != timestamp_col],
                        date_range = test[timestamp_col].values.tolist(),
                        no_predictions = args.no_predictions,
                        context_window = args.context_window,
                        num_repeats = args.num_repeats,
                        )
    
    settings = {k:v for k,v in vars(args).items() if k not in ['data_dir', 'out_file']}
    logs = {'settings': settings,
            'results': logs}
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'a') as f:
        f.write(json.dumps(logs)+"\n")