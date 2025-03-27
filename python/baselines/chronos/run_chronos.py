import pandas as pd
import torch
from chronos import ChronosPipeline
import argparse
import json
from time import time
import os

def run_predictions(df, timestamp_col, corr_cols, date_range, no_predictions, 
                    context_window, model, num_repeats):
    
    logs = []
    
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        device_map="cuda:1",  # use "cpu" for CPU inference
        torch_dtype=torch.bfloat16,
    )
    
    for nod, date in enumerate(date_range):
        if nod % 5 == 0:
            print('Query {}/{}\r'.format(nod, len(date_range)), end='')
        index = df[timestamp_col].eq(date).to_numpy().argmax()
        sampled_df = df.iloc[index-context_window:index]
        
        log = {'qid': date, 'all': {k: [] for k in corr_cols+['times']},
               'avg': {}, 'true': {}}
        
        total_time = 0
        for corr_col in corr_cols:
            # corr_df = sampled_df[[timestamp_col, corr_col]]
            
            t = time()
            predictions = pipeline.predict(num_samples=num_repeats, 
                                           context=torch.tensor(sampled_df[corr_col].values),
                                           prediction_length=no_predictions)
            total_time += time() - t 
            
            log['all'][corr_col] = predictions.squeeze().tolist()
            log['avg'][corr_col] = predictions.mean().item()
            log['true'][corr_col] = float(df.iloc[index][corr_col])

        log['all']['times'] = total_time
        log['avg']['times'] = total_time/num_repeats
        
        logs.append(log)
        # if nod > 2:
        #     break
        
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
    parser.add_argument("--model", default="amazon/chronos-t5-small", type=str, help="Chronos model to use")
    
    parser.add_argument("--data_dir", type=str, help="Path to the datasets")
    parser.add_argument("--out_file", type=str, help="Path to output file")
    
    args = parser.parse_args()
    print("\n", args)
    
    train = pd.read_csv(f'{args.data_dir}{args.dataset}_train.csv')
    test = pd.read_csv(f'{args.data_dir}{args.dataset}_test.csv')
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
                        model = args.model,
                        num_repeats = args.num_repeats,
                        )
    
    settings = {k:v for k,v in vars(args).items() if k not in ['data_dir', 'out_file']}
    logs = {'settings': settings,
            'results': logs}
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'a') as f:
        f.write(json.dumps(logs)+"\n")