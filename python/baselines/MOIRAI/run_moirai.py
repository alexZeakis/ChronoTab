import pandas as pd
import argparse
import json
from time import time
import os
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


def run_predictions(df, date_range, no_predictions, context_window, 
                    model, model_size, num_repeats):
    
    MODEL = model  # model name: choose from {'moirai', 'moirai-moe'}
    SIZE = model_size  # model size: choose from {'small', 'base', 'large'}
    PDT = no_predictions  # prediction length: any positive integer
    CTX = context_window  # context length: any positive integer
    PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 1  # batch size: any positive integer
    TEST = len(date_range)  # test set length: any positive integer
    num_samples = num_repeats
    
    ds = PandasDataset(dict(df))
    
    # Split into train/test set
    train, test_template = split(
        ds, offset=-TEST
    )  # assign last TEST time steps as test set
    
    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=TEST // PDT,  # number of windows in rolling window evaluation
        distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
    )
    
    # Prepare pre-trained model by downloading model weights from huggingface hub
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=16,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    
    total_time = time()
    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(test_data.input)
    total_time = time() - total_time
    # avg_time = total_time / (len(date_range) * df.shape[1]) # time per column
    avg_time = total_time / len(date_range)  # time per timeseries
    
    logs = {date: {'qid': str(date),
                   'all': {k: [] for k in df.columns},
                   'avg': {},
                   'true': {}
                   } for date in date_range}
    
    
    for forecast in forecasts:
        corr_col = forecast.item_id
        date = forecast.start_date.to_timestamp()
        
        logs[date]['all'][corr_col] = [float(x) for x in forecast.samples.flatten().tolist()]
        logs[date]['avg'][corr_col] = float(forecast.samples.mean())
        logs[date]['true'][corr_col] = float(df.loc[date][corr_col])
        logs[date]['avg']['times'] = avg_time

        # if nod > 2:
        #     break
    
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
    parser.add_argument("--model", default="moirai-moe", choices=['moirai', 'moirai-moe'], type=str, help="MOIRAI model to use")
    parser.add_argument("--model_size", default="small", choices=['small', 'base', 'large'], type=str, help="Size of MOIRAI model")
    
    parser.add_argument("--data_dir", type=str, help="Path to the datasets")
    parser.add_argument("--out_file", type=str, help="Path to output file")
    
    args = parser.parse_args()
    print("\n", args)
    
    train = pd.read_csv(f'{args.data_dir}{args.dataset}_train.csv', parse_dates=True)
    test = pd.read_csv(f'{args.data_dir}{args.dataset}_test.csv', parse_dates=True)
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
    df = df.set_index(timestamp_col)
    
    start_date = "2024-01-01"  # Set a start date
    new_index = pd.date_range(start=start_date, periods=len(df), freq="D")
    df.index = new_index
    date_range = df.tail(test.shape[0]).index.tolist()
    
    logs = run_predictions(df, 
                        # date_range = test[timestamp_col].values.tolist(),
                        date_range = date_range,
                        no_predictions = args.no_predictions,
                        context_window = args.context_window,
                        model = args.model,
                        model_size = args.model_size,
                        num_repeats = args.num_repeats,
                        )
    
    settings = {k:v for k,v in vars(args).items() if k not in ['data_dir', 'out_file']}
    logs = {'settings': settings,
            'results': logs}
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'a') as f:
        f.write(json.dumps(logs)+"\n")