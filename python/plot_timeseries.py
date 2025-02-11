import plotly.io as pio
import pandas as pd
import json
import plotly.graph_objects as go
import sys
pio.renderers.default = "svg"  # or "png"


def plot_time_series(values, y_title, x_test_min, x_test_max, file):
    
    fig = go.Figure()
    
    y_test_min, y_test_max = sys.float_info.max, sys.float_info.min
    for label, vals in values.items():
        if label == 'Time':
            continue
        fig.add_trace(go.Scatter(x=values['Time'], y=vals, mode='lines', name=label,))
                                 
        temp_y_min = min(vals)
        y_test_min = temp_y_min if temp_y_min < y_test_min else y_test_min
        temp_y_max = max(vals)
        y_test_max = temp_y_max if temp_y_max > y_test_max else y_test_max
    
    fig.add_shape(type='rect', x0=x_test_min, x1=x_test_max, y0=y_test_min,
                  y1=y_test_max, fillcolor='rgba(240,0,0,0.15)', line=dict(width=0))
    
    fig.update_layout(xaxis_title='Time', yaxis_title=y_title,template='plotly_white')
    fig.show()
    fig.write_image(file)

def prepare_data(dataset, col, parameter):
    data_dir = '../data/clean/'
    log_dir = '../logs/context/'
    fig_dir = '../figures/'
    timestamp_cols = {'Gas': 'time', 'Electricity': 'date', 'Weather': 'Date Time'}
    
    timestamp_col = timestamp_cols[dataset]
    
    train = pd.read_csv(f'{data_dir}{dataset}_train.csv')
    test = pd.read_csv(f'{data_dir}{dataset}_test.csv')
    
    values = {}
    with open(f'{log_dir}{dataset}_logs.jsonl') as f:
        for line in f:
            j = json.loads(line)
            results = {x['qid']: {k:v for k,v in x['avg'].items() if k!='time'} for x in j['results']}
            pred_df = pd.DataFrame(results).T
            pred_df = pred_df.reset_index(drop=False)
            pred_df = pred_df.rename(columns={'index': timestamp_col})
            
            total_pred_df = pd.concat([train,pred_df])
            values[f"{parameter[0]}={j['settings'][parameter]}"] = total_pred_df[col]
    
    df = pd.concat([train,test])
    values['Original'] = df[col]
    values['Time'] = df[timestamp_col]
    
    x_test_min = test[timestamp_col].min()
    x_test_max = test[timestamp_col].max()
    
    file = '{}{}_{}_{}.pdf'.format(fig_dir, dataset, col, parameter)
    plot_time_series(values, y_title=col, x_test_min=x_test_min, x_test_max=x_test_max, file=file)
    
prepare_data(dataset='Gas', col='CO2%', parameter='context_window')
prepare_data(dataset='Weather', col='Tpot (K)', parameter='context_window')