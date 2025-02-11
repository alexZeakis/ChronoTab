import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import traceback
import json
from time import time
import editdistance
import argparse

torch.cuda.empty_cache()
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
loaded_models = {'tablegpt/TableGPT2-7B': {
                'model': AutoModelForCausalLM.from_pretrained(
                    'tablegpt/TableGPT2-7B', torch_dtype=torch.float16, device_map={"": 0}
                    ).to(device),
                'tokenizer': AutoTokenizer.from_pretrained('tablegpt/TableGPT2-7B')
                }
    }

def mean(x):
    return sum(x) / len(x)

def serialize(df, method):
    if method == 'table':
        header = '|' + '|'.join(df.columns) + '|'
        separator = '|' + '|'.join('---' for _ in df.columns) + '|'
        content = '\n'.join('|' + '|'.join(f"{val:.10g}" if isinstance(val, float) else str(val) for val in row) + '|'for row in df.values)
        return f"{header}\n{separator}\n{content}"
    

def extract_json_fields(json_str, template):
    parsed_json = {}
    
    for key, val in template.items():
        multiple = isinstance(val, list) # Multiple predictions (array)
        
        if multiple:
            pattern = r'{}.*?\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]'
        else:
            pattern = r'{}.*?(-?\d*\.?\d+)'
            
        pat = pattern.format(re.escape(key))
        match = re.search(pat, json_str)
        
        if match:
            if multiple:  # Convert to list of floats
                parsed_json[key] = list(map(float, match.group(1).split(',')))
            else:  # Single float
                parsed_json[key] = float(match.group(1))
        else:
            parsed_json[key] = [] if multiple else None  # Handle missing fields
    
    return parsed_json

def run_prompt(prompt, model_name, endpoint=None, token=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        if endpoint is None: # HuggingFace alternative:
            models = loaded_models[model_name]
            model = models['model']
            tokenizer = models['tokenizer']
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else: # OpenAI protocol
            client = OpenAI(base_url = endpoint, api_key=token)
            
            response = client.chat.completions.create(
              model=model_name, messages=messages
            )
            response = response.choices[0].message.content
        return response
    except Exception as e:
        traceback.print_exc()
        return None 
    

def run_predictions(df, timestamp_col, corr_cols, date_range, no_predictions, 
                    context_window, serialization, model, num_repeats, 
                    timestamp, domain):
    
    logs = []
    
    models = {'TableGPT2': ('tablegpt/TableGPT2-7B', None, None),
              'Llama3.1': ('llama3.1:latest', "http://localhost:11434/v1", "ollama"),
              #'Llama3.1': ('llama-3.1-8b-instant', "https://api.groq.com/openai/v1", "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"),
              'Qwen2.5': ('qwen2.5:32b', "http://localhost:11434/v1", "ollama")
            }
    
    model_info  = models[model]
    if model_info[1] is not None:
        # del loaded_models
        torch.cuda.empty_cache()
    
    
    for nod, date in enumerate(date_range):
        if nod % 5 == 0:
            print('Query {}/{}\r'.format(nod, len(date_range)), end='')
        # print(date)
        # index = df[df[timestamp_col].astype(str).str.startswith(date)].index[0]
        index = df[timestamp_col].eq(date).to_numpy().argmax()
        sampled_df = df.iloc[index-context_window:index]
        # print(sampled_df)
        if timestamp == 'aware':
            corr_df = sampled_df[[timestamp_col]+corr_cols]
        else:
            corr_df = sampled_df[corr_cols]
        serialized_df = serialize(corr_df, serialization)
        
        if no_predictions == 1:
            ret_str = ""
            for corr_col in corr_cols:
                ret_str += "\"" + corr_col + "\": <predicted_value>,"
            template = {corr_col: None for corr_col in corr_cols}
            ret_str = ret_str[:-1]
            
            if domain is None:
                structure = "#Task Description: {}\n\n#Input:\n**Table:**\n{}\n\nReturn the final result as JSON in the format {{"+ ret_str+ "}}.\n\n# Output:"
                task_description = 'Based on the following table that contains correlated time-series, predict the values only for the next row for each column. Do not explain your answer and do not provide any code, return only the result.'
                prompt = structure.format(task_description, serialized_df)
            else:
                structure = "#Task Description: {}\n\n#Input:\n**Table:**\n{}\n**Table Information:**\n{}\n\nReturn the final result as JSON in the format {{"+ ret_str+ "}}.\n\n# Output:"
                task_description = 'Based on the following table that contains correlated time-series, predict the values only for the next row for each column. You can use the Table Information as you see fit. Do not explain your answer and do not provide any code, return only the result.'
                prompt = structure.format(task_description, serialized_df, domain)
            log = {'qid': date, 'all': {k: [] for k in corr_cols+['times']},
                   'avg': {}, 'true': {}}
        else: 

            pred_str = ""
            for nov in range(1, no_predictions+1):
                pred_str += f"<predicted_value_{nov}>,"
            pred_str = pred_str[:-1]
            
            ret_str = ""
            for corr_col in corr_cols:
                ret_str += "\"" + corr_col + "\": [" + pred_str + "],"
            template = {corr_col: [] for corr_col in corr_cols}
            ret_str = ret_str[:-1]
            
            if domain is None:
                structure = "#Task Description: {}\n\n#Input:\n**Table:**\n{}\n\nReturn the final result as JSON in the format {{"+ ret_str+ "}}.\n\n# Output:"
                # structure = "#Task Description: {}\n\n#Input:\n**Table:**\n{}\n\n.\n\n# Output:"
                task_description = 'Based on the following table that contains correlated time-series, predict the next {} values as an array for each column. Do not explain your answer and do not provide any code, return only the result.'.format(no_predictions)
                prompt = structure.format(task_description, serialized_df)
            else:
                structure = "#Task Description: {}\n\n#Input:\n**Table:**\n{}\n**Table Information:**\n{}\n\nReturn the final result as JSON in the format {{"+ ret_str+ "}}.\n\n# Output:"
                task_description = 'Based on the following table that contains correlated time-series, predict the next {} values as an array for each column. You can use the Table Information as you see fit. Do not explain your answer and do not provide any code, return only the result.'.format(no_predictions)
                prompt = structure.format(task_description, serialized_df, domain)
        
            log = {'qid': date, 'all': {k: [[] for _ in range(no_predictions)] for k in corr_cols+['times']},
                   'avg': {}, 'true': {}}
        for i in range(num_repeats):
            try:
                t = time()
                # print(prompt)                
                response = run_prompt(prompt, model_info[0], model_info[1], model_info[2])
                # print(response)
                t = time() - t
                response = extract_json_fields(response, template)
                # print(response)
            except:
                traceback.print_exc()
                continue
            
            if no_predictions == 1:    
                log['all']['times'].append(t)
                for k,v in response.items():
                    if v is None:
                        continue
                    log['all'][k].append(float(v)) #sometimes it gets back as string
            else:
                log['all']['times'][0].append(t)
                for k, vs in response.items():
                    for nov, v in enumerate(vs):
                        log['all'][k][nov].append(float(v)) #sometimes it gets back as string
        
        if no_predictions == 1:                            
            if len(log['all']['times']) == 0: # empty
                continue
            log['avg']['time'] = mean(log['all']['times'])
            for corr_col in corr_cols:
                log['avg'][corr_col] = mean(log['all'][corr_col])
                log['true'][corr_col] = df.iloc[index][corr_col]
        else:
            if len(log['all']['times']) == 0: # empty
                continue
            log['avg']['time'] = mean(log['all']['times'][0])
            for corr_col in corr_cols:
                log['avg'][corr_col] = []
                for vals in log['all'][corr_col]:
                    if len(vals) == 0:
                        continue
                    log['avg'][corr_col].append(mean(vals))
                log['true'][corr_col] = []
                for offs_index in range(index, index+no_predictions):
                    if offs_index >= df.shape[0]: #near the end of the dataframe
                        continue
                    log['true'][corr_col].append(df.iloc[offs_index][corr_col])
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

    parser.add_argument("--dataset", type=str, choices=['Gas', 'Electricity', 'Weather'], help="Dataset to use")
    parser.add_argument("--no_predictions", default=1, type=positive_int, help="Number of predictions (must be positive)")
    parser.add_argument("--context_window", default=10, type=positive_int, help="Size of the context window (must be positive)")
    parser.add_argument("--serialization", default="table", type=str, choices=["table"], help="Serialization method")
    parser.add_argument("--model", type=str, choices=["TableGPT2", "Llama3.1", "Qwen2.5"], help="Model to use")
    parser.add_argument("--num_repeats", default=5, type=positive_int, help="Number of repeats per prompt(must be positive)")
    parser.add_argument("--timestamp", default="aware", type=str, choices=['aware','agnostic'], help="Specify if the prompt should be aware of timestamps or treat them as irrelevant")
    parser.add_argument("--domain", default="agnostic", type=str, choices=['aware','agnostic'], help="Specify if the prompt should consider domain-specific context or remain neutral")

    parser.add_argument("--data_dir", type=str, help="Path to the datasets")
    parser.add_argument("--out_file", type=str, help="Path to output file")
    
    args = parser.parse_args()
    print("\n", args)
    
    train = pd.read_csv(f'{args.data_dir}{args.dataset}_train.csv')
    test = pd.read_csv(f'{args.data_dir}{args.dataset}_test.csv')
    df = pd.concat([train,test])
    
    timestamp_cols = {'Gas': 'time', 'Electricity': 'date', 'Weather': 'Date Time'}
    timestamp_col = timestamp_cols[args.dataset]
    
    domain_info = {'Gas': 'This is a 2-dimensional dataset containing carbon dioxide (CO2) emissions. The first dimension contains the input CO2 measurements (ft3/min) in a gas furnace. The second dimension contains the output CO2 percentage. The dataset is obtained from the darts library. Of course, the two dimensions are correlated, which makes this dataset ideal for multivariate forecasting.',
                   'Electricity': 'This multivariate time series is part of the Elec- tricity Transformer Dataset (ETDataset). It contains hourly measurements of various metrics, which were resampled on a 3-day basis, for a total of 242 timestamps. From this dataset, we extracted 3 dimensions of electricity measurements, specif- ically the High UseFul Load (HUFL), High UseLess Load (HULL), and Oil Temperature (OT). Again, the dimensions are correlated; specifically, OT is used as a target variable in regression problems.',
                   'Weather': 'The weather dataset was generated by the Max Planck Institute and contains 21 weather-related metrics obtained from a weather station located in Germany. From the 21 variables, we extracted the air temperatures (Tlog) measured in Celsius degrees, the water vapor concentration (H2OC) measured in mmol/mol, the saturation water vapor pressure (VPmax), measured in mbar, and the potential temperature (Tpot) measured in Kelvin degrees. Again, being weather-related, all dimensions are correlated.'
        }
    
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
                            }
    }
    
    df = df.rename(columns=columns[args.dataset])
    
    domain = None
    if args.domain == 'aware':
        domain = domain_info[args.dataset]
    
    logs = run_predictions(df, 
                        timestamp_col = timestamp_col, 
                        corr_cols = [c for c in df.columns if c != timestamp_col],
                        date_range = test[timestamp_col].values.tolist(),
                        no_predictions = args.no_predictions,
                        context_window = args.context_window,
                        serialization = args.serialization,
                        # corr_models  = ['Llama3.1'],
                        model = args.model,
                        num_repeats = args.num_repeats,
                        timestamp = args.timestamp,
                        domain = domain,
                        )
    
    settings = {k:v for k,v in vars(args).items() if k not in ['data_dir', 'out_file']}
    logs = {'settings': settings,
            'results': logs}
            
    with open(args.out_file, 'a') as f:
        f.write(json.dumps(logs)+"\n")