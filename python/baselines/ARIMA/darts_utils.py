import pandas as pd
from darts import TimeSeries
import darts.models
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA as staARIMA


def get_arima_predictions_data(train, test, p=12, d=1, q=0, num_samples=100, **kwargs):
    num_samples = max(num_samples, 1)
    if not isinstance(train, list):
        # assume single train/test case
        train = [train]
        test = [test]
    for i in range(len(train)):    
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index = pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index = pd.RangeIndex(len(train[i]),len(test[i])+len(train[i])))

    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

    model = darts.models.ARIMA(p=p, d=d, q=q)

    scaled_train_ts_list = []
    scaled_test_ts_list = []
    scaled_combined_series_list = []
    scalers = []


    # Iterate over each series in the train list
    for train_series, test_series in zip(train,test):
        # for ARIMA we scale each series individually
        scaler = MinMaxScaler()
        combined_series = pd.concat([train_series,test_series])
        scaler.fit(combined_series.values.reshape(-1,1))
        scalers.append(scaler)
        scaled_train_series = scaler.transform(train_series.values.reshape(-1,1)).reshape(-1)
        scaled_train_series_ts = TimeSeries.from_times_and_values(train_series.index, scaled_train_series)
        scaled_train_ts_list.append(scaled_train_series_ts)

        scaled_test_series = scaler.transform(test_series.values.reshape(-1,1)).reshape(-1)
        scaled_test_series_ts = TimeSeries.from_times_and_values(test_series.index, scaled_test_series)
        scaled_test_ts_list.append(scaled_test_series_ts)
        
        scaled_combined_series = scaler.transform(pd.concat([train_series,test_series]).values.reshape(-1,1)).reshape(-1)
        scaled_combined_series_list.append(scaled_combined_series)
        

    rescaled_predictions_list = []
    nll_all_list = []
    samples_list = []

    for i in range(len(scaled_train_ts_list)):
        try:
            model.fit(scaled_train_ts_list[i])
            prediction = model.predict(len(test[i]), num_samples=num_samples).data_array()[:,0,:].T.values
            scaler = scalers[i]
            rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape(num_samples,-1)
            fit_model = model.model.model.fit()
            fit_params = fit_model.conf_int().mean(1)
            all_model = staARIMA(
                    scaled_combined_series_list[i],
                    exog=None,
                    order=model.order,
                    seasonal_order=model.seasonal_order,
                    trend=model.trend,
            )
            nll_all = -all_model.loglikeobs(fit_params)
            nll_all = nll_all[len(train[i]):].sum()/len(test[i])
            nll_all -= np.log(scaler.scale_)
            nll_all = nll_all.item()
        except np.linalg.LinAlgError:
            rescaled_prediction = np.zeros((num_samples,len(test[i])))
            # output nan
            nll_all = np.nan

        samples = pd.DataFrame(rescaled_prediction, columns=test[i].index)
        
        rescaled_predictions_list.append(rescaled_prediction)
        nll_all_list.append(nll_all)
        samples_list.append(samples)
        
    out_dict = {
        'NLL/D': np.mean(nll_all_list),
        'samples': samples_list if len(samples_list)>1 else samples_list[0],
        'median': [samples.median(axis=0) for samples in samples_list] if len(samples_list)>1 else samples_list[0].median(axis=0),
        'info': {'Method':'ARIMA', 'p':p, 'd':d}
    }

    return out_dict

