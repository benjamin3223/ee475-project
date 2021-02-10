#!/usr/bin/env python3
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
from sklearn.metrics import mean_squared_error


def accuracy(y_true, y_pred):
    try:
        distance = y_pred - y_true
        print(distance)
        numerator = np.sum(np.absolute(distance), axis=0)
        print(numerator)
        denumerator = 2 * np.sum(y_true, axis=0)
        print(denumerator)
        return float(1 - (numerator / denumerator))
    except:
        print('it is fucking up')
        return np.nan


def the_other_accruacy(y_true, y_pred):
    try:
        distance = y_pred - y_true
        numerator = np.absolute(np.sum(distance, axis=0))
        denumerator = 2 * np.sum(y_true, axis=0)
        return float(1 - (numerator / denumerator))
    except:
        return np.nan


def root_mean_squared_error(y_true, y_pred):
    try:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    except:
        return np.nan


def normalised_signal_aggregate_error(y_true, y_pred):
    try:
        distance = y_true - y_pred
        numerator = np.sum(np.absolute(distance))
        denumerator = np.sum(y_true)
        return numerator / denumerator
    except:
        return np.nan


def match_rate(y_true, y_pred):  # EE8 Metric Score: 35
    try:
        numerator = np.sum(np.minimum(y_pred, y_true))
        denumerator = np.sum(np.maximum(y_pred, y_true))
        return numerator / denumerator
    except:
        return np.nan


def energy_error(y_true, y_pred):  # EE6 Metric Score: 32
    try:
        distance = y_pred - y_true
        absolute_distance = np.absolute(distance)
        numerator = np.sum(absolute_distance)
        denumerator = np.sum(y_true)
        return numerator / denumerator
    except:
        return np.nan


def energy_accuracy(y_true, y_pred):  # EE7 Metric Score: 30
    try:
        alpha = 1.4
        energyerror = energy_error(y_true, y_pred)
        return np.exp(-alpha * energyerror)
    except:
        return np.nan


def relative_error(y_true, y_pred):  # EE1 Metric Score: 30
    try:
        numerator = np.sum(y_true) - np.sum(y_pred)
        denumerator = np.sum(y_true)
        return numerator / denumerator
    except:
        return np.nan


def root_mean_square_deviation(y_true, y_pred):  # EE2 Metric Score: 16
    try:
        N = y_true.shape[0]
        distance = y_true - y_pred
        distance_squared = np.square(distance)
        sum_of_distance_squared = np.sum(distance_squared)
        numerator = np.sqrt(1 / N * sum_of_distance_squared)
        denumerator = np.sum(np.mean(y_true))
        return numerator / denumerator
    except:
        return np.nan


def percent_standard_deviation_explained(y_true, y_pred):  # EE4 Metric Score: 16
    try:
        r_sq = r_squared(y_true, y_pred)
        return float(1 - np.sqrt(1 - r_sq))
    except:
        return np.nan


def average_error(y_true, y_pred):  # EE3.1 Metric Score: 12
    try:
        N = y_true.shape[0]
        summed_error = np.sum(y_pred - y_true)
        return (1 / N) * summed_error
    except:
        return np.nan


def standard_deviation_error(y_true, y_pred):  # EE3.2 Metric Score: 12
    try:
        N = y_true.shape[0]
        error_between = y_pred - y_true
        summed_average_error_between = np.mean(np.sum(error_between))
        squared = np.square(error_between - summed_average_error_between)
        return np.sqrt((1 / N) * np.sum(squared))
    except:
        return np.nan


def r_squared(y_true, y_pred):  # EE5 Metric Score: 12
    try:
        y_true_mean = np.mean(y_true, axis=0)
        numerator = np.sum(np.square(y_true - y_pred), axis=0)
        denumerator = np.sum(np.square(y_true - y_true_mean), axis=0)
        return float(1 - (numerator / denumerator))
    except:
        return np.nan


def standard_error_mean(y_true, y_pred):  # EE9 Metric Score: 12
    try:
        N = y_true.shape[0]
        std_dev_error = standard_deviation_error(y_true, y_pred)
        return std_dev_error / np.sqrt(N)
    except:
        return np.nan


def fraction_energy_explained(y_true, y_pred):
    try:
        numerator = np.sum(y_pred, axis=0)
        denumerator = np.sum(y_true, axis=0)
        return float(numerator / denumerator)
    except:
        return np.nan


def total_energy_correctly_assigned(y_true_agg, y_true_app, y_pred_app):
    try:
        '''Appliance Sub-metering Required'''
        numerator = np.sum(np.absolute(y_pred_app - y_true_app))
        denumerator = 2 * np.sum(y_true_agg)
        return numerator / denumerator
    except:
        return np.nan


def error_total_energy_assigned(y_true, y_pred):
    try:
        '''Appliance Only'''
        return np.absolute(np.sum(y_pred, axis=0) - np.sum(y_true, axis=0))
    except:
        return np.nan


def dev(y_true, y_pred):
    try:
        '''Appliance Only'''
        numerator = error_total_energy_assigned(y_true, y_pred)
        denumerator = np.sum(y_true)
        return numerator / denumerator
    except:
        return np.nan


def fraction_total_energy_assigned_correctly(y_true_agg, y_true_app, y_pred_app):
    try:
        '''Appliance Sub-metering Required'''
        numerator_true = np.sum(y_true_app, axis=0)
        denumerator = np.sum(y_true_agg)
        result_true = numerator_true / denumerator
        numerator_pred = np.sum(y_pred_app, axis=0)
        result_pred = numerator_pred / denumerator
        return float(np.minimum(result_true, result_pred))
    except:
        return np.nan


def error_percentage(y_true, y_pred):
    try:
        distance = y_pred - y_true
        numerator = np.sum(np.absolute(distance), axis=0)
        denumerator = np.sum(y_true, axis=0)
        return float(numerator / denumerator)
    except:
        return np.nan


def error_percentage_nonabs(y_true, y_pred):
    try:
        distance = y_pred - y_true
        numerator = np.sum(distance, axis=0)
        denumerator = np.sum(y_true, axis=0)
        return float(numerator / denumerator)
    except:
        return np.nan


def get_score_names():
    names = ['Accuracy', 'TheOtherAccruacy', 'RootMeanSquareError', 'NormalisedSignalAggregateError', 'MatchRate',
             'EnergyError', 'EnergyAccuracy', 'RelativeError', 'RootMeanSquareDeviation',
             'PrecentStandardDeviationExplained', 'AverageError', 'StandardDeviationError',
             'RSquared', 'MeanStandardError', 'FactionEnergyExplained', 'ErrorPercentage',
             'error_percentage_nonabs']
    return names


def get_scores(y_true, y_pred):
    scores = [accuracy(y_true, y_pred),
              the_other_accruacy(y_true, y_pred),
              root_mean_squared_error(y_true, y_pred),
              normalised_signal_aggregate_error(y_true, y_pred),
              match_rate(y_true, y_pred),
              energy_error(y_true, y_pred),
              energy_accuracy(y_true, y_pred),
              relative_error(y_true, y_pred),
              root_mean_square_deviation(y_true, y_pred),
              percent_standard_deviation_explained(y_true, y_pred),
              average_error(y_true, y_pred),
              standard_deviation_error(y_true, y_pred),
              r_squared(y_true, y_pred),
              standard_error_mean(y_true, y_pred),
              fraction_energy_explained(y_true, y_pred),
              error_percentage(y_true, y_pred),
              error_percentage_nonabs(y_true, y_pred)]
    return scores


if __name__ == "__main__":
    in_array = np.linspace(-np.pi, np.pi, 100)
    y_true = np.absolute(np.sin(in_array))
    y_pred = np.absolute(np.sin(in_array))
    y_pred[55] = 10
    print(len(get_scores(y_true, y_pred)))
    print(len(get_score_names()))
    data = {'Name': get_score_names(),
            'Score': get_scores(y_true, y_pred)}
    print(data)
    df = pd.DataFrame(data)
    print(df)
