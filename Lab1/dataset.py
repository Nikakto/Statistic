import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import collections


def add_regression(figure, coefficients=None, x=None, secondary=None):

    if not isinstance(coefficients, collections.Iterable):
        return

    axes = figure.get_axes()[0]
    if x is None:
        xlim = axes.get_xlim()
        x = [value for value in range(int(xlim[0]), int(xlim[1]))]

    if secondary is not None:
        y = [get_poly_result([x_point] + [secondary[index]], coefficients) for index, x_point in enumerate(x)]
    else:
        y = [get_poly_result([x_point], coefficients) for x_point in x]

    axes.set_xlim([0, 80])
    axes.set_ylim([0, 2500])
    axes.plot(x, y, 'r--')
    figure.show()


def clean(path='../CalomirisPritchett_data.csv'):

    dataset = pd.read_csv(path)

    # remove:
    # no price
    # slaves_per_sale != 1
    # omit reason (injury, etc)
    # children
    dataset = dataset[dataset['Price'].notnull()]
    # dataset = dataset[dataset['Price'] != '.']
    dataset = dataset[dataset['Price'] != 'NaN']
    dataset = dataset[dataset['Price'].str.isdecimal()]
    dataset = dataset[dataset['Age'] != '.']
    dataset = dataset[dataset['Number of Total Slaves'] != 1]
    dataset = dataset[dataset['Reason for Omission'].isnull()]
    dataset = dataset[dataset['Family Relationship'] == '.']

    dataset['Age'] = pd.to_numeric(dataset['Age'], errors='coerce')
    dataset['Sex'] = np.where(dataset['Sex'] == 'M', 1, 0)
    dataset['Price'] = pd.to_numeric(dataset['Price'], errors='raise')

    dataset = dataset.sort_values(['Age', 'Price'])

    return dataset.as_matrix(['Age', 'Sex', 'Price'])


def get_coefficients(predictors, response, N = 1):
    """
    B = invertable( (X.transpoce() * X) ) * X.transpose() * Y
    :return: coefficients for polynom
    """

    if np.array(predictors).ndim == 1:
        X = np.vander(predictors, N, increasing=True)
        B = np.matrix(X.transpose() @ X).I @ X.transpose() @ response
    else:
        # X = np.polynomial.polynomial.polyvander2d(predictors[:, 0], predictors[:, 1], [N-1, 0 if N == 1 else 1])
        X = vander2d(predictors, N)
        B = np.matrix(X).I @ response

    return B.tolist()[0]


def get_errors(predictors, response, coefficients):
    return [get_poly_result(pr, coefficients) - response[index] for index, pr in enumerate(predictors)]


def get_mse(predictors, response, coefficients):
    sum_errors = sum([pow(err, 2) for err in get_errors(predictors, response, coefficients)])
    return sum_errors / len(predictors)


def get_poly_result(predictors, coefficients):

    if not isinstance(predictors, collections.Iterable):
        predictors = [predictors]

    if len(predictors) == 1:
        result = sum([coefficients[index] * pow(predictors[0], index) for index, value in enumerate(coefficients)])
    else:

        if len(coefficients) == 1:
            result = coefficients[0]
        else:
            degree = 0
            while pow(degree, 2) - degree*(degree-1)*0.5 < len(coefficients):
                degree += 1

            if pow(degree, 2) - degree*(degree-1)*0.5 != len(coefficients):
                raise ValueError('WTF')

            result = [1] + [pow(predictors[0], power-i) * pow(predictors[1], i) for power in range(1, degree)
                            for i in range(power+1)]
            result = np.matmul(coefficients, result)

    return result


def plot(predictor, responce):

    figure = plt.figure()
    plt.plot(predictor, responce, 'k.', markersize=1.5)
    plt.xlabel('Age')
    plt.ylabel('Price')
    plt.show()

    return figure


def vander2d(predictors, N):

    # use it
    vander = []
    for predictors_row in predictors:
        vander.append(
            [1] +
            [pow(predictors_row[0], power-i) * pow(predictors_row[1], i) for power in range(1, N)
             for i in range(power+1)]
        )

    # np.polynomial.polynomial.polyvander2d(predictors[:, 0], N-1, N-1)
    # vander = []
    # for predictors_row in predictors:
    #
    #     vander_row = []
    #     for xpower in range(N):
    #         for ypower in range(N):
    #             vander_row += [pow(predictors_row[0], xpower) * pow(predictors_row[1], ypower)]
    #     vander.append(vander_row)

    return np.array(vander)
