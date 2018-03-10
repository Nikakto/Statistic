import matplotlib.pyplot as plt
import numpy as np

from Lab1.dataset import (
    add_regression,
    clean,
    get_coefficients,
    get_errors,
    get_mse,
    plot
)


if __name__ == '__main__':

    degree = 15
    dataset = clean('../CalomirisPritchett_data.csv')
    predictors, response = dataset[:, :2], dataset[:, 2]

    mse = [0] * degree
    for block_k in range(10):

        predictors_learn, response_learn = predictors, response
        predictors_test, response_test = [], []

        for i in range(block_k, len(predictors), 9):
            predictors_test.append(predictors[i]), response_test.append(response[i])
            np.delete(predictors_learn, i), np.delete(response_learn, i)
        predictors_test, response_test = np.array(predictors_test), np.array(response_test)

        coefficients = [get_coefficients(predictors_learn[:, 0], response_learn, N) for N in range(1, degree)]
        mse = [mse[N] + get_mse(predictors_test[:, 0], response_test, coefficients[N]) for N in range(len(coefficients))]

    mse = [mse[N] / 10 for N in range(degree-1)]

    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.plot(range(len(coefficients)), mse, 'k-')
    # plt.ylim([63250, 64000])
    plt.show()

    coefficients = get_coefficients(predictors[:, 0], response, mse.index(min(mse)))
    figure = plot(predictors[:, 0], response)
    add_regression(figure, coefficients)

    get_errors(predictors[:, 0], response, coefficients)

    # TWO PREDICTORS

    mse = [0] * degree
    for block_k in range(10):

        predictors_learn, response_learn = predictors, response
        predictors_test, response_test = [], []

        for i in range(block_k, len(predictors), 9):
            predictors_test.append(predictors[i]), response_test.append(response[i])
            np.delete(predictors_learn, i), np.delete(response_learn, i)
        predictors_test, response_test = np.array(predictors_test), np.array(response_test)

        coefficients = [get_coefficients(predictors_learn, response_learn, N) for N in range(1, degree)]
        mse = [mse[N] + get_mse(predictors_test, response_test, coefficients[N]) for N in range(len(coefficients))]

    mse = [mse[N] / 10 for N in range(degree-1)]

    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.plot(range(len(coefficients)), mse, 'k-')
    # plt.ylim([58450, 59500])
    plt.show()

    # figure = plot(predictors[:, 0], response)
    coefficients = get_coefficients(predictors, response, mse.index(min(mse)))
    # add_regression(figure, coefficients[3], predictors[:, 0], secondary=predictors[:, 1])

    print('ok')

    # 3d
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs=predictors[:, 0], ys=predictors[:, 1], zs=response, linestyle='None', c='k', marker='.', markersize=1.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('')
    ax.set_ylim(-1, 2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Woman', 'Man'])
    ax.set_zlabel('Price')
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    from Lab1.dataset import get_poly_result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs=predictors[:, 0], ys=predictors[:, 1], zs=response, linestyle='None', c='k', marker='.', markersize=1.5)

    ax.set_xlabel('Age')
    ax.set_ylabel('')

    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Woman', 'Man'])

    ax.set_zlabel('Price')
    surf_data = {
        'x': np.outer(np.ones((2,)), np.linspace(0, 65, 65)),
        'y': np.outer(np.linspace(0, 1, 2), np.ones((65,))),
        'z': np.array([[get_poly_result([x,y], coefficients) for x in range(65)]
                       for y in [0, 1]])
    }
    surf = ax.plot_surface(X=surf_data['x'], Y=surf_data['y'], Z=surf_data['z'])
    surf.set_facecolor((0, 0.25, 0.5, 0.85))
    plt.show()

    # proection

    x = [i for i in range(70)]
    y = [[get_poly_result([xx, sex], coefficients) for xx in x] for sex in [0, 1]]

    pr_man = []
    pr_woman = []

    resp_man = []
    resp_woman = []

    for index, row in enumerate(predictors):
        if row[1] == 0:
            pr_woman.append(row)
            resp_woman.append(response[index])
        else:
            pr_man.append(row)
            resp_man.append(response[index])

    plt.title('Woman slave price')
    plt.plot(np.array(pr_woman)[:, 0], resp_woman, 'k.', markersize=1.5)
    plt.xlabel('Age')
    plt.ylabel('Price')
    plt.ylim([0, 3000])
    plt.plot(x, y[0], 'r--')
    plt.show()

    plt.title('Man slave price')
    plt.plot(np.array(pr_man)[:, 0], resp_man, 'k.', markersize=1.5)
    plt.xlabel('Age')
    plt.ylabel('Price')
    plt.ylim([0, 3000])
    plt.plot(x, y[0], 'b--')
    plt.show()