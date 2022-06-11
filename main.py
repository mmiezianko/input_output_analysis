# coding=utf-8
"""Basic linear algebra for IO Analysis."""

import numpy as np

from sympy import *
from sympy.vector import matrix_to_vector, CoordSys3D


def gross_production_calc(technology, demand):
    """Calculate gross production based on demand and technology matrix.
        Formula:
        X = (I - A) ^ -1 d

    :param technology: technology matrix (ndarray).
    :param demand: demand vector (ndarray).
    :return:gross production calculation (ndarray).
    """
    return leontief_inverse(technology) @ demand


def technology_matrix(flow, output):
    """Get technology matrix (Leontief matrix) based on intersectoral flow matrix and output vector.


        :param flow: intersectoral flow matrix (ndarray) - IO table
        :param output: economy output VECTOR (ndarray) - vector of global production

    Returns:
        technology matrix (ndarray).
    """
    return flow / output


def mean_relative_error(ground_truth, estimate):
    """Calculate mean relative error of given estimate with respect to given ground truth vector.

    Args:
        ground_truth: true value of calculation (ndarray).
        estimate: model estimate (ndarray).

    Returns:
        mean relative error.
    """
    return np.mean((ground_truth - estimate) / ground_truth)


def leontief_inverse(technology):
    """Calculate Leontief inverse.

    Formula:
        L = (I - A) ^ -1

    :param technology: technology matrix (ndarray).
    :return Leontief inverse (ndarray).
    """

    return np.linalg.inv(np.identity(technology.shape[0]) - technology)


def final_demand_vector(flow, x, is_technology=False):
    """
    ZAD 8 A)
    Calculate final demand vector based on flow matrix and x (output vector)
    :param technology: technology matrix
    :param x: output vector (gross outputs)
    :param is_technology: check True if maatrix is already technology matrix instead of flow matrix
    :return: final demand vector (y)
    """
    if is_technology is True:
        L = np.identity(flow.shape[0]) - flow

    else:
        technology = technology_matrix(flow, x)
        L = np.identity(technology.shape[0]) - technology

    return L @ x


def taylor_series_estimate(technology, final_demand, converge_threshold=None, x_old=None, is_technology=True,
                           number_of_iterations=None):
    """
    ZAD 8 B)
    Calculate Leontief inverse estimate using matrix power series.

    Formula:
        L = I + A + A^2 + A^3 + A^4 + ...
        X = L * Y

    :param is_technology: check True if matrix is already technology matrix instead of flow matrix
    :param x_old: optional parameter if param is_technology = False. Output vector (gross outputs)
    :param number_of_iterations: level of convergence when algorithm will stop.
    :param converge_threshold: level of convergence when algorithm will stop.
    :param technology: technology matrix (ndarray).
    :param final_demand: final demand vector (ndarray) -f or y.


    Returns:
        n - number of rounds economy needs to converge with given threshold to balanced state.
        convex_result - exact result made by using the Leontief inverse
        x_new - new gross outputs
    """

    if is_technology is False:
        technology = technology_matrix(technology, x_old)

    convex_result = gross_production_calc(technology, final_demand)

    print("Calculating Taylor series estimate of Leontief inverse.")
    n = 0
    inverse_estimate = np.identity(technology.shape[0])
    taylor_expansion_term = technology
    if converge_threshold is not None and number_of_iterations is None:
        while mean_relative_error(convex_result,
                                  inverse_estimate @ final_demand) > converge_threshold or number_of_iterations is not None and number_of_iterations > n:
            inverse_estimate += taylor_expansion_term
            taylor_expansion_term = taylor_expansion_term @ technology

            n += 1
        x_new = inverse_estimate @ final_demand
    elif number_of_iterations is not None and converge_threshold is None:
        while number_of_iterations > n:
            inverse_estimate += taylor_expansion_term
            taylor_expansion_term = taylor_expansion_term @ technology
            n += 1
        x_new = inverse_estimate @ final_demand
    else:
        message = 'Fill in one converge treshold'
        return message
    return n, x_new, convex_result


def new_IO_table_based_on_delta(old_x, old_flow, old_y, delta_vector_x, delta_vector_y, type_of_delta: str,
                                is_technology=False, old_x_np=None):
    """
    ZAD 5
    Based on the given IO table for the specified year find the IO table for the another (next) year if considering
    the delta of X or Y:

    Formulas:
    X = L * Y

    FLOW = (I-A) ^-1 * DELTA_Y + A

    :param old_x: output vector (gross outputs)
    :param old_flow: intersectoral flow matrix (ndarray) - IO table
    :param old_y: final demand vector (ndarray) -f or y.
    :param delta_vector: delta vector of x or final demand (y)
    :param type_of_delta: describve if delta_vector is of x or y type
    :return new_flow: new intersectoral flow matrix (ndarray) - IO table based on delta vector
    :param is_technology: check True if matrix is already technology matrix instead of old_flow matrix
    """

    if type_of_delta.upper() == 'X':
        new_x = old_x + delta_vector_x
        if is_technology is False:
            tech = technology_matrix(flow=old_flow, output=old_x)
        else:
            tech = old_flow
        new_IO_table = tech * new_x
        return new_IO_table


    elif type_of_delta.upper() == 'Y':
        # FLOW = (I - A) ^ -1 * DELTA_Y + A
        if is_technology is False:
            tech = technology_matrix(flow=old_flow, output=old_x)
        else:
            tech = old_flow
        L = leontief_inverse(technology=tech)
        new_y = old_y + delta_vector_y
        delta_x = L @ new_y  # correct
        # TODO: CHECK IF THIS IS CORRECT
        new_IO_table = tech * delta_x
        print(new_IO_table)
        return new_IO_table

        # TODO: XY
    # elif type_of_delta.upper() == 'XY' or type_of_delta.upper() == 'YX':
    #     # old_y_sympy = Matrix(old_y)
    #     # delta_vector_y_sympy = Matrix(delta_vector_y)
    #     # new_y = old_y_sympy + delta_vector_y_sympy
    #     # print(new_y)
    #     # old_x_sympy = Matrix(old_x)
    #     # delta_vector_x_sympy = Matrix(delta_vector_x)
    #     # new_x = old_x_sympy+ delta_vector_x_sympy
    #     new_x = old_x + delta_vector_x
    #     print(new_x)
    #     new_y = old_y + delta_vector_y
    #     print(new_y)
    #     new_matrix = Matrix()
    #
    #
    #     if is_technology is False:
    #
    #         tech = Matrix(technology_matrix(flow=old_flow, output=old_x_np))
    #     else:
    #         tech = Matrix(old_flow)
    #     print(tech)
    #
    #     for i in range(new_x.shape[0]):
    #         print(new_x[i])
    #         # print(tech.col(i))
    #         a = new_x[i]*tech.col(i)
    #         print(a)
    #         list = []
    #         list.append(a)
    #         # print(a[0]+a[1]+a[2])
    #         # b = new_y[i]
    #         # # print(solve(a, -b, symbols('a, b, c ')))
    #         #
    #     print(list)


def backward_linkages(x, flow, final_demand, policy, is_technology=False):
    """
    ZAD 4 EXCEL

    :param is_technology: check True if matrix is already technology matrix instead of flow matrix
    :param policy: eg. CO2
    :param x: output vector (gross outputs)
    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param final_demand: final demand vector (ndarray) -f or y.
    :return: standarized backward linkages,  key sectors values, indices of key sectors
    """
    if is_technology is False:
        tech = technology_matrix(flow=flow, output=x)
    else:
        tech = flow

    L = leontief_inverse(tech)
    pi_policy = policy / x
    backward_link = L.T @ pi_policy.T
    standarized_backward = backward_link / backward_link.mean()
    results = standarized_backward[standarized_backward > 1]
    index = np.where(standarized_backward > 1)

    return standarized_backward, results, index[0]


def ghosh_technology_matrix(x, flow):
    return flow / x[:, None]


def forward_linkages(x, flow, final_demand, policy):
    """
    ZAD 4 EXCEL


    :param policy: eg. CO2
    :param x: output vector (gross outputs)
    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param final_demand: final demand vector (ndarray) -f or y.
    :return: standarized backward linkages,  key sectors values, indices of key sectors
    """
    E = ghosh_technology_matrix(x, flow)
    G = leontief_inverse(E)

    pi_policy = policy / x
    forward_link = G.T @ pi_policy.T
    standarized_forward = forward_link / forward_link.mean()
    results = standarized_forward[standarized_forward > 1]
    index = np.where(standarized_forward > 1)

    return standarized_forward, results, index[0]


def linkage_based_classification(x, flow, final_demand, policy):
    results_bl = backward_linkages(x, flow, final_demand, policy, is_technology=False)[2]
    results_fl = forward_linkages(x, flow, final_demand, policy)[2]
    message = []
    for bl in list(results_bl):
        if bl in list(results_fl):
            message.append(f'Country key sectors on index {bl}')
        elif bl not in list(results_fl):
            message.append(f'Strong backward linkage country-sector on index {bl}')
    for fl in list(results_fl):
        if fl not in list(results_bl):
            message.append(f'Strong forward linkage country-sector on index {fl}')
    return message




if __name__ == '__main__':
    # from sympy import linear_eq_to_matrix, symbols
    # a, b, c = symbols('a, b, c ')
    # flow2 = np.array([[200, 0, 200], [300, 0, 100], [0, 400, 0]])
    # x_old2 = Matrix([1000, 800, 500])
    # x_old2_np = np.array([1000, 800, 500])
    # y_old2 = Matrix([600, 400, 100])
    # delta_vector_x = Matrix([200, 100, 'a'])
    # delta_vector_y = Matrix(['b', 'c', 0])
    # new_IO_table_based_on_delta(old_x=x_old2, old_y=y_old2, old_flow=flow2, delta_vector_x=delta_vector_x, delta_vector_y=delta_vector_y,
    #                             type_of_delta='xy', old_x_np=x_old2_np)

    print("Linear Algebra for IO Analysis.")
    tech = np.array([[500, 350], [320, 360]])
    dem = np.array([200, 100])
    x_old = np.array([[1000, 800]])
    print(taylor_series_estimate(tech, dem, number_of_iterations=2, is_technology=False, x_old=x_old))
    print("\n")

    tech = np.array([[0.05, 0.2, 0.025], [0.05, 0.1, 0.025], [0.6, 0.6, 0.15]])
    dem = np.array([35 * 3 / 4, 36, 25 * 0.95])
    print(gross_production_calc(tech, dem))

    flow = np.array(([[240, 90], [180, 180]]))
    output = np.array(([420, 540]))
    print('\ntech matrix')
    print(technology_matrix(flow, output))

    print('\nfinal_demand_vector (Y)')
    # flow1 = np.array(([[500, 350], [320, 360]]))
    # output1 = np.array(([1000, 800]))
    flow1 = np.array(([[3 / 10, 2 / 5], [1 / 10, 1 / 2]]))
    output1 = np.array(([200, 300]))
    print(final_demand_vector(flow1, output1, is_technology=True))

    flow2 = np.array([[200, 0, 200], [300, 0, 100], [0, 400, 0]])
    x_old2 = np.array([1000, 800, 500])
    y_old2 = np.array([600, 400, 100])
    delta_vector = np.array([50, 50, 50])
    new_IO_table_based_on_delta(old_x=x_old2, old_y=y_old2, old_flow=flow2, delta_vector_x=None,
                                delta_vector_y=delta_vector, type_of_delta='y')

    print('\nBackward linkage')
    x = np.array([1000, 800, 500])
    flow = np.array([[200, 0, 200], [300, 0, 100], [0, 400, 0]])
    y = np.array([600, 400, 100])
    policy = y = np.array([20, 30, 10])
    results = backward_linkages(x=x, flow=flow, final_demand=y, policy=policy, is_technology=False)
    print(results)

    print('\nForward linkage')
    ghosh_technology_matrix(x, flow)
    print(forward_linkages(x=x, flow=flow, final_demand=y, policy=policy))

    print('\nClassification')
    print(linkage_based_classification(x=x, flow=flow, final_demand=y, policy=policy))
