# coding=utf-8
"""Basic linear algebra for IO Analysis."""
from utils import *
import numpy as np

from sympy import *
from sympy.vector import matrix_to_vector, CoordSys3D


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
        print(delta_x)
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


def linkage_based_classification(x, flow, final_demand, policy):
    """
    ZAD 4
    :param x: output vector (gross outputs)
    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param final_demand: final demand vector (ndarray) -f or y.
    :param policy: eg. CO2
    :return: classification of linkages,
    reyurn_fl -standarized forward linkages,  key sectors values, indices of key sectors
    return_fl - standarized backward linkages,  key sectors values, indices of key sectors

    """
    return_bl = backward_linkages(x, flow, final_demand, policy, is_technology=False)
    return_fl = forward_linkages(x, flow, final_demand, policy)
    results_bl = return_bl[2]
    results_fl = return_fl[2]
    message = []
    for bl in list(results_bl):
        if bl in list(results_fl):
            message.append(f'Country key sectors on index {bl}')
        elif bl not in list(results_fl):
            message.append(f'Strong backward linkage country-sector on index {bl}')
    for fl in list(results_fl):
        if fl not in list(results_bl):
            message.append(f'Strong forward linkage country-sector on index {fl}')
    return message, return_fl, return_bl


def profitability(x, flow, profit):
    """
    ZAD 4
    :param x: output vector (gross outputs)
    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param profit: percentage profit - fill in as decimal
    :return:
    """
    sum1 = np.sum(flow, 0)
    z = (profit * (sum1 + x - sum1)) / (1 + profit)
    # z = np.zeros_like(x)
    # for index, x_i in enumerate(x):
    #     z[index] = (profit * (sum1[index] + x_i - sum1[index]) / (1 + profit))
    return z


def deflation_to_another_year(flow, current_prices, old_prices, x, is_technology=False):
    """

    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param current_prices: prices in the newer year
    :param old_prices: prices in the older year
    :param x: output vector (gross outputs)
    :param is_technology: check True if matrix is already technology matrix instead of flow matrix
    :return: IO table, technical coeff., vector of total outputs - deflated to the previous year value terms
    """
    if is_technology is True:
        flow = flow * x

    ratio_prices_matrix = np.zeros_like(flow, dtype=np.float16)
    row, col = np.diag_indices(ratio_prices_matrix.shape[0])
    ratio_old_current = old_prices / current_prices
    ratio_prices_matrix[row, col] = ratio_old_current

    old_flow = ratio_prices_matrix @ flow
    old_x = x @ ratio_prices_matrix
    old_tech = technology_matrix(old_flow, old_x)
    return old_flow, old_x, old_tech


def accoutinng_for_pollution_impacts(first_type_coeff, second_type_coeff, flow, x, is_technology=False):
    """
    ZAD 21 !!IMPORTANT FOR GUI: first_type_coeff, second_type_coeff are fixed size vector 1x2 since we take into account 2 sectors!!!

    :param first_type_coeff: direct impact coeff. The amount of I pollutant type generated per dollar's worth of industry.
    :param second_type_coeff: direct impact coeff. The amount of II pollutant type generated per dollar's worth of industry.
    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param x: output vector (gross outputs)
    :param is_technology: check True if matrix is already technology matrix instead of old_flow matrix
    :return: sector_I_effects, sector_II_effects respectively to the order of type coeff.
    Total impact generated
    """
    # D -Matrix of direct impact coeff. Matrix that describes the amount of pollutant type/creation of different pollutions (or pollutions and jobs etc)


    D = np.r_[[first_type_coeff], [second_type_coeff]]
    if is_technology is False:
        tech = technology_matrix(flow=flow, output=x)
    else:
        tech = flow

    L = leontief_inverse(technology=tech)
    effects_matrix = D@L
    sector_I_effects = effects_matrix[:,0]
    sector_II_effects = effects_matrix[:, 1]
    index = np.where(sector_I_effects > sector_II_effects)
    return sector_I_effects, sector_II_effects


def inverse_importance(flow, x, row:int, column:int, alpha, beta, is_technology=False):
    """

    Check if alpha % change in one element generates beta % change in one or more elements
    in the associated Leontief inverse

    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param x: output vector (gross outputs)
    :param row: row position of element that will be changed
    :param column: column position of element that will be changed
    :param alpha: change in element eg. 10 (%)
    :param beta: inverse importance factor eg. 5 (%)
    :param is_technology: check True if matrix is already technology matrix instead of old_flow matrix
    :return: percentage_reaction matrix,
    percentage_reaction[indices] - values in percentage reaction matrix higher than beta
    """
    row = row-1
    column = column-1
    if is_technology is False:
        tech_initial = technology_matrix(flow=flow, output=x)
    else:
        tech_initial = flow
    L_initial = leontief_inverse(tech_initial)
    tech_adjusted = tech_initial
    new_element = tech_adjusted[row][column] * (1+alpha/100)
    tech_adjusted[row][column] = new_element

    L_adjusted = leontief_inverse(tech_adjusted)
    percentage_reaction = ((L_adjusted - L_initial)/L_initial)*100

    indices = np.where(percentage_reaction > beta)

    return percentage_reaction, percentage_reaction[indices]

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

    print('\nNew_IO_table_based_on_delta')
    flow2 = np.array([[200, 0, 200], [300, 0, 100], [0, 400, 0]])
    x_old2 = np.array([1000, 800, 500])
    y_old2 = np.array([600, 400, 100])
    delta_vector = np.array([50, 50, 50])
    print(new_IO_table_based_on_delta(old_x=x_old2, old_y=y_old2, old_flow=flow2, delta_vector_x=None,
                                      delta_vector_y=delta_vector, type_of_delta='y'))

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

    print('\nProfitability')
    print(profitability(x=x, flow=flow, profit=0.2))

    print('\nDeflation_to_another_year')
    # flow = np.array([[24, 86, 56, 64],
    #                  [32, 15, 78, 78],
    #                  [104, 49, 62, 94],
    #                  [14, 16, 63, 78]])
    flow = np.array([[0.06030151, 0.27388535, 0.11940299, 0.14096916],
                     [0.08040201, 0.0477707, 0.1663113, 0.17180617],
                     [0.26130653, 0.15605096, 0.13219616, 0.20704846],
                     [0.03517588, 0.05095541, 0.13432836, 0.17180617]])
    price_2005 = np.array([5, 6, 9, 12])
    price_2000 = np.array([2, 3, 5, 7])
    x = np.array([398, 314, 469, 454])
    print(
        deflation_to_another_year(flow=flow, current_prices=price_2005, old_prices=price_2000, x=x, is_technology=True))

    print('\nAccoutinng_for_pollution_impacts')
    pollutions = np.array([0.3, 0.5])
    jobs = np.array([0.005, 0.07])
    flow = np.array(([[140, 350], [800, 50]]))
    x = np.array([1000, 1000])
    print(accoutinng_for_pollution_impacts(first_type_coeff=pollutions, second_type_coeff=jobs, flow=flow, x=x))


    # A = np.array([[0.168, 0.155, 0.213, 0.212],
    #              [0.194, 0.193, 0.168, 0.115],
    #              [0.105, 0.025, 0.126, 0.124],
    #              [0.178, 0.101, 0.219, 0.186]])
    #
    #
    # print(leontief_inverse(technology=A))

    print('\nInverse importance')
    flow = np.array([[8, 64, 89],
                    [28, 44, 77],
                    [48, 24, 28]])
    x = np.array([300, 250, 200])
    inverse_importance(flow=flow, x=x, row=1, column=3, alpha=10, beta=2)
