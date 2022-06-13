import numpy as np

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




def ghosh_technology_matrix(x, flow):
    return flow / x[:, None]



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




def forward_linkages(x, flow, final_demand, policy):
    """
    ZAD 4 EXCEL


    :param policy: eg. CO2
    :param x: output vector (gross outputs)
    :param flow: intersectoral flow matrix (ndarray) - IO table
    :param final_demand: final demand vector (ndarray) -f or y.
    :return: standarized forward linkages,  key sectors values, indices of key sectors
    """
    E = ghosh_technology_matrix(x, flow)
    G = leontief_inverse(E)

    pi_policy = policy / x
    forward_link = G.T @ pi_policy.T
    standarized_forward = forward_link / forward_link.mean()
    results = standarized_forward[standarized_forward > 1]
    index = np.where(standarized_forward > 1)

    return standarized_forward, results, index[0]
