import numpy as np
import pytest

from doubleml import DoubleMLData
from doubleml.datasets import make_plr_turrell2018, make_irm_data, make_iivm_data, make_pliv_CHS2015
from doubleml.tests.conftest import make_data_pliv_partialZ


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_plr(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    data = make_plr_turrell2018(N, p, theta)

    return data


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_irm(request):
    N_p = request.param
    np.random.seed(2222)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    data = make_irm_data(N, p, theta)

    return data


@pytest.fixture(scope='session',
                params=[(500, 11)])
def generate_data_iivm(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5
    gamma_z = 0.4

    # generating data
    data = make_iivm_data(N, p, theta, gamma_z)

    return data


@pytest.fixture(scope='session',
                params=[(500, 10),
                        (1000, 20),
                        (1000, 100)])
def generate_data_pliv(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    data = make_pliv_CHS2015(n_obs=N, dim_x=p, alpha=theta, dim_z=1)

    return data


@pytest.fixture(scope='session',
                params=[(500, 100)])
def generate_data_pliv_partialXZ(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 1.

    # generating data
    data = make_pliv_CHS2015(N, alpha=theta, dim_x=p, dim_z=50)

    return data


@pytest.fixture(scope='session',
                params=[(500, 20)])
def generate_data_pliv_partialX(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 1.

    # generating data
    data = make_pliv_CHS2015(N, alpha=theta, dim_z=5, dim_x=p)

    return data


@pytest.fixture(scope='session',
                params=[(500, 5)])
def generate_data_pliv_partialZ(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 1.

    # generating data
    x_cols = [f'X{i + 1}' for i in np.arange(p)]
    z_cols = [f'Z{i + 1}' for i in np.arange(150)]
    df = make_data_pliv_partialZ(N, alpha=theta, dim_x=p, dim_z=150)
    data = DoubleMLData(df, 'y', 'd', x_cols, z_cols)

    return data
