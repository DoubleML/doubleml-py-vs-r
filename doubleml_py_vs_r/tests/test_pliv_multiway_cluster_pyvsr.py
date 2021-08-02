import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import doubleml as dml
from _utils_pyvsr import export_smpl_split_to_r, r_MLPLIV_multiway_cluster

rpy2 = pytest.importorskip("rpy2")
from rpy2.robjects import pandas2ri
pandas2ri.activate()


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_twoway_cluster_pyvsr_fixture(generate_data_pliv_two_way_cluster, dml_procedure):
    n_folds = 2

    # collect data
    obj_dml_data = generate_data_pliv_two_way_cluster

    # Set machine learning methods for g, m & r
    learner = LinearRegression()
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure)
    print(obj_dml_data)
    dml_pliv_obj.fit()

    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(obj_dml_data.data)
    res_r = r_MLPLIV_multiway_cluster(r_dataframe, 'partialling out', dml_procedure,
                                      all_train, all_test,
                                      obj_dml_data.cluster_cols[0],
                                      obj_dml_data.cluster_cols[1])

    res_dict = {'coef_py': dml_pliv_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_pliv_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_pliv_twoway_cluster_pyvsr_coef(dml_pliv_twoway_cluster_pyvsr_fixture):
    assert math.isclose(dml_pliv_twoway_cluster_pyvsr_fixture['coef_py'],
                        dml_pliv_twoway_cluster_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_twoway_cluster_pyvsr_se(dml_pliv_twoway_cluster_pyvsr_fixture):
    assert math.isclose(dml_pliv_twoway_cluster_pyvsr_fixture['se_py'],
                        dml_pliv_twoway_cluster_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_pliv_one_cluster_pyvsr_fixture(generate_data_pliv_one_way_cluster, dml_procedure):
    n_folds = 2

    # collect data
    obj_dml_data = generate_data_pliv_one_way_cluster

    # Set machine learning methods for g, m & r
    learner = LinearRegression()
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure)
    dml_pliv_obj.fit()

    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(obj_dml_data.data.drop(columns='cluster_var_j'))
    res_r = r_MLPLIV_multiway_cluster(r_dataframe, 'partialling out', dml_procedure,
                                      all_train, all_test,
                                      obj_dml_data.cluster_cols[0])

    res_dict = {'coef_py': dml_pliv_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_pliv_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_pliv_one_cluster_pyvsr_coef(dml_pliv_one_cluster_pyvsr_fixture):
    assert math.isclose(dml_pliv_one_cluster_pyvsr_fixture['coef_py'],
                        dml_pliv_one_cluster_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_one_cluster_pyvsr_se(dml_pliv_one_cluster_pyvsr_fixture):
    assert math.isclose(dml_pliv_one_cluster_pyvsr_fixture['se_py'],
                        dml_pliv_one_cluster_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)
