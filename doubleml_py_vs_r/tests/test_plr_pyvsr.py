import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import doubleml as dml
from _utils_pyvsr import export_smpl_split_to_r, r_MLPLR

rpy2 = pytest.importorskip("rpy2")
from rpy2.robjects import pandas2ri
pandas2ri.activate()


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_pyvsr_fixture(generate_data_plr, score, dml_procedure, n_rep):
    n_folds = 2

    # collect data
    obj_dml_data = generate_data_plr

    # Set machine learning methods for l, m & g
    learner = LinearRegression()
    ml_l = clone(learner)
    ml_m = clone(learner)
    if score == 'IV-type':
        ml_g = clone(learner)
    else:
        ml_g = None

    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_l, ml_m, ml_g,
                                  n_folds=n_folds,
                                  n_rep=n_rep,
                                  score=score,
                                  dml_procedure=dml_procedure)

    # np.random.seed(3141)
    dml_plr_obj.fit()

    # fit the DML model in R
    smpls_for_r = list()
    for i_rep in range(n_rep):
        all_train, all_test = export_smpl_split_to_r(dml_plr_obj.smpls[i_rep])
        smpls_for_r.append([all_train, all_test])

    r_dataframe = pandas2ri.py2rpy(obj_dml_data.data)
    res_r = r_MLPLR(r_dataframe, score, dml_procedure, n_rep, smpls_for_r)

    res_dict = {'coef_py': dml_plr_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_plr_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_plr_pyvsr_coef(dml_plr_pyvsr_fixture):
    assert math.isclose(dml_plr_pyvsr_fixture['coef_py'],
                        dml_plr_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_pyvsr_se(dml_plr_pyvsr_fixture):
    assert math.isclose(dml_plr_pyvsr_fixture['se_py'],
                        dml_plr_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)
