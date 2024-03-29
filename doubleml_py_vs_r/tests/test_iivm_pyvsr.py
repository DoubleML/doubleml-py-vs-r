import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression

import doubleml as dml
from _utils_pyvsr import export_smpl_split_to_r, r_IIVM

rpy2 = pytest.importorskip("rpy2")
from rpy2.robjects import pandas2ri
pandas2ri.activate()


@pytest.fixture(scope='module',
                params=['LATE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_iivm_pyvsr_fixture(generate_data_iivm, score, dml_procedure):
    n_folds = 2

    # collect data
    obj_dml_data = generate_data_iivm

    # Set machine learning methods for m & gg
    learner_classif = LogisticRegression(penalty='none', solver='newton-cg')
    learner_reg = LinearRegression()
    ml_g = clone(learner_reg)
    ml_m = clone(learner_classif)
    ml_r = clone(learner_classif)

    dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds=n_folds,
                                    dml_procedure=dml_procedure)

    np.random.seed(3141)
    dml_iivm_obj.fit()

    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_iivm_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(obj_dml_data.data)
    res_r = r_IIVM(r_dataframe, score, dml_procedure,
                   all_train, all_test)

    res_dict = {'coef_py': dml_iivm_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_iivm_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_iivm_pyvsr_coef(dml_iivm_pyvsr_fixture):
    assert math.isclose(dml_iivm_pyvsr_fixture['coef_py'],
                        dml_iivm_pyvsr_fixture['coef_r'],
                        rel_tol=1e-4, abs_tol=1e-1)


def test_dml_iivm_pyvsr_se(dml_iivm_pyvsr_fixture):
    assert math.isclose(dml_iivm_pyvsr_fixture['se_py'],
                        dml_iivm_pyvsr_fixture['se_r'],
                        rel_tol=1e-4, abs_tol=1e-1)
