import pytest

rpy2 = pytest.importorskip("rpy2")
from rpy2 import robjects
from rpy2.robjects import ListVector
from rpy2.robjects.vectors import IntVector


def export_smpl_split_to_r(smpls):
    n_smpls = len(smpls)
    all_train = ListVector.from_length(n_smpls)
    all_test = ListVector.from_length(n_smpls)

    for idx, (train, test) in enumerate(smpls):
        all_train[idx] = IntVector(train + 1)
        all_test[idx] = IntVector(test + 1)

    return all_train, all_test


# The R code to fit the DML model
r_MLPLR = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, n_rep, smpls_for_r) {
            data = data.table(data)
            mlmethod_m = 'regr.lm'
            mlmethod_g = 'regr.lm'

            Xnames = names(data)[names(data) %in% c("y", "d") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames)

            double_mlplr_obj = DoubleMLPLR$new(data_ml,
                                               n_folds = 2,
                                               ml_g = mlmethod_g,
                                               ml_m = mlmethod_m,
                                               dml_procedure = dml_procedure,
                                               score = score)
            smpls = list()
            for (i_rep in 1:n_rep) {
                this_smpl = smpls_for_r[[i_rep]]
                smpls[[i_rep]] = list(train_ids=this_smpl[[1]], test_ids=this_smpl[[2]])
            }
            double_mlplr_obj$set_sample_splitting(smpls)

            double_mlplr_obj$fit()
            print(data_ml)
            return(list(coef = double_mlplr_obj$coef,
                        se = double_mlplr_obj$se))
        }
        ''')


r_MLPLIV = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[names(data) %in% c("y", "d", "Z1") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = "Z1")

            double_mlpliv_obj = DoubleMLPLIV$new(data_ml,
                                                 n_folds = 2,
                                                 ml_g = mlmethod_g,
                                                 ml_m = mlmethod_m,
                                                 ml_r = mlmethod_r,
                                                 dml_procedure = dml_procedure,
                                                 score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_MLPLIV_PARTIAL_X = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[grepl('X', names(data))]
            Znames = names(data)[grepl('Z', names(data))]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = Znames)

            double_mlpliv_obj = DoubleML:::DoubleMLPLIV.partialX(data_ml,
                                                      n_folds = 2,
                                                      ml_g = mlmethod_g,
                                                      ml_m = mlmethod_m,
                                                      ml_r = mlmethod_r,
                                                      dml_procedure = dml_procedure,
                                                      score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_MLPLIV_PARTIAL_Z = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[grepl('X', names(data))]
            Znames = names(data)[grepl('Z', names(data))]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = Znames)

            double_mlpliv_obj = DoubleML:::DoubleMLPLIV.partialZ(data_ml,
                                                      n_folds = 2,
                                                      ml_r = mlmethod_r,
                                                      dml_procedure = dml_procedure,
                                                      score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_MLPLIV_PARTIAL_XZ = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[grepl('X', names(data))]
            Znames = names(data)[grepl('Z', names(data))]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = Znames)

            double_mlpliv_obj = DoubleML:::DoubleMLPLIV.partialXZ(data_ml,
                                                       n_folds = 2,
                                                       ml_g = mlmethod_g,
                                                       ml_m = mlmethod_m,
                                                       ml_r = mlmethod_r,
                                                       dml_procedure = dml_procedure,
                                                       score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_IRM = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'classif.log_reg'

            Xnames = names(data)[names(data) %in% c("y", "d") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames)

            double_mlirm_obj = DoubleMLIRM$new(data_ml,
                                               n_folds = 2,
                                               ml_g = mlmethod_g,
                                               ml_m = mlmethod_m,
                                               dml_procedure = dml_procedure,
                                               score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlirm_obj$set_sample_splitting(smpls)

            double_mlirm_obj$fit()
            return(list(coef = double_mlirm_obj$coef,
                        se = double_mlirm_obj$se))
        }
        ''')


r_IIVM = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            ml_g = 'regr.lm'
            ml_m = 'classif.log_reg'
            ml_r = 'classif.log_reg'

            Xnames = names(data)[names(data) %in% c("y", "d", "z") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y",
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = "z")

            double_mliivm_obj = DoubleMLIIVM$new(data_ml,
                                                 n_folds = 2,
                                                 ml_g = ml_g,
                                                 ml_m = ml_m,
                                                 ml_r = ml_r,
                                                 dml_procedure = dml_procedure,
                                                 score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mliivm_obj$set_sample_splitting(smpls)

            double_mliivm_obj$fit()
            return(list(coef = double_mliivm_obj$coef,
                        se = double_mliivm_obj$se))
        }
        ''')

r_MLPLIV_multiway_cluster = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure,
                      train_ids, test_ids,
                      cluster_var1, cluster_var2=NULL) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'

            if (is.null(cluster_var2)) cluster_vars = cluster_var1 else cluster_vars = c(cluster_var1, cluster_var2)
            Xnames = names(data)[names(data) %in% c("Y", "D", "Z", cluster_vars) == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "Y",
                                                     d_cols = "D", x_cols = Xnames,
                                                     z_col = "Z",
                                                     cluster_cols = cluster_vars)

            double_mlpliv_obj = DoubleMLPLIV$new(data_ml,
                                                 n_folds = 2,
                                                 ml_g = mlmethod_g,
                                                 ml_m = mlmethod_m,
                                                 ml_r = mlmethod_r,
                                                 dml_procedure = dml_procedure,
                                                 score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            smpls_cluster = list(train_ids = list(), test_ids = list())
            for (i_smpl in 1:length(train_ids)) {
                this_cluster_smpl_train = list()
                this_cluster_smpl_test = list()
                for (i_var in 1:length(cluster_vars)) {
                    this_cluster_var = data[[cluster_vars[i_var]]]
                    train_clusters = unique(this_cluster_var[train_ids[[i_smpl]]])
                    test_clusters = unique(this_cluster_var[test_ids[[i_smpl]]])
                    this_cluster_smpl_train[[i_var]] = train_clusters
                    this_cluster_smpl_test[[i_var]] = test_clusters
                }
                smpls_cluster$train_ids[[i_smpl]] = this_cluster_smpl_train
                smpls_cluster$test_ids[[i_smpl]] = this_cluster_smpl_test
            }
            double_mlpliv_obj$.__enclos_env__$private$smpls_ = smpls
            double_mlpliv_obj$.__enclos_env__$private$smpls_cluster_ = list(smpls_cluster)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')
