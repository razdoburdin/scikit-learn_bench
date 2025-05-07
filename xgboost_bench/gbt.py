# ===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

from typing import Callable

import argparse

import bench
import numpy as np
import xgboost as xgb

class IterForDMatrixDemo(xgb.core.DataIter):
    """A data iterator for XGBoost DMatrix.

    `reset` and `next` are required for any data iterator, other functions here
    are utilites for demonstration's purpose.

    """
    def __init__(self, X_train, y_train, batch_size=int(1e7)) -> None:
        self.batch_size = np.minimum(batch_size, X_train.shape[0])
        self._data = X_train.values
        self._labels = y_train.values
        self._labels.shape = (y_train.shape[0], 1)

        self.it = 0  # set iterator to 0
        super().__init__()

    def data(self):
        """Utility function for obtaining current batch of data."""
        begin = self.it
        end = np.minimum(self.it + self.batch_size, len(self._data))
        return self._data[begin:end, :]

    def labels(self):
        """Utility function for obtaining current batch of label."""
        begin = self.it
        end = np.minimum(self.it + self.batch_size, len(self._data))
        return self._labels[begin:end]

    def reset(self) -> None:
        """Reset the iterator"""
        self.it = 0

    def next(self, input_data: Callable) -> bool:
        """Yield the next batch of data."""
        if self.it >= len(self._data):
            # Return False to let XGBoost know this is the end of iteration
            return False

        # input_data is a keyword-only function passed in by XGBoost and has the similar
        # signature to the ``DMatrix`` constructor.
        input_data(data=self.data(), label=self.labels())
        self.it += self.batch_size
        return True


def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_xgb_predictions(y_pred, objective):
    if objective == 'multi:softprob':
        y_pred = convert_probs_to_classes(y_pred)
    elif objective == 'binary:logistic':
        y_pred = (y_pred >= 0.5).astype(np.int32)
    return y_pred


parser = argparse.ArgumentParser(description='xgboost gradient boosted trees benchmark')


parser.add_argument('--colsample-bytree', type=float, default=1,
                    help='Subsample ratio of columns '
                         'when constructing each tree')
parser.add_argument('--count-dmatrix', default=False, action='store_true',
                    help='Count DMatrix creation in time measurements')
parser.add_argument('--enable-experimental-json-serialization', default=True,
                    choices=('True', 'False'), help='Use JSON to store memory snapshots')
parser.add_argument('--grow-policy', type=str, default='depthwise',
                    help='Controls a way new nodes are added to the tree')
parser.add_argument('--inplace-predict', default=False, action='store_true',
                    help='Perform inplace_predict instead of default')
parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                    help='Step size shrinkage used in update '
                         'to prevents overfitting')
parser.add_argument('--max-bin', type=int, default=256,
                    help='Maximum number of discrete bins to '
                         'bucket continuous features')
parser.add_argument('--max-delta-step', type=float, default=0,
                    help='Maximum delta step we allow each leaf output to be')
parser.add_argument('--max-depth', type=int, default=6,
                    help='Maximum depth of a tree')
parser.add_argument('--max-leaves', type=int, default=0,
                    help='Maximum number of nodes to be added')
parser.add_argument('--min-child-weight', type=float, default=1,
                    help='Minimum sum of instance weight needed in a child')
parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')
parser.add_argument('--n-estimators', type=int, default=100,
                    help='The number of gradient boosted trees')
parser.add_argument('--objective', type=str, required=True,
                    choices=('reg:squarederror', 'binary:logistic',
                             'multi:softmax', 'multi:softprob'),
                    help='Specifies the learning task')
parser.add_argument('--reg-alpha', type=float, default=0,
                    help='L1 regularization term on weights')
parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')
parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')
parser.add_argument('--single-precision-histogram', default=False, action='store_true',
                    help='Build histograms instead of double precision')
parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')
parser.add_argument('--tree-method', type=str, required=True,
                    help='The tree construction algorithm used in XGBoost')
parser.add_argument('--device_name', type=str, required=True,
                    help='Device')

params = bench.parse_args(parser)
# Default seed
if params.seed == 12345:
    params.seed = 0

# Load and convert data
X_train, X_test, y_train, y_test = bench.load_data(params)

xgb_params = {
    'booster': 'gbtree',
    'verbosity': 3,
    'learning_rate': params.learning_rate,
    'min_split_loss': params.min_split_loss,
    'max_depth': params.max_depth,
    'min_child_weight': params.min_child_weight,
    'max_delta_step': params.max_delta_step,
    'subsample': params.subsample,
    'sampling_method': 'uniform',
    'colsample_bytree': params.colsample_bytree,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'reg_lambda': params.reg_lambda,
    'reg_alpha': params.reg_alpha,
    'tree_method': params.tree_method,
    'device': params.device_name,
    'scale_pos_weight': params.scale_pos_weight,
    'grow_policy': params.grow_policy,
    'max_leaves': params.max_leaves,
    'max_bin': params.max_bin,
    'objective': params.objective,
    'seed': params.seed,
    'single_precision_histogram': params.single_precision_histogram,
    'enable_experimental_json_serialization':
        params.enable_experimental_json_serialization
}

if params.threads != -1:
    xgb_params.update({'nthread': params.threads})

if params.objective.startswith('reg'):
    task = 'regression'
    metric_name, metric_func = 'rmse', bench.rmse_score
else:
    task = 'classification'
    metric_name = 'accuracy'
    metric_func = bench.accuracy_score
    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))

    # Covtype has one class more than there is in train
    if params.dataset_name == 'covtype':
        params.n_classes += 1

    if params.n_classes > 2:
        xgb_params['num_class'] = params.n_classes

# it = IterForDMatrixDemo(X_train, y_train)
# m_with_it = xgb.QuantileDMatrix(it)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
n_samples = int(min(1e4, X_test.shape[0]))

def fit(dmatrix):
    if dmatrix is None:
        dmatrix = xgb.DMatrix(X_train, y_train)
    return xgb.train(xgb_params, dmatrix, params.n_estimators)


if params.inplace_predict:
    def predict(*args):
        return booster.inplace_predict(np.ascontiguousarray(X_test.values,
                                                            dtype=np.float32))
else:
    def predict(dmatrix):  # type: ignore
        if dmatrix is None:
            dmatrix = xgb.DMatrix(X_test, y_test)
        return booster.predict(dmatrix)


if params.inplace_predict:
    def predict2(*args):
        return booster.inplace_predict(np.ascontiguousarray(X_test.iloc[0:n_samples].values,
                                                            dtype=np.float32))
else:
    def predict2(dmatrix):  # type: ignore
        if dmatrix is None:
            dmatrix = xgb.DMatrix(X_test, y_test)
        return booster.predict(dmatrix)

params.box_filter_measurements = 1


fit_time, booster = bench.measure_function_time(
    # fit, None if params.count_dmatrix else m_with_it, params=params)
    fit, None if params.count_dmatrix else dtrain, params=params)

train_metric = metric_func(
    convert_xgb_predictions(
        # booster.predict(m_with_it),
        booster.predict(dtrain),
        params.objective),
    y_train)

predict_time, y_pred = bench.measure_function_time(
    predict, None if params.inplace_predict or params.count_dmatrix else dtest, params=params)

test_metric = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)

dtest = xgb.DMatrix(X_test.iloc[0:n_samples], y_test[0:n_samples])
predict_batch_time, y_pred_batch = bench.measure_function_time(
    predict2, None if params.inplace_predict or params.count_dmatrix else dtest, params=params)

test_metric_batch = metric_func(convert_xgb_predictions(y_pred_batch, params.objective), y_test[0:n_samples])

predict_time_1b1 = 0
y_pred_1b1 = np.zeros(n_samples, dtype=y_pred_batch.dtype)
# Single line inference
booster.set_param('nthread', 1)
for line in range(n_samples):
    if params.inplace_predict:
        def single_line_inplace_predict():
            return booster.inplace_predict(np.ascontiguousarray(X_test.iloc[line:line+1].values,
                                                                dtype=np.float32))
        predict_time_single, y_pred_single = bench.measure_function_time(single_line_inplace_predict, params=params)

    else:
        dtest_single = xgb.DMatrix(X_test.iloc[line:line+1], label=[y_test[line]])
        predict_time_single, y_pred_single = bench.measure_function_time(
            predict2, None if params.inplace_predict or params.count_dmatrix else dtest_single, params=params)

    predict_time_1b1 += predict_time_single
    # print(y_pred_single)
    # print(y_pred.shape)
    # print(y_test.shape)
    # print(convert_xgb_predictions(y_pred_single, params.objective))
    y_pred_1b1[line:line+1] = convert_xgb_predictions(y_pred_single, params.objective)

test_1b1_metric = metric_func(y_pred_1b1, y_test[0:n_samples])
assert(test_1b1_metric == test_metric_batch)

bench.print_output(library='xgboost', algorithm=f'gradient_boosted_trees_{task}',
                stages=['training', 'prediction', 'batch_prediction', 'single line prediction'],
                params=params, functions=['gbt.fit', 'gbt.predict'],
                times=[fit_time, predict_time, predict_batch_time, predict_time_1b1], metric_type=metric_name,
                metrics=[train_metric, test_metric, test_metric_batch, test_1b1_metric], data=[X_train, X_test, X_test.iloc[0:n_samples], X_test.iloc[0:n_samples]],
                alg_instance=booster, alg_params=xgb_params)

# for i in range(1):
#     # CPU
#     dtest = xgb.DMatrix(X_test, y_test)
#     booster.set_param({"device": "cpu"})
#     booster.set_param({"verbosity": "0"})
#     predict_time_cpu, y_pred = bench.measure_function_time(
#         predict, None if params.inplace_predict or params.count_dmatrix else dtest, params=params)
#     test_metric_cpu = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)

#     # SYCL CPU
#     dtest = xgb.DMatrix(X_test, y_test)
#     booster.set_param({"device": "sycl:cpu"})
#     predict_time_sycl_cpu, y_pred = bench.measure_function_time(
#         predict, None if params.inplace_predict or params.count_dmatrix else dtest, params=params)
#     test_metric_sycl_cpu = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)

#     # SYCL GPU
#     dtest = xgb.DMatrix(X_test, y_test)
#     booster.set_param({"device": "sycl:gpu"})
#     predict_time_sycl_gpu, y_pred = bench.measure_function_time(
#         predict, None if params.inplace_predict or params.count_dmatrix else dtest, params=params)
#     test_metric_sycl_gpu = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)


#     bench.print_output(library='xgboost', algorithm=f'gradient_boosted_trees_{task}',
#                     stages=['training', 'prediction_cpu', 'prediction_sycl_cpu', 'prediction_sycl_gpu'],
#                     params=params,
#                     functions=['gbt.fit', 'gbt.predict'],
#                     times=[fit_time, predict_time_cpu, predict_time_sycl_cpu, predict_time_sycl_gpu],
#                     metric_type=metric_name,
#                     metrics=[train_metric, test_metric_cpu, test_metric_sycl_cpu, test_metric_sycl_gpu],
#                     data=[X_train, X_test, X_test, X_test],
#                     alg_instance=booster, alg_params=xgb_params)
