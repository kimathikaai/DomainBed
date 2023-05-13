# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np
from domainbed import datasets

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed') # organize records by hparams
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
            # organize hparam_seed results from highest to lowest based on
            # val_acc calculated from self.run_acc
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            # get first [0] result and get the self.run_acc dict [0]
            # and return the 'test_acc' value
            print(
                "%",
                _hparams_accs[0][1][0]['args']['algorithm'],
                _hparams_accs[0][1][0]['args']['dataset'],
                _hparams_accs[0][1][0]['args']['overlap'],
                _hparams_accs[0][1][0]['args']['test_envs'],
                _hparams_accs[0][1][0]['args']['hparams_seed'],
                _hparams_accs[0][1][0]['args']['output_dir'],
                _hparams_accs[0][1][0]['args'],
                _hparams_accs[0][0]
                  )
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"
    selec_metric = None
    eval_metric = None

    @classmethod
    def run_acc(cls, run_records):
        assert cls.selec_metric is not None
        assert cls.eval_metric is not None
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_{}'.format(test_env, cls.eval_metric)
        test_in_acc_key = 'env{}_in_{}'.format(test_env, cls.eval_metric)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"
    selec_metric = None
    eval_metric = None

    # @classmethod
    # def _step_acc(cls, record):
    #     """Given a single record, return a {val_acc, test_acc} dict."""
    #     test_env = record['args']['test_envs'][0]
    #     val_env_keys = []
    #     for i in itertools.count():
    #         # because you don't know the number of envs ahead of time
    #         if f'env{i}_out_{cls.selec_metric}' not in record:
    #             break
    #         if i != test_env:
    #             val_env_keys.append(f'env{i}_out_{cls.selec_metric}')
    #     test_in_acc_key = 'env{}_in_{}'.format(test_env, cls.eval_metric)
    #     return {
    #         'val_acc': np.mean([record[key] for key in val_env_keys]), # average of 20% split of train envs
    #         'test_acc': record[test_in_acc_key] # 80% split of the test env
    #     } 

    @classmethod
    def _step_acc(cls, record):
        """
        Given a single record, return a {val_acc, test_acc} dict.
            macc = weighted average oacc and nacc to get macro-acc
            vacc = non-weighted average between oacc and nacc
        """
        test_env = record['args']['test_envs'][0]
        dataset = vars(datasets)[record['args']['dataset']]
        overlap = str(record['args']['overlap'])

        val_env_keys = []
        test_env_keys = []

        oacc_weight = dataset.N_OC[overlap]["N_oc"]/dataset.NUM_CLASSES
        nacc_weight = 1 - oacc_weight

        # print(dataset, overlap, cls.selec_metric, cls.eval_metric)

        if cls.eval_metric == "macc":
            test_env_keys.append((f'env{test_env}_in_nacc', 2*nacc_weight))
            test_env_keys.append((f'env{test_env}_in_oacc', 2*oacc_weight))
            # print("eval",oacc_weight, nacc_weight)
        elif cls.eval_metric == "vacc":
            test_env_keys.append((f'env{test_env}_in_nacc', 1))
            test_env_keys.append((f'env{test_env}_in_oacc', 1))
            # print("eval",1,1)
        else:
            test_env_keys.append((f'env{test_env}_in_{cls.eval_metric}', 1))

        for i in itertools.count():
            # because you don't know the number of envs ahead of time
            if f'env{i}_out_nacc' not in record:
                break
            if i != test_env:
                if cls.selec_metric == "macc":
                    val_env_keys.append((f'env{i}_out_nacc', 2*nacc_weight))
                    val_env_keys.append((f'env{i}_out_oacc', 2*oacc_weight))
                    # print("val",oacc_weight, nacc_weight)
                elif cls.selec_metric == "vacc":
                    val_env_keys.append((f'env{i}_out_nacc', 1))
                    val_env_keys.append((f'env{i}_out_oacc', 1))
                    # print("val",1, 1)
                else:
                    val_env_keys.append((f'env{i}_out_{cls.selec_metric}', 1))

        # test_in_acc_key = 'env{}_in_{}'.format(test_env, cls.eval_metric)
        results_dict = {
            'val_acc': np.mean([record[key]*weight for key, weight in val_env_keys]), # average of 20% split of train envs
            'test_acc': np.mean([record[key]*weight for key, weight in test_env_keys]), # average of 80% split of train envs
            # 'test_acc': record[test_in_acc_key] # 80% split of the test env
        }
        return results_dict

    @classmethod
    def run_acc(cls, run_records):
        assert cls.selec_metric is not None
        assert cls.eval_metric is not None
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(cls._step_acc).argmax('val_acc') 
        # return the record maximizing val_acc to represent the hparam group

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"
    selec_metric = None
    eval_metric = None

    @classmethod
    def _step_acc(cls, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        # grab all val_accs exluding test_env (around it)
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(cls, records):
        assert cls.selec_metric is not None
        assert cls.eval_metric is not None
        step_accs = records.group('step').map(lambda step, step_records:
            cls._step_acc(step_records)
        ).filter_not_none()
        # records.group('step') creates a seperate record for each step
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None
