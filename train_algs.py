import copy
import logging
import time

import numpy as np
import pandas

import utils
from cayley_adam import stiefel_optimizer
from definitions import *


def train(model, param_stiefel, maxiterations, maxinnertime=np.inf, lr=0.1,
          Optimizer1=stiefel_optimizer.AdamG, eval_f=None, save_best=False, es_patience=10) -> pandas.DataFrame:
    param = param_stiefel[0]

    dict_stiefel = {'params': param_stiefel, 'lr': lr, 'stiefel': True}

    optimizer1 = Optimizer1([dict_stiefel])

    t = 1
    es_count = 0
    terminating_condition = True
    elapsed_minutes = 0
    start_timestamp = time.time()
    loss, orto, zerosum = model.loss()
    best_valmetric, eval = eval_f()
    best_params = copy.deepcopy(model.state_dict())
    log_dict = {'outer_i': 0, 'inner_i': 0, 'i': 'init',
                'j_tot': float(loss.detach().cpu()),
                # 'grad': float(grad_q),
                'orto': float(orto.detach().cpu()),
                'zerosum': float(zerosum.detach().cpu()),
                'X': None if t % 100 != 0 and t != 1 else [torch.clone(param).detach().cpu().numpy()],
                'train_acc': eval['train_acc'],
                'val_acc': eval['val_acc'],
                'cos_sim': eval['cos_sim'],
                'comb_metric': eval['comb_metric'],
                'test_acc': eval['test_acc']}
    val_table = pandas.DataFrame(eval, index=[0])
    train_table = pandas.DataFrame(log_dict, index=[0])
    logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).
                 to_string(header=(t == 1), index=False, justify='right', col_space=15,
                           float_format=utils.float_format,
                           formatters={'mu': lambda x: "%.2f" % x, 'train_acc': lambda x: "%.2f" % x,
                                       'val_acc': lambda x: "%.2f" % x, 'test_acc': lambda x: "%.2f" % x,
                                       'cos_sim': lambda x: "%.2f" % x, 'comb_metric': lambda x: "%.2f" % x,
                                       'val_metric': lambda x: "%.2f" % x},
                           columns=train_table.columns.drop(['X', 'outer_i', 'inner_i'])))

    while terminating_condition and t < maxiterations and elapsed_minutes < maxinnertime and es_count < es_patience:  # inner loop
        loss, orto, zerosum = model.loss()
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step(lambda: model.loss()[0])
        with torch.no_grad():
            model.semisup_layer.solve_and_update(model.gckm_layers[-1].h_st.t())

        valmetric, eval = eval_f() if eval_f is not None else -1
        if save_best and valmetric > best_valmetric:
            best_valmetric, best_params = valmetric, copy.deepcopy(model.state_dict())
            es_count = 0
        else:
            es_count += 1
        log_dict = {'outer_i': 1, 'inner_i': t, 'i': t,
                    'j_tot': float(loss.detach().cpu()),
                    # 'grad': float(grad_q),
                    'orto': float(orto.detach().cpu()),
                    'zerosum': float(zerosum.detach().cpu()),
                    'X': None if t % 100 != 0 and t != 1 else [torch.clone(param).detach().cpu().numpy()],
                    'train_acc': eval['train_acc'],
                    'val_acc': eval['val_acc'],
                    'cos_sim': eval['cos_sim'],
                    'comb_metric': eval['comb_metric'],
                    'test_acc': eval['test_acc']}
        train_table = pandas.concat([train_table, pandas.DataFrame(log_dict, index=[0])])
        val_table = pandas.concat([val_table, pandas.DataFrame(eval, index=[0])])
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).
                     to_string(header=(t == 1), index=False, justify='right', col_space=15,
                               float_format=utils.float_format,
                               formatters={'mu': lambda x: "%.2f" % x, 'train_acc': lambda x: "%.2f" % x,
                                           'val_acc': lambda x: "%.2f" % x, 'test_acc': lambda x: "%.2f" % x,
                                           'cos_sim': lambda x: "%.2f" % x, 'comb_metric': lambda x: "%.2f" % x,
                                           'val_metric': lambda x: "%.2f" % x},
                               columns=train_table.columns.drop(['X', 'outer_i', 'inner_i'])))
        t += 1
        elapsed_minutes = (time.time() - start_timestamp) / 60

    # Add last X to train_table
    train_table.at[t - 1, 'X'] = [torch.clone(param).detach().cpu().numpy()]

    if save_best:
        model.load_state_dict(best_params)

    return train_table, val_table
