import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
from dataset_loader import DataLoader
import random
from datetime import datetime
from model_GCKM import DeepGCKM
from utils import fixed_splits
from train_algs import *
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
import torch_geometric.utils


@hydra.main(version_base=None, config_path="../GCKM/conf", config_name="default_config")
def run_experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.maxiterations == -1: cfg.maxiterations = np.inf
    cfg.dataset = cfg.dataset.lower()
    cfg.SemiSup_layer.kernel2.dataset = cfg.dataset
    # ==================================================================================================================
    # Set random seed
    seed = 1941488137  # The seed does not influence results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ==================================================================================================================
    # Load Training Data

    data, num_classes, _ = DataLoader(cfg.dataset, cfg.preprocess)

    if cfg.datasplit == 'val_subsets':
        assert cfg.dataset.lower() in ['cora', 'citeseer',
                                       'pubmed'], f"{cfg.datasplit.lower()} only available for 'cora', 'citeseer', 'pubmed'"
        num_vals = ['1', '5', '10', 'full']
        data.train_mask = torch.load('masks/' + cfg.dataset + '_train_mask.pt')
        data.test_mask = torch.load('masks/' + cfg.dataset + '_test_mask.pt')
        data.val_mask = dict()
        [data.val_mask.update({num_val: torch.load('masks/' + cfg.dataset + '_' + num_val + '_val_mask.pt')}) for
         num_val in num_vals]

    elif cfg.datasplit == ('semi_fix_fewlabels'):
        if cfg.dataset == 'ogbn-arxiv':
            data = fixed_splits(data, 40, 106, 4234, 'ogbn-arxiv')
        else:
            data.train_mask = torch.load('masks/' + cfg.dataset.lower() + '_fewlabels_full_train_mask.pt')
            data.val_mask = data.train_mask  # set to training mask to avoid errors
            data.test_mask = torch.load('masks/' + cfg.dataset.lower() + '_fewlabels_test_mask.pt')

    elif cfg.datasplit.lower() == 'standard':
        if cfg.dataset.lower() in ["chameleon", "squirrel"]:
            train_rate = 0.025
            val_rate = 0.025
            percls_trn = int(round(train_rate * len(data.y) / num_classes))
            val_lb = int(round(val_rate * len(data.y)))
            data = fixed_splits(data, num_classes, percls_trn, val_lb, cfg.dataset)
        elif cfg.dataset.lower() in ["cora", "citeseer", "pubmed", "ogbn-arxiv"]:
            pass

    edge_index, x = data.edge_index.to(device), data.x.to(device)
    edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]
    if cfg.to_undirected:
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    p = data.y.max() + 1
    data.y_coded = 2 * F.one_hot(data.y) - 1
    data.codes = -torch.ones(p, p) + 2 * torch.eye(p)
    y_coded, y, codes = data.y_coded.to(device), data.y.to(device), data.codes.to(device)
    if data.train_mask.dim() == 2:
        data.train_mask, data.val_mask, data.test_mask = data.train_mask[:, cfg.split_id].bool(), data.val_mask[:,
                                                                                                  cfg.split_id].bool(), data.test_mask[
                                                                                                                        :,
                                                                                                                        cfg.split_id].bool()
    trainmask, valmask, testmask = data.train_mask, data.val_mask, data.test_mask

    num_nodes = data.num_nodes
    del data

    # ==================================================================================================================
    # Define supervision set
    c = y_coded.cpu().type(TensorType).squeeze()
    for i in range(num_nodes):
        if not trainmask[i]:
            c[i, :] = torch.zeros(num_classes)  # mask validation and test labels

    l = trainmask.type(TensorType)
    l, c = l.to(device), c.to(device)

    start = datetime.now()
    dGCKM = DeepGCKM(cfg, edge_index, x, l, c, codes, trainmask, valmask, testmask, y).to(device)
    initialization_time = datetime.now() - start
    time.sleep(1)
    logging.info("\nInitialization complete in: " + str(initialization_time))

    logging.info(dGCKM.init_performance)

    test_init = dGCKM.init_performance['test_acc']

    logging.info(dGCKM)
    logging.info(cfg)
    # ==================================================================================================================
    # Divide differentiable parameters in 2 groups: 1. Manifold parameters 2. Other parameters

    param_h_st = []
    for i in range(len(dGCKM.gckm_layers)):
        param_h_st += [param[1] for param in dGCKM.gckm_layers[i].named_parameters() if "h_st" in param[0]]

    for param in param_h_st:
        param.requires_grad = True

    # ==================================================================================================================
    # Define evaluation function
    def eval():
        valdict = dGCKM.acc_dict(y, trainmask=trainmask, valmask=valmask, testmask=testmask)
        return valdict[cfg.valmetric], valdict

    # Train - finetuning =========================================================================================================
    start = datetime.now()
    optimizer1 = stiefel_optimizer.AdamG

    train_table, val_table = train(dGCKM, param_h_st, cfg.maxiterations + 1, lr=cfg.lr,
                                   Optimizer1=optimizer1, eval_f=eval, save_best=cfg.save_best,
                                   es_patience=cfg.early_stopping_patience)

    finetuning_time = datetime.now() - start
    time.sleep(1)
    logging.info("\nFinetuning complete in: " + str(finetuning_time))

    ## Summarise Results ======================================================================================
    if dGCKM.semisup_layer.multiview:
        model = 'GCKM-MV'
    else:
        model = 'GCKM'
    if cfg.datasplit == 'val_subsets':
        logging.info('Validation performances')
        logging.info(val_table)
        logging.info(f'\nSummary\nDataset: {cfg.dataset}\nDatasplit: {cfg.datasplit}\nModel: {model}')
        eval_dict = {}
        test_max_cos = val_table['test_acc'].iloc[val_table['cos_sim'].argmax()]
        eval_dict.update({'test_0': test_max_cos})

        for numval in valmask:
            test_max_val = val_table['test_acc'].iloc[val_table['val_acc_' + numval].argmax()]
            eval_dict.update({'test_' + numval: test_max_val})
        logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    else:
        logging.info(f'\nSummary\nDataset: {cfg.dataset}\nDatasplit: {cfg.datasplit}\nModel: {model}')
        dGCKM.eval()
        eval_dict = dGCKM.acc_dict(y, trainmask, valmask, testmask)
        eval_dict.update({'test_init': test_init})
        logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))


if __name__ == "__main__":
    run_experiment()
