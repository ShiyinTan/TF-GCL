from tabnanny import verbose
import torch
# from dataload.mathoverflow_data import MathoverflowDataset
# from dataload.retweet_data import ReTweetDataset
# from dataload.dblp_data import DBLPDataset
import dataload
import utils.utils as utils
from net.model import MyFeatureExtractModel, DotPredictor
import itertools
import time
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import random
from dataload.walks2adj import CustomEdgeDataset
import warnings

warnings.filterwarnings('ignore')

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--data', type=str, default='Mathoverflow')
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--cl_batch_size', type=int, default=64)
parser.add_argument('--h_feats_list', nargs='+', help='list of hidden layers', required=True)
# parser.add_argument('--h_feats_1', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--fea_drop_percent', type=float, default=0.1)
parser.add_argument('--edge_pert_percent', type=float, default=0.1)
parser.add_argument('--ts_mask_percent', type=float, default=0.2)
parser.add_argument('--cl_loss_weight', type=float, default=0.1)
parser.add_argument('--use_max_timestamp_mask', type=str_to_bool, default='true')
parser.add_argument('--timestamp_mask', type=int, default=0)
parser.add_argument('--ts_decay_rate', type=float, default=0.02)
parser.add_argument('--pos_class_weight', type=float, default=1.0)
parser.add_argument('--neg_class_weight', type=float, default=0.6)
parser.add_argument('--eval_epochs', type=int, default=5)
parser.add_argument('--temporal_weight_loss', type=str_to_bool, default='true')
parser.add_argument('--cl_loss', type=str_to_bool, default='true')
parser.add_argument('--decay_by', type=str, default='month')
parser.add_argument('--concat', type=str_to_bool, default='true')
parser.add_argument('--train_embedding', type=str_to_bool, default='false')
parser.add_argument('--pretrain_emb', type=str_to_bool, default='false')
parser.add_argument('--temporal_cl_loss', type=str_to_bool, default='false')
parser.add_argument('--save_model', type=str_to_bool, default='false')
parser.add_argument('--save_model_file_name', type=str, default="gcl")
parser.add_argument('--neighbor_as_pos', type=str_to_bool, default='true')
parser.add_argument('--symmetric_cl_loss', type=str_to_bool, default='false')
parser.add_argument('--aug_1_fea_mask', type=str_to_bool, default='true')
parser.add_argument('--aug_2_fea_mask', type=str_to_bool, default='true')
parser.add_argument('--aug_1_time_mask', type=str_to_bool, default='true')
parser.add_argument('--aug_2_time_mask', type=str_to_bool, default='false')
parser.add_argument('--aug_1_edge_pert', type=str_to_bool, default='false')
parser.add_argument('--aug_2_edge_pert', type=str_to_bool, default='true')
parser.add_argument('--loss_print_mode', type=str_to_bool, default='false')
parser.add_argument('--verbose', type=str_to_bool, default='true')
parser.add_argument('--parameter_sens_mode', type=str_to_bool, default='false')
parser.add_argument('--robust_test', type=str_to_bool, default='false')
parser.add_argument('--robust_ratio', type=float, default=0.0)
parser.add_argument('--GCN_model', type=str_to_bool, default='false')

args = parser.parse_args()
# torch.set_num_threads(4)

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ----------- 1. load dataset -------------- #
    assert args.data in ["Mathoverflow", "ReTweet", "DBLP", "Email", "Facebook", "CollegeMsg"], 'Dataset is not included!'
    if args.data == "Mathoverflow":
        dataset = dataload.MathoverflowDataset(verbose=args.verbose, robust_test=args.robust_test, robust_ratio=args.robust_ratio)
        directory = "./data/sx-mathoverflow/"
    elif args.data == "ReTweet":
        dataset = dataload.ReTweetDataset(verbose=args.verbose, robust_test=args.robust_test, robust_ratio=args.robust_ratio)
        directory = "./data/ia-retweet-pol/"
    elif args.data == "DBLP":
        dataset = dataload.DBLPDataset(verbose=args.verbose, robust_test=args.robust_test, robust_ratio=args.robust_ratio)
        directory = "./data/DBLP/"
    elif args.data == "Email":
        dataset = dataload.EmailDataset(verbose=args.verbose, robust_test=args.robust_test, robust_ratio=args.robust_ratio)
        directory = "./data/email-Eu-core/"
    elif args.data == "Facebook":
        dataset = dataload.FacebookDataset(slice_interval=35,verbose=args.verbose, robust_test=args.robust_test, robust_ratio=args.robust_ratio)
        directory = "./data/facebook/"
    elif args.data == "CollegeMsg":
        dataset = dataload.CollegeMsgDataset(slice_interval=35,verbose=args.verbose, robust_test=args.robust_test, robust_ratio=args.robust_ratio)
        directory = "./data/CollegeMsg/"
    else:
        print("no dataset find!")

    if args.loss_print_mode:
        loss_f = open(directory+"losses.txt",'w')
        cl_loss_f = open(directory+"cl_losses.txt",'w')
        auc_f = open(directory+"aucs.txt",'w')
        ap_f = open(directory+"aps.txt",'w')

    if args.parameter_sens_mode:
        param_sens_f = open(directory+"param_sens.txt",'a+')

    save_model_file_name = args.save_model_file_name.strip("'")

    # ----------- 2. set up model -------------- #
    # embedding 结论：不参与训练效果更好
    embedding = torch.nn.Embedding(dataset.num_nodes, args.emb_size)

    dataset.to(device)
    # train_neg_graph = train_neg_graph.to(device)
    # test_neg_graph = test_neg_graph.to(device)

    batch_size = args.cl_batch_size

    embedding.to(device)
    pos_timestamps = dataset.train_graph.edata['timestamp']
    neg_timestamps = dataset.train_neg_graph.edata['timestamp']

    if args.use_max_timestamp_mask:
        timestamp_mask = pos_timestamps.max()
    else:
        timestamp_mask = args.timestamp_mask
    
    # aug_graph1 = aug_graph1.to(device)
    # aug_graph2 = aug_graph2.to(device)

    # init embedding with node2vec embedding, 结论：会降低sx-mathoverflow的效果
    if args.pretrain_emb:
        node2vec_emb = pd.read_csv(directory + "node2vec.emb",  index_col=0, sep=' ', skiprows=1, header=None)
        node2vec_emb.sort_index(inplace=True)
        node2vec_emb = torch.from_numpy(node2vec_emb.to_numpy(dtype=np.float32)).to(device)
        embedding.weight = torch.nn.parameter.Parameter(node2vec_emb)

    h_feats_list = list(map(int, args.h_feats_list))
    # define model
    model = MyFeatureExtractModel(dataset.num_nodes, embedding.embedding_dim, h_feats_list=h_feats_list, concat=args.concat).to(device)
    # model = GraphSAGE(embedding(dataset.train_graph.nodes()).shape[1], 64).to(device)

    # You can replace DotPredictor with MLPPredictor.
    # pred = MLPPredictor(16)
    pred = DotPredictor().to(device)

    # ----------- 3. set up optimizer -------------- #
    # in this case, loss will in training loop
    if args.train_embedding:
        optimizer = torch.optim.Adam(itertools.chain(embedding.parameters(), model.parameters(), pred.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

    max_auc = 0.0
    max_ap = 0.0
    # ----------- 4. training -------------------------------- #
    if args.save_model:
        torch.save(embedding.state_dict(), directory+"{}_embedding_weights.pth".format(save_model_file_name))
    all_logits = []
    for e in range(args.epochs):
        # forward
    #     h = model(dataset.train_graph, embedding(dataset.train_graph.nodes()))
    #     h = model(dataset)
    #     h = model(aug_graph1, embedding(dataset.train_graph.nodes()))
        t_st = time.time()
        if args.GCN_model:
            h = model(dataset.train_graph, embedding(dataset.train_graph.nodes()))
            h1_norm = F.normalize(h)
            h2_norm = F.normalize(h)
        else:
            if args.aug_1_fea_mask:
                aug_embedding_1 = utils.random_feature_mask(embedding, drop_percent=args.fea_drop_percent, device=device)
            else:
                aug_embedding_1 = embedding
            if args.aug_2_fea_mask:
                aug_embedding_2 = utils.random_feature_mask(embedding, drop_percent=args.fea_drop_percent, device=device)
            else:
                aug_embedding_2 = embedding
            
            if args.aug_1_time_mask:
                aug_graph1 = utils.random_masking_timestamp(dataset, mask_num = int(dataset.train_graph.number_of_edges()*args.ts_mask_percent), 
                                                    timestamp_mask=timestamp_mask, device=device)
            else:
                aug_graph1 = dataset.train_graph
            if args.aug_2_time_mask:
                aug_graph2 = utils.random_masking_timestamp(dataset, mask_num = int(dataset.train_graph.number_of_edges()*args.ts_mask_percent), 
                                                    timestamp_mask=timestamp_mask, device=device)
            else:
                aug_graph2 = dataset.train_graph
            
            if args.aug_1_edge_pert:
                aug_graph1 = utils.random_remove_and_add_edges(dataset, replace_num=int(dataset.train_graph.number_of_edges()*args.edge_pert_percent), 
                                                        timestamp_mask=timestamp_mask, device=device)
            else:
                aug_graph1 = dataset.train_graph
            if args.aug_2_edge_pert:
                aug_graph2 = utils.random_remove_and_add_edges(dataset, replace_num=int(dataset.train_graph.number_of_edges()*args.edge_pert_percent), 
                                                        timestamp_mask=timestamp_mask, device=device)
            else:
                aug_graph2 = dataset.train_graph
            
            h1 = model(aug_graph1, aug_embedding_1(dataset.train_graph.nodes()))
            h2 = model(aug_graph2, aug_embedding_2(dataset.train_graph.nodes()))
            h = h1+h2
            if torch.any(torch.isnan(h1)) or torch.any(torch.isnan(h2)):
                print("h1 nan", "h2 nan")
            h1_norm = F.normalize(h1)
            h2_norm = F.normalize(h2)

        # my alg
        # aug_embedding = utils.random_feature_mask(embedding, drop_percent=args.fea_drop_percent, device=device)
        # aug_graph1 = utils.random_remove_and_add_edges(dataset, replace_num=int(dataset.train_graph.number_of_edges()*args.edge_pert_percent), 
        #                                         timestamp_mask=timestamp_mask, device=device)
        # aug_graph2 = utils.random_masking_timestamp(dataset, mask_num = int(dataset.train_graph.number_of_edges()*args.ts_mask_percent), 
        #                                     timestamp_mask=timestamp_mask, device=device)
        # h1 = model(dataset.train_graph, aug_embedding(dataset.train_graph.nodes()))
        # h2 = model(aug_graph1, aug_embedding(dataset.train_graph.nodes()))
        # h = h1+h2
        # print("h1 nan: ", torch.any(torch.isnan(h1)), "h2 nan: ", torch.any(torch.isnan(h2)))
        # h1_norm = F.normalize(h1)
        # h2_norm = F.normalize(h2)
        pos_score = pred(dataset.train_graph, h).to(device)
        neg_score = pred(dataset.train_neg_graph, h).to(device)
    #     loss = compute_loss(pos_score, neg_score, device)
        
    #     for i in range(dataset.num_nodes//batch_size+1):
    #         t_cl_loss_compute_st = time.time()
    #         batch_nodes = dataset.train_graph.nodes()[i*batch_size:(i+1)*batch_size].tolist()
    #         cl_loss = batch_CL_loss(batch_nodes, dataset.train_graph, device=device)
    #         t_cl_loss_compute_ed = time.time()
        if args.temporal_weight_loss:
            loss = utils.compute_loss_with_weight_temporal_decay(pos_score, neg_score, pos_timestamps, 
                                    neg_timestamps, pos_timestamps.max(), decay_rate=args.ts_decay_rate, decay_by=args.decay_by,
                                    class_weight=[args.pos_class_weight,args.neg_class_weight], device=device)
        else:
            loss = utils.compute_loss(pos_score, neg_score, device)
        
        if args.loss_print_mode:
            loss_f.write(str(loss.detach().cpu().numpy()))
            loss_f.write("\n")

        if args.cl_loss and (args.GCN_model==False):
            if dataset.num_nodes%batch_size == 0:
                i = e%(dataset.num_nodes//batch_size)
            else:
                i = e%(dataset.num_nodes//batch_size+1)
            batch_nodes = dataset.train_graph.nodes()[i*batch_size:(i+1)*batch_size].tolist()
            cl_loss = utils.batch_CL_loss(batch_nodes, dataset.train_graph, h1_norm, h2_norm, 
                                          temporal_loss=args.temporal_cl_loss,neig_positive=args.neighbor_as_pos,
                                          sym_loss=args.symmetric_cl_loss, device=device)
            loss += args.cl_loss_weight*cl_loss
            if args.loss_print_mode:
                cl_loss_f.write(str(cl_loss.detach().cpu().numpy()))
                cl_loss_f.write("\n")

        # loss = args.cl_loss_weight*cl_loss +\
        #         utils.compute_loss_with_weight_temporal_decay(pos_score, neg_score, pos_timestamps, 
        #                             neg_timestamps, pos_timestamps.max(), decay_rate = args.ts_decay_rate, 
        #                             class_weight=[args.pos_class_weight,args.neg_class_weight] ,device=device)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t_ed = time.time()
        
        if e % args.eval_epochs == 0:
            with torch.no_grad():
                pos_score = pred(dataset.test_graph, h)
                neg_score = pred(dataset.test_neg_graph, h)
                auc = utils.compute_auc(pos_score, neg_score)
                ap_score = utils.compute_apscore(pos_score, neg_score)
                if args.loss_print_mode:
                    auc_f.write(str(auc))
                    auc_f.write("\n")
                    ap_f.write(str(ap_score))
                    ap_f.write("\n")
                if auc > max_auc: 
                    max_auc = auc
                    if args.save_model and e > 200:
                        torch.save(model.state_dict(), directory+'{}_model_weights.pth'.format(save_model_file_name))
                if ap_score > max_ap: max_ap = ap_score
            if args.verbose:
                if args.cl_loss and (args.GCN_model==False):
                    print('In epoch {}, loss: {:.6f}, cl_loss: {:.6f}, auc: {:.6f}, ap: {:.6f}, epoch_time: {:.2f}'\
                        .format(e, loss, cl_loss, auc, ap_score, t_ed-t_st))
                else:
                    print('In epoch {}, loss: {:.6f}, auc: {:.6f}, ap: {:.6f}, epoch_time: {:.2f}'\
                        .format(e, loss, auc, ap_score, t_ed-t_st))
            

    # ----------- 5. check results ------------------------ #
    
    with torch.no_grad():
        pos_score = pred(dataset.test_graph, h)
        neg_score = pred(dataset.test_neg_graph, h)
        auc = utils.compute_auc(pos_score, neg_score)
        ap_score = utils.compute_apscore(pos_score, neg_score)
        if auc > max_auc: max_auc = auc
        if ap_score > max_ap: max_ap = ap_score
        print('AUC: ', auc, ", AP: ", ap_score)
        print("MAX_AUC: ", max_auc, "MAX_AP: ", max_ap)
    
    if args.verbose:
        if args.aug_1_fea_mask:
            print("aug_1: feature mask")
        if args.aug_1_time_mask:
            print("aug_1: time mask")
        if args.aug_1_edge_pert:
            print("aug_1: edge pert")
        if args.aug_2_fea_mask:
            print("aug_2: feature mask")
        if args.aug_2_time_mask:
            print("aug_2: time mask")
        if args.aug_2_edge_pert:
            print("aug_2: edge pert")
    
    
    
    if args.loss_print_mode:
        loss_f.close()
        cl_loss_f.close()
        auc_f.close()
        ap_f.close()
    if args.parameter_sens_mode:
        param_sens_f.write("feat_mask:{}, edge_pert:{}, ts_mask:{}, alpha:{}, MaxAUC:{}, MaxAP:{}\n"\
            .format(args.fea_drop_percent, args.edge_pert_percent, args.ts_mask_percent, args.cl_loss_weight, max_auc, max_ap))
        param_sens_f.close()