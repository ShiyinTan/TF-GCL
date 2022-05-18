import os
import copy

opt = dict()
opt['data'] = 'ReTweet'
opt['h_feats_list'] = 256
opt['epochs'] = 2000
opt['emb_size'] = 256
opt['lr'] = 5e-3
# opt['input_dim'] = 1433
opt['cl_loss_weight'] = 1.0
opt['cl_batch_size'] = 256
opt['neg_class_weight'] = 0.4
opt['temporal_weight_loss'] = 'true'
opt['cl_loss'] = 'true'
# opt['add_edge'] = 0.15
opt['decay_by'] = 'year'
opt['ts_decay_rate'] = 0.15
opt['concat'] = 'false'
opt['temporal_cl_loss'] = 'true'
opt['save_model'] = 'false'
opt['save_model_file_name'] = 'tf-gcl'
opt['neighbor_as_pos'] = 'true'
opt['symmetric_cl_loss'] = 'true'
opt['fea_drop_percent'] = 0.6
opt['edge_pert_percent'] = 0.6
opt['ts_mask_percent'] = 0.6
opt['aug_1_fea_mask'] = 'true'
opt['aug_1_time_mask'] = 'false'
opt['aug_1_edge_pert'] = 'true'
opt['aug_2_fea_mask'] = 'true'
opt['aug_2_time_mask'] = 'false'
opt['aug_2_edge_pert'] = 'true'
opt['eval_epochs'] = 1
opt['loss_print_mode'] = 'false'
opt['verbose'] = 'true'
opt['parameter_sens_mode'] = 'false'

opt['robust_test'] = 'false'
opt['robust_ratio'] = 0.0


# best is False
whole_net = True

def command(opt):
    script = 'python main_dynamicCL_ver1.py'
    for opt, val in opt.items():
        script += ' --' + opt + ' ' + str(val)
    return script

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(command(opt_))

if __name__ == '__main__':
    run(opt)