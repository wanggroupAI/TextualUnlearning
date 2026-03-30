import logging
from transformers import logging as transformers_logging

logging.basicConfig(level=logging.ERROR)


transformers_logging.set_verbosity_error()
import sys, argparse
import os
import datetime
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from init_ul import get_init_gpt_cls

from utilities_ul import get_closest_tokens_gpt, get_reconstruction_loss, fix_special_tokens, remove_padding,unlearn,TextDataset_cls,fineTune_cls
from copy import deepcopy
from args_factory import get_args

import time
import ssl
ssl._create_default_https_context=ssl._create_unverified_context

from scipy.optimize import linear_sum_assignment
import os 
import warnings
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
warnings.filterwarnings("ignore")
np.random.seed(42)
args = get_args()



def get_loss(args, lm, model, ids, x_embeds, true_labels, true_grads, create_graph=False):

    perplexity = lm(input_ids=ids, labels=ids).loss
    rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=create_graph)
    return perplexity, rec_loss, rec_loss + args.coeff_perplexity * perplexity

def swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads):
    print("Attempt swap", flush=True)
    best_x_embeds, best_tot_loss = None, None
    changed = None
    for sen_id in range(x_embeds.data.shape[0]):
        for sample_idx in range(200):
            perm_ids = np.arange(x_embeds.shape[1])

            if sample_idx != 0:
                if sample_idx % 4 == 0:  # swap two tokens
                    i, j = 1 + np.random.randint(
                        max_len[sen_id] - 2
                    ), 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                elif sample_idx % 4 == 1:  # move a token to another place
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    j = 1 + np.random.randint(max_len[sen_id] - 1)
                    if i < j:
                        perm_ids = np.concatenate(
                            [
                                perm_ids[:i],
                                perm_ids[i + 1 : j],
                                perm_ids[i : i + 1],
                                perm_ids[j:],
                            ]
                        )
                    else:
                        perm_ids = np.concatenate(
                            [
                                perm_ids[:j],
                                perm_ids[i : i + 1],
                                perm_ids[j:i],
                                perm_ids[i + 1 :],
                            ]
                        )
                elif sample_idx % 4 == 2:  # move a sequence to another place
                    b = 1 + np.random.randint(max_len[sen_id] - 1)
                    e = 1 + np.random.randint(max_len[sen_id] - 1)
                    if b > e:
                        b, e = e, b
                    p = 1 + np.random.randint(max_len[sen_id] - 1 - (e - b))
                    if p >= b:
                        p += e - b
                    if p < b:
                        perm_ids = np.concatenate(
                            [perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]]
                        )
                    elif p >= e:
                        perm_ids = np.concatenate(
                            [perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]]
                        )
                    else:
                        assert False
                elif sample_idx % 4 == 3:  # take some prefix and put it at the end
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids = np.concatenate(
                        [perm_ids[:1], perm_ids[i:-1], perm_ids[1:i], perm_ids[-1:]]
                    )

            new_ids = cos_ids.clone()
            new_ids[sen_id] = cos_ids[sen_id, perm_ids]
            new_x_embeds = x_embeds.clone()
            new_x_embeds[sen_id] = x_embeds[sen_id, perm_ids, :]

            _, _, new_tot_loss = get_loss(
                args, lm, model, new_ids, new_x_embeds, true_labels, true_grads
            )

            if (best_tot_loss is None) or (new_tot_loss < best_tot_loss):
                best_x_embeds = new_x_embeds
                best_tot_loss = new_tot_loss
                if sample_idx != 0:
                    changed = sample_idx % 4
        if not (changed is None):
            change = [
                "Swapped tokens",
                "Moved token",
                "Moved sequence",
                "Put prefix at the end",
            ][changed]
            print(change, flush=True)
        x_embeds.data = best_x_embeds
        
def reconstruct(args, device, sample, metric, tokenizer, lm, model,init_model,dataset,retain_set):
    sequences, true_labels = sample
    
    lm_tokenizer = tokenizer


    lm_embeddings = model.get_input_embeddings()
    lm_embeddings_weight = lm_embeddings.weight.unsqueeze(0)

    orig_batch = tokenizer(sequences, padding="max_length", truncation=True, max_length=12,return_tensors='pt').to(device)

    true_embeds = lm_embeddings(orig_batch['input_ids'])

    true_embeds = true_embeds.detach()


    ul_model=unlearn(model, true_embeds, true_labels,retain_set,lr=args.train_lr,unlearn_md=args.unlearn_md,ep=args.unlearn_ep)

    

    true_grads =[]
    for ul_param, param in zip(ul_model.parameters(), model.parameters()):
        if param.requires_grad:
            if args.unlearn_md=="taskVec":
                g=-(param.data- ul_param.data)#cancel the *lr for precision
            else:
                g=(param.data- ul_param.data)#cancel the *lr for precision
            true_grads.append(g)
    


    # If length of sentences is known to attacker keep padding fixed
    pads = None
    if args.know_padding:
        pads = [orig_batch["input_ids"].shape[1]] * orig_batch["input_ids"].shape[0]
        for sen_id in range(orig_batch["input_ids"].shape[0]):
            for i in range(orig_batch["input_ids"].shape[1] - 1, 0, -1):
                if orig_batch["input_ids"][sen_id][i] == tokenizer.pad_token:
                    pads[sen_id] = i
                else:
                    break
    print(f'Debug: ids_shape = {orig_batch["input_ids"].shape[1]}, pads = {pads}', flush=True)
    print(f'Debug: input ids = {orig_batch["input_ids"]}', flush=True)
    print(f'Debug: ref = {tokenizer.batch_decode(orig_batch["input_ids"])}', flush=True)

    x_embeds = get_init_gpt_cls(
        args,
        model,
        true_embeds.shape,
        true_labels,
        true_grads,
        lm_embeddings,
        lm_embeddings_weight,
        tokenizer,
        lm,
        lm_tokenizer,
        orig_batch["input_ids"],
        pads,
    )
    
    lm_embeddings_weight = lm_embeddings.weight.unsqueeze(0)
    if args.know_label==0:
        true_labels = torch.randn(true_labels.shape).to(device).requires_grad_(True)
        if args.opt_alg == 'adam':
            opt = optim.Adam([x_embeds,true_labels], lr=args.lr)
        elif args.opt_alg == 'bfgs':
            opt = optim.LBFGS([x_embeds,true_labels], lr=args.lr)
        elif args.opt_alg == 'bert-adam':
            opt = torch.optim.AdamW([x_embeds,true_labels], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    else:
        if args.opt_alg == 'adam':
            opt = optim.Adam([x_embeds], lr=args.lr)
        elif args.opt_alg == 'bfgs':
            opt = optim.LBFGS([x_embeds], lr=args.lr)
        elif args.opt_alg == 'bert-adam':
            opt = torch.optim.AdamW([x_embeds], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)


    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=args.lr_decay)

    print('Nsteps:',args.n_steps, flush=True)

    if pads is None:
        max_len = [x_embeds.shape[1]]*x_embeds.shape[0]
    else:
        max_len = pads

    max_vals, _ = torch.max(lm_embeddings.weight, dim=0)
    min_vals, _ = torch.min(lm_embeddings.weight, dim=0)
    init_size=lm_embeddings.weight.norm(p=2,dim=1).mean().item()

    upper_bound = max_vals.unsqueeze(0) 
    lower_bound = min_vals.unsqueeze(0)  

    best_final_error, best_final_x = None, x_embeds.detach().clone()
    ori_embeds=x_embeds.detach().clone()
    for it in range(args.n_steps):
        t_start = time.time()

        def closure():
            opt.zero_grad()

            rec_loss = get_reconstruction_loss(model, x_embeds , true_labels if args.know_label==1 else  F.softmax(true_labels, dim=-1).long(), true_grads, args, create_graph=True)

            reg_loss = ( x_embeds.norm(p=2,dim=2).mean() - init_size ).square() 


            tot_loss = rec_loss + args.mean_reg * reg_loss
            tot_loss.backward(retain_graph=True)
            return tot_loss

        error = opt.step(closure)
        if args.bound_reg==1:
            x_embeds.data=torch.clamp(x_embeds.data,min=lower_bound.unsqueeze(0),max=upper_bound.unsqueeze(0))
        if best_final_error is None or error <= best_final_error:
            best_final_error = error.item()
            best_final_x.data[:] = x_embeds.data[:]
        del error

        lr_scheduler.step()

        
        _, cos_ids = get_closest_tokens_gpt(x_embeds, lm_embeddings_weight)

        if args.use_swaps and it >= args.swap_burnin * args.n_steps and it % args.swap_every == 1:
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels if args.know_label==1 else  F.softmax(true_labels, dim=-1).long(), true_grads)

        steps_done = it+1
        if steps_done % args.print_every == 0:
            _, cos_ids = get_closest_tokens_gpt(x_embeds,  lm_embeddings_weight)
            x_embeds_proj = lm_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / lm_embeddings(cos_ids).norm(dim=2, p=2, keepdim=True)
            _, _, tot_loss_proj = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels if args.know_label==1 else  F.softmax(true_labels, dim=-1).long(), true_grads)
            perplexity, rec_loss, tot_loss = get_loss(args, lm, model, cos_ids, x_embeds, true_labels if args.know_label==1 else  F.softmax(true_labels, dim=-1).long(), true_grads)

            step_time = time.time() - t_start
            
            print('[%4d/%4d] tot_loss=%.3f (perp=%.3f, rec=%.3f), tot_loss_proj:%.3f [t=%.2fs]' % (
                steps_done, args.n_steps, tot_loss.item(), perplexity.item(), rec_loss.item(), tot_loss_proj.item(), step_time), flush=True)

            print('prediction: %s'% (tokenizer.batch_decode(cos_ids)), flush=True)
            
            tokenizer.batch_decode(cos_ids)

    
    if args.use_swaps_at_end:
        swap_at_end_it = int( (1 - args.swap_burnin) * args.n_steps // args.swap_every )
        print('Trying %i swaps' % swap_at_end_it, flush=True )
        for i in range( swap_at_end_it ):
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels if args.know_label==1 else  F.softmax(true_labels, dim=-1).long(), true_grads)
        

    x_embeds.data = best_final_x
    fix_special_tokens(x_embeds, lm_embeddings.weight, pads)
    d, cos_ids = get_closest_tokens_gpt(x_embeds, lm_embeddings_weight)
    x_embeds_proj = lm_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / lm_embeddings(cos_ids).norm(dim=2, p=2, keepdim=True)

    best_ids = cos_ids

    
    prediction, reference = [], []
    for i in range(best_ids.shape[0]):
        prediction += [remove_padding(tokenizer, best_ids[i])]
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i])]
    
    # Matching
    cost = np.zeros((x_embeds.shape[0], x_embeds.shape[0]))
    for i in range(x_embeds.shape[0]):
        for j in range(x_embeds.shape[0]):
            fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1'].mid.fmeasure
            cost[i, j] = 1.0 - fm
    row_ind, col_ind = linear_sum_assignment(cost)

    ids = list(range(x_embeds.shape[0]))
    ids.sort(key=lambda i: col_ind[i])
    new_prediction = []
    for i in range(x_embeds.shape[0]):
        new_prediction += [prediction[ids[i]]]
    prediction = new_prediction

    return prediction, reference

def print_metrics(res, suffix):
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        curr = res[metric].mid
        print(f'{metric:10} | fm: {curr.fmeasure*100:.3f} | p: {curr.precision*100:.3f} | r: {curr.recall*100:.3f}', flush=True)

    sum_12_fm = res['rouge1'].mid.fmeasure + res['rouge2'].mid.fmeasure

    print(f'r1fm+r2fm = {sum_12_fm*100:.3f}\n', flush=True)

def main():
    from huggingface_hub import login
    print( '\n\n\nCommand:', ' '.join( sys.argv ), '\n\n\n', flush=True)

    device = torch.device(args.device)
    metric = load_metric('rouge')
    dataset = TextDataset_cls(args.device, args.dataset, dataset_size=args.dataset_size, batch_size=1, split="train")
    test_dataset = TextDataset_cls(args.device, args.dataset, dataset_size=args.test_dataset_size, batch_size=1, split="test")


    lm=AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    lm.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.model_max_length = 512
    if not args.model_name=="bert-base-uncased":
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ori_model_path=f"{args.model_name}_{args.dataset}"+f"_{args.dataset_size}_ft"+".pth"
    
    init_model = AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=2).to(device)
    init_model.eval()
    init_model.config.pad_token_id = init_model.config.eos_token_id
    if os.path.exists(ori_model_path):
        model = AutoModelForSequenceClassification.from_pretrained(ori_model_path,num_labels=2).to(device)
        model.eval()
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model=deepcopy(init_model)
        model=fineTune_cls(model,tokenizer,dataset,test_dataset,lr=args.train_lr,ep=args.train_ep,bs=args.train_batch_size)
        model.save_pretrained(ori_model_path)
        
    
    att_dataset = TextDataset_cls(args.device, args.dataset, dataset_size=args.dataset_size, batch_size=args.batch_size, split="train")
    all_idxs = list(range(args.dataset_size))
    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    t_start = time.time()
    for i in range(args.n_inputs):
        t_input_start = time.time()
        sample = att_dataset[i] # (seqs, labels)
        retain_set=dataset.get_subset(set(all_idxs)-set(all_idxs[i*args.batch_size : (i+1)*args.batch_size]))
        print(f'Running input #{i} of {args.n_inputs}.')


        print('reference: ')
        for seq in sample[0]:
            print('========================')
            print(seq)

        print('========================', flush=True)
        
        prediction, reference = reconstruct(args, device, sample, metric, tokenizer, lm,model,init_model,dataset,retain_set)
        predictions += prediction
        references += reference

        print(f'Done with input #{i} of {args.n_inputs}.')
        print('reference: ')
        for seq in reference:
            print('========================')
            print(seq)
        print('========================')

        print('predicted: ')
        for seq in prediction:
            print('========================')
            print(seq)
        print('========================', flush=True)

        print('[Curr input metrics]:')
        res = metric.compute(predictions=prediction, references=reference)
        print_metrics(res, suffix='curr')

        print('[Aggregate metrics]:')
        res = metric.compute(predictions=predictions, references=references)
        print_metrics(res, suffix='agg')

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}\n\n', flush=True)

    print('Done with all.', flush=True)


if __name__ == '__main__':
    main()