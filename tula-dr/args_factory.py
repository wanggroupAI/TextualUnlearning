import argparse

def get_args():
    parser = argparse.ArgumentParser(description='TULA-DR')


    parser.add_argument('--dataset',  required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--loss', choices=['cos', 'l2'], required=True)
    parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, required=True) 

    parser.add_argument('--coeff_reg', type=float, default=0.1) 
    
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--opt_alg', choices=['adam', 'bfgs', 'bert-adam'], default='adam')
    parser.add_argument('--n_steps', type=int, default=2000) 
    parser.add_argument('--init_candidates', type=int, default=500) 
    parser.add_argument('--init', choices=['lm', 'random',"my","latent"], default='random')

    #for swap trick in https://github.com/eth-sri/lamp
    parser.add_argument('--use_swaps', type=bool, default=True) 
    parser.add_argument('--no-use_swaps', dest='use_swaps', action='store_false')
    parser.add_argument('--use_swaps_at_end', action='store_true')    
    parser.add_argument('--swap_burnin', type=float, default=0.1)
    parser.add_argument('--swap_every', type=int, default=200)

    parser.add_argument('--use_embedding', action='store_true')
    parser.add_argument('--know_padding', type=bool, default=True)
    parser.add_argument('--init_size', type=float, default=1.5041) 
    parser.add_argument('--defense', type=str, default='0')
    

    parser.add_argument('--coeff_perplexity', type=float, default=0.1) 
    parser.add_argument('--mean_reg', type=float, default=0.1) 
    parser.add_argument('--bound_reg', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=0.01) 
    
    parser.add_argument('--train_lr', type=float, default=2e-5) 
    parser.add_argument("--dataset_size", type=int, default=1000, help="Data size") 
    parser.add_argument("--test_dataset_size", type=int, default=200, help="Data size")
    parser.add_argument('--train_ep', type=int, default=2)
    parser.add_argument('--unlearn_ep', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--unlearn_md', type=str, default='ga')
    parser.add_argument('--know_label', type=int, default=1)

    parser.add_argument('--lr_decay', type=float, default=0.9) 

    parser.add_argument('--print_every', type=int, default=50)

    args = parser.parse_args()

    
    return args
