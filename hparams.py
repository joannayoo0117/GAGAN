import argparse

class Hparams:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--no-cuda', action='store_true', default=False, 
        help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=int, default=0,
        help="Specifies cuda device")
    parser.add_argument('--seed', type=int, default=72, 
        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, 
        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=10, 
        help='Batch size for training / testing')
    parser.add_argument('--lr', type=float, default=0.005, 
        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
        help='Weight decay (L2 loss on parameters).')
    
    parser.add_argument('--train_test_split', type=float, default=0.05,
        help='Train-test split ratio')
    parser.add_argument('--num_positive', type=int, default=1000,
        help='Number of positive examples')
    parser.add_argument('--num_negative', type=int, default=1000,
        help='Number of negative examples')

    parser.add_argument('--n_attns', type=int, default=4,
        help='Number of attention layers.')
    parser.add_argument('--n_dense', type=int, default=3,
        help='Number of dense layers.')
    parser.add_argument('--dim_attn', type=int, default=140, 
        help='Dimension of attn layer.')
    parser.add_argument('--dim_dense', type=int, default=128, 
        help='Dimension of dense layer.')
    parser.add_argument('--dropout', type=float, default=0.3, 
        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    parser.add_argument('--model_file', type=str, default='./model/sample.pkl', 
        help='Model File for testing')
    parser.add_argument('--model_dir', type=str, default='./model',
        help='Directory to save model')
    parser.add_argument('--log_dir', type=str, default='./logs',
        help='Directory to save logs')