from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_Train = True
        self.parser.add_argument('--batchsize', type=int, default=64,
                                help='input batch size')
        self.parser.add_argument('--lr', type=float, default=0.0002,
                                help='learning rate')
        self.parser.add_argument('--N_expr', type=int, default=6,
                                help='number of expression class')
        self.parser.add_argument('--vgg_coeff', type=float, default=1,
                                help='weight of G/D adv loss')
        self.parser.add_argument('--q_coeff', type=float, default=1,
                                help='weight of G expr loss')
        self.parser.add_argument('--fm_coeff', type=float, default=1,
                                help='weight of G id loss')
        self.parser.add_argument('--count_epoch', type=int, default=1,
                                help='strat num of count epoch')
        self.parser.add_argument('--epochs', type=int, default=1001,
                                help='nums of eopch')
        self.parser.add_argument('--beta1', type=float, default=0.5, 
                                help='adam parameter 1')
        self.parser.add_argument('--beta2', type=float, default=0.999,
                                help='adam parameter 2')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50,
                                help='frequency of saving result')
        self.parser.add_argument('--D_interval', type=int, default=1,
                                help='interval of optimization D')
        self.parser.add_argument('--G_interval', type=int, default=1,
                                help='interval of optimization G')
        self.parser.add_argument('--load_epoch', type=int, default=0,
                                help='num of load epoch')
                                