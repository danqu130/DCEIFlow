import os
import argparse
import datetime

class ArgParser:
    def __init__(self):
        self.args = None
        self.parse = argparse.ArgumentParser()
        self.parse.add_argument("--batch_size", type=int, default=1, help="")
        self.parse.add_argument("--test_batch_size", type=int, default=-1, help="")
        self.parse.add_argument("--lr", type=float, default=0.0001, help="")
        self.parse.add_argument("--weight_decay", type=float, default=0.0001, help="")
        self.parse.add_argument("--epsilon", type=float, default=1e-8, help="")
        self.parse.add_argument("--clip", type=float, default=1.0, help="")

        self.parse.add_argument("--distributed", type=str, default="ddp", help="ddp(data distributed) or dp(data parallel)")
        self.parse.add_argument("--ip", type=str, default="127.0.0.1", help="ddp ip")
        self.parse.add_argument("--port", type=str, default="23130", help="ddp port")

        self.parse.add_argument('--gpus', type=int, nargs='+', default=[-1])
        self.parse.add_argument('--jobs', type=int, default=4)
        self.parse.add_argument('--mixed_precision', action='store_true', default=False)
        self.parse.add_argument("--resume", action='store_true', help="")
        self.parse.add_argument("--checkpoint", type=str, default="", help="")
        self.parse.add_argument('--weight_fix', type=str, default="", \
            help='fix some weights for transfer training')

        self.parse.add_argument("--model", type=str, default="DCEIFlow", help="")
        self.parse.add_argument('--iters', type=int, default=6, help="iters from low level to higher")
        self.parse.add_argument("--backbone", type=str, default="BasicEncoder", help="")
        self.parse.add_argument("--corr", type=str, default="Corr", help="")
        self.parse.add_argument("--decoder", type=str, default="Updater", help="")
        self.parse.add_argument('--small', action='store_true', default=False, help='use small model')
        self.parse.add_argument('--warm_start', action='store_true', default=False, help='use warm start in evaluate stage')

        self.parse.add_argument('--event_bins', type=int, default=5, \
            help='number of bins in the voxel grid event tensor')
        self.parse.add_argument('--no_event_polarity', dest='no_event_polarity', action='store_true', \
            default=False, help='Don not divide event voxel by polarity')

        self.parse.add_argument("--loss", type=str, nargs='+', default=["L1Loss"], help="")
        self.parse.add_argument("--loss_gamma", type=float, default=0.8, help="")
        self.parse.add_argument("--loss_weights", type=float, nargs='+', default=[1.0], help="")

        self.parse.add_argument("--name", type=str, default="", help="")
        self.parse.add_argument("--task", type=str, default="train", help="")
        self.parse.add_argument("--stage", type=str, default="chairs2", help="")
        self.parse.add_argument("--metric", type=str, nargs='+', default=["epe"], help="")

        self.parse.add_argument('--isbi', action='store_true', default=False, help='bidirection flow training')
        self.parse.add_argument("--seed", type=int, default=20, help="")
        self.parse.add_argument('--not_save_board', action='store_true', default=False)
        self.parse.add_argument('--not_save_log', action='store_true', default=False)
        self.parse.add_argument("--log_path", type=str, default="logs", help="")

        self.parse.add_argument('--run_epoch', action='store_true', default=True)
        self.parse.add_argument("--epoch", type=int, default=200, help="")
        self.parse.add_argument("--step", type=int, default=20000, help="")
        self.parse.add_argument("--summary_step", type=int, default=1000, help="")
        self.parse.add_argument("--eval_feq", type=int, default=5, help="every summary_step*eval_feq for step, every eval_feq for epoch")
        self.parse.add_argument("--save_feq", type=int, default=5, help="every summary_step*save_feq for step, every save_feq for epoch")
        self.parse.add_argument("--save_path", type=str, default="logs", help="")
        self.parse.add_argument("--debug", action='store_true', default=False)

        self.parse.add_argument("--crop_size", type=int, nargs='+', default=[368, 496])
        self.parse.add_argument("--pad", type=int, default=8, help="")

        self.parse.add_argument('--skip_num', type=int, default=1, help='skip images in dataset to get more events')
        self.parse.add_argument('--skip_mode', type=str, default='i', \
            help='skip images mode in dataset to get more events i(interrupt)/c(continue)')

    def _print(self):
        print(">>> ======= Options ==========")
        for k, v in vars(self.args).items():
            print(k, '=', v)
        print("<<< ======= Options ==========")

    def logger_debug(self, logger):
        for k, v in vars(self.args).items():
            logger.log_debug('{}\t=\t{}'.format(k, v), "argparser")

    def parser(self):
        self.args = self.parse.parse_args()

        if self.args.test_batch_size == -1:
            self.args.test_batch_size = self.args.batch_size

        if self.args.name == '' or self.args.task[:4] == 'test' or self.args.task == 'submit':
            self.args.not_save_board = True
            self.args.not_save_log = True
            if self.args.task != 'test':
                print("not_save_board and not_save_log are set to True")

        time = datetime.datetime.now()
        self.args.log_name = "{:0>2d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_{}".format(time.year % 100,
            time.month, time.day, time.hour, time.minute, time.second, self.args.name) if self.args.name != "" else ""
        self.args.log_path = os.path.join(self.args.log_path, self.args.log_name)
        self.args.save_path = os.path.join(self.args.save_path, self.args.log_name)

        if self.args.task != 'test':
            self._print()

        return self.args
