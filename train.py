# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import os
os.environ['FLAGS_enable_parallel_graph'] = "0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
os.environ['FLAGS_check_nan_inf'] = "1"
os.environ['FLAGS_memory_fraction_of_eager_deletion'] = "1"

import time
import numpy as np
import sys
import fleetx as X
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import argparse
import six
import ast
import paddle.distributed.fleet.meta_optimizers.sharding as sharding

paddle.enable_static()

def print_param(program, print_vars):
    from paddle.fluid.framework import Parameter
    def is_parameter(var):
        return var.name in print_vars
    def get_tensor(var):
        t = paddle.fluid.global_scope().find_var(var.name).get_tensor()
        return np.array(t)
    def get_name(var):
        return var.name
    parameter_list = list(filter(is_parameter, program.list_vars()))
    for p in sorted(parameter_list, key=get_name):
        print("{} : {}".format(p.name, p.shape))
        print(get_tensor(p)[:20])

def parse_args():
    parser = argparse.ArgumentParser("bert-large")
    parser.add_argument(
        "--shuffle", type=ast.literal_eval, default=True, help="Whether use sharding")
    parser.add_argument(
        "--save_model", type=ast.literal_eval, default=False, help="Whether save_model")
    parser.add_argument(
        "--run_id", type=str, default="bert-large", help="name of this run")    
    parser.add_argument(
        "--use_amp", type=ast.literal_eval, default=False, help="Whether use amp(Automatic Mixed Precision)")
    parser.add_argument(
        "--use_recompute", type=ast.literal_eval, default=False, help="Whether use recompute.")
    parser.add_argument(
        "--use_sharding", type=ast.literal_eval, default=False, help="Whether use sharding")   
    parser.add_argument(
        "--hybrid_dp", type=ast.literal_eval, default=False, help="Whether use sharding")   
    parser.add_argument(
        "--offload", type=ast.literal_eval, default=False, help="Whether use offload")   
    parser.add_argument(
        "--sharding_group_size", type=int, default=8, help="Whether use sharding")   

    args = parser.parse_args()
    return args

def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in six.iteritems(vars(args)):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def set_seed(prog, mode="op"):
  if mode == "op":
    for op in prog.global_block().ops:
      if op.desc.has_attr('seed'):
        op._set_attr('seed', 9000)
  if mode == "prog":
    prog.random_seed = 9000

def set_dropout_seed(model):
  for prog in [model.main_prog, model.startup_prog]:
    for op in prog.global_block().ops:
      if op.desc.has_attr('fix_seed'):
        op._set_attr('fix_seed', True)

def disable_dropout(model):
  for prog in [model.main_prog, model.startup_prog]:
    for op in prog.global_block().ops:
      if op.desc.has_attr('dropout_prob'):
        op._set_attr('dropout_prob', 0.0)


def main(args):


    np.random.seed(9001) 

    run_id = args.run_id
    if not os.path.isdir(run_id):
        os.system('mkdir -p {}'.format(run_id))

    profile = False
    batch_size = 512 * 50
    lr = 1e-4

    fleet.init(is_collective=True)
    # load Bert_large / Bert_base model
    model = X.applications.BertLarge(lang="en")

    model.main_prog.random_seed=9001
    model.startup_prog.random_seed=9001

    local_path="./data"
    data_loader = model.get_val_dataloader(
        data_dir='{}'.format(local_path),
        max_seq_len=512,
        batch_size=batch_size,
        in_tokens=True,
        shuffle=False
    )

    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 2
    exec_strategy.num_iteration_per_drop_scope = 1

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy
    dist_strategy.nccl_comm_num = 1
    dist_strategy.amp = args.use_amp

    # recompute 
    checkpoints = ['elementwise_add_{}.tmp_0'.format(i * 2) for i in range(1,24)]
    dist_strategy.recompute = args.use_recompute
    if args.use_recompute :
        dist_strategy.recompute_configs = {
            "checkpoints": checkpoints
            }

    scheduled_lr = X.utils.linear_warmup_decay(lr, warmup_steps=4000,
                                                num_train_steps=1000000)
    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)

    clip_norm_thres = 1.0
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres))

    ops, param_grads = optimizer.minimize(model.loss)


    filename = "./" + args.run_id + "/main_program.txt"
    with open(filename + str(int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
        f.write(str(fluid.default_main_program()))
    filename = "./" + args.run_id + "/start_program.txt"
    with open(filename + str(int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
        f.write(str(fluid.default_startup_program()))

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    model.main_prog.random_seed=9001
    model.startup_prog.random_seed=9001
    np.random.seed(9001) 
    
    fetch_list = [model.loss.name] + list(model.target.values()) + \
                    [scheduled_lr.name, "loss_scaling_0"]

    start_time = -1
    speeds = []
    profile = False
    costs = []
    accs = []

    print("============start training============")
    for i, data in enumerate(data_loader()):
        
        # profile
        if profile and i == 2050:
            print("begin profiler")
            profiler.start_profiler("All")
        elif profile and i == 2065:
            print("end profiler")
            filename = "./run_id/profile_" + str(fleet.worker_index())
            profiler.stop_profiler("total", filename)
            print("end profiler break!")
            print("avg speed = {} step / s".format(np.mean(speeds)))
            sys.exit("profile finish !")


        cost_val, next_sent_acc, lm_loss, np_lr, loss_scaling_0  = exe.run(fluid.default_main_program(),
                        feed=data,
                        fetch_list=fetch_list,
                        use_program_cache=True)

        costs.append(cost_val[0])
        accs.append(next_sent_acc[0])
    
        # count speed
        if (i + 1) % 10 == 0 :

            duration = time.time() - start_time
            speed = 10 / duration
            print("step {}, loss {}, acc {}, np_lr {}". \
            format(i, np.mean(costs), np.mean(accs), np_lr[0]))
            start_time = time.time()
            costs = []
            accs  = []


                            

if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    main(args)
