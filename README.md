
# Paddle Recompute training Bert-Large 

## requirement
 * Python3.7+
 * CUDA 10.0 with CuDNN 7.6.5
 * NCCL 2.7.8 
 * V100 GPU


## How to Run
* compile & install paddle

```
mkdir recompute_workspace
cd recompute_workspace
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON -DWITH_MKL=OFF -DWITH_GPU=ON -DWITH_DISTRIBUTE=ON -DWITH_TESTING=OFF -DWITH_FLUID_ONLY=ON -DCUDA_ARCH_NAME=Volta

pip3 install  python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl -U
```

more detail of compiling please refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html


* install paddle fleetx lib for loading model and data

```
pip3 install fleet_x-0.0.8-py3-none-any.whl -U
```

* start training

```
bash start_job.sh
```

the log and training speed info would be found in ./log directory.

  * you could use the "use_recompute" argument in `start_job.sh` to turn ON and turn OFF the Recompute feature.

