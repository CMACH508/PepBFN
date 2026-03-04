#!/bin/bash
# 参数检查
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ROOT_DIR> [SIDE_CHAIN_PACKING]"
    exit 1
fi

# 参数设置
ROOT_DIR="$1"                              # 第一个参数：根目录
SIDE_CHAIN_PACKING="${2:-false}"          # 第二个参数：是否开启侧链打包，默认 false

echo "ROOT_DIR = $ROOT_DIR"
echo "SIDE_CHAIN_PACKING = $SIDE_CHAIN_PACKING"

if [ "$SIDE_CHAIN_PACKING" = true ]; then
    python core/callbacks/side_chain_packer.py --root_dir $ROOT_DIR
    wait
    python core/callbacks/evaluate.py --root_dir $ROOT_DIR --sc_packing
    wait
    python train_eval_other.py --root_dir $ROOT_DIR --sc_packing
    wait
    for i in {0..10}
    do
        echo "Running energy evaluation rank $i"
        python core/callbacks/energy.py --root_dir $ROOT_DIR --sc_packing --num_workers 4 --rank $i &
    done
    wait
else
    python core/callbacks/evaluate.py --root_dir $ROOT_DIR
    wait
    python train_eval_other.py --root_dir $ROOT_DIR
    wait
    for i in {0..10}
    do
        echo "Running energy evaluation rank $i"
        python core/callbacks/energy.py --root_dir $ROOT_DIR --num_workers 4 --rank $i &
    done
    wait
fi

