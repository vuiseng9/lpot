#!/usr/bin/env bash
export LOGLEVEL=DEBUG
WORKDIR=/path/to/lpot/examples/pytorch/image_recognition/imagenet/cpu/ptq

ARCH=resnet50
mkdir -p ${ARCH}_autoq_run

cd ${WORKDIR}

nohup python main.py \
    --pretrained \
    -a $ARCH \
    -b 30 \
    -e -t \
    --tuned_checkpoint ${ARCH}_autoq_run \
    --conf ./conf.${ARCH}.autoq.yaml \
    /data/dataset/imagenet/ilsvrc2012/torchvision 2>&1 | tee ${ARCH}_autoq_run/log.${ARCH}.autoq &
