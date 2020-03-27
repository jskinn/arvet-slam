#!/usr/bin/env bash
SEQUENCES=(
  "MH_01_easy"
  "MH_02_easy"
  "MH_03_medium"
  "MH_04_difficult"
  "MH_05_difficult"
)
for SEQ in "${SEQUENCES[@]}"
do
  ZIP="$SEQ.zip"
  if [ ! -f $ZIP ]; then
    wget "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/$SEQ/$SEQ.zip"
  fi
  if [ ! -d $SEQ ] && [ -f $ZIP ]; then
    unzip -n $ZIP -d $SEQ
  fi
done

SEQUENCES=(
  "V1_01_easy"
  "V1_02_medium"
  "V1_03_difficult"
)
for SEQ in "${SEQUENCES[@]}"
do
  ZIP="$SEQ.zip"
  if [ ! -f $ZIP ]; then
    wget "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/$SEQ/$SEQ.zip"
  fi
  if [ ! -d $SEQ ] && [ -f $ZIP ]; then
    unzip -n $ZIP -d $SEQ
  fi
done

SEQUENCES=(
  "V2_01_easy"
  "V2_02_medium"
  "V2_03_difficult"
)
for SEQ in "${SEQUENCES[@]}"
do
  ZIP="$SEQ.zip"
  if [ ! -f $ZIP ]; then
    wget "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/$SEQ/$SEQ.zip"
  fi
  if [ ! -d $SEQ ] && [ -f $ZIP ]; then
    unzip -n $ZIP -d $SEQ
  fi
done


