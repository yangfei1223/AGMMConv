#!/bin/bash

for area in $(seq 5 6)
do
  echo "Train Area $area ..."
  python trainval.py --dataset S3DIS --mode train --test_area $area --batch_size 8
  echo "Test Area $area ..."
  python trainval.py --dataset S3DIS --mode test --test_area $area --batch_size 32
done
echo "Done."
