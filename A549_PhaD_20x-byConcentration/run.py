#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Fri May 10 01:01:13 2024
@author: marcalbesa

"""

import subprocess

if __name__ == "__main__":
    # Define the values of nb_classes you want to iterate over

    # Define the number of times you want to run the script for each value of nb_classes
    num_runs_per_nb_classes = 4

    # Iterate over each value of nb_classes
    for nb_classes in nb_classes_values:
        # Execute the script multiple times for each value of nb_classes
        for _ in range(num_runs_per_nb_classes):
            !python DAC_immunos_M.py --nb_classes str(nb_classes) --mom 0.8 --batch_size 16 --lr 0.001
            