# Framework for Compositional Reinforcement Learning

This repository contains code for the compositional learning framework presented in the paper
__Robust Subtask Learning for Compositional Generalization__ by Kishor Jothimurugan, Steve Hsu,
Osbert Bastani and Rajeev Alur, published in ICML 2023.

## Dependencies

Python version is 3.7.10. Install dependencies in requirements.txt, preferably in a virtual environment:

```
pip install -r requirements.txt
```

## Experiments

To run the experiments presented in the papar, move to the directory corresponding to the environemnt. For rooms environment,

```
cd test/rooms
```

For F110 environment,

```
cd test/f110_turn
```

After changing the working directory appropriately, run the command corresponding to the algorithm to use. To run ROSAC:

```
python masac.py -d {save_directory} -n {run_number} -v {gpu_number} -g -z
```

For the ablation in which subtasks are picked randomly during training, use the above command without the `-z` option. To run AROSAC,
the asynchronous version of the algorithm:

```
python cegrl.py -d {save_directory} -n {run_number} -v {gpu_number} -g -z -c
```

In the above command, omit option '-z' for the DAGGER baseline and omit both '-z' and '-c' for the NAIVE baseline. To run the MADDPG baseline,

```
python maddpg.py -d {save_directory} -n {run_number}
```

To run the PAIRED baseline,

```
python paired.py d {save_directory} -n {run_number} -v {gpu_number} -g
```