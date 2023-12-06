# Controlling-semantic-segment-of-motion
Official Implementation for Frame-Level Event Representation Learning for Semantic-Level Generation and Editing of Avatar Motion(https://dl.acm.org/doi/abs/10.1145/3577190.3614175)



"Proposed" folder is the implementation of our method, and the "Comparative" is the implementation of the comparative method by Guo et al.(2022).
"Comparative" folder is almost the same to the original implementation (https://github.com/EricGuo5513/text-to-motion), but we made small changes for reproductivity.
"Eval" is the implementation for evaluating the code, and it is also based on the original implementation by Guo et al. (https://github.com/EricGuo5513/text-to-motion).
Put this folder and the dataset HumanML3D so that the folder paths become as below:

```
.../dataset/HumanML3D
.../{folder with any name}/Controlling-semantic-segment-of-motion
```

## Proposed method
For the experiment for the proposed method "Proposed", 
use python3.7.4.

Below is the explanation of the required environment.

To install torch, execute
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
.
Other libraries are written in 
```
requirements.txt
```
.
In this environment, 
use A100 GPU, and execute
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python Models.py {gpu_id}
```
．

## Comparative method
For the comparative method and the evaluation for the experiment, use python3.7.9.
The required libraries for the environment are written in
```
Comp_requirements.txt
```

For the comparative method "Comparative" and the evaluation for the experiment "Eval", use
V100 GPU.

Execute
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_length_est.py --name length_est_bigru --gpu_id {gpu_id} --dataset_name t2m
```
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_decomp_v3.py --name Decomp_SP001_SM001_H512 --gpu_id {gpu_id} --window_size 24 --dataset_name t2m
```
respectively, (this can be done in the same time) then after the processes end, execute
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_comp_v6.py --name Comp_v6_KLD01 --gpu_id {gpu_id} --lambda_kld 0.01 --dataset_name t2m
```
.

Now the model is trained.


## Evaluation
For the evaluation for the experiment, use the environment with python3.7.9 .
To train the "Eval" model, execute the commands below in the folder "Eval"
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_tex_mot_match.py --name text_mot_match --gpu_id {gpu_id} --batch_size 8 --dataset_name t2m
```

To evaluate the performance, enter the folder
"Eval", and do
```
mkdir checkpoints/t2m/Comp_v6_KLD01
scp -r ../Comparative/checkpoints/t2m/Comp_v6_KLD01/opt.txt checkpoints/t2m/Comp_v6_KLD01
```
first, and execute these commands in the environment for "Comparative" and "Eval":
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python final_evaluations.py 0 ../Comparative/checkpoints {gpu_id}
CUBLAS_WORKSPACE_CONFIG=:4096:8 python final_evaluations.py 1 Proposed {gpu_id}
```


## Visualization
In order to generate motion from text with the proposed method, 
move to the "MakeMotion" folder and run 
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python Gen.py {gpu_id} Proposed
```
.
This is only needed for visualization, and is not needed for evaluating the quantitative performance.．

In order to generate motion from text with the comparative method, enter teh "Eval" folder and execute
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python eval_comp_v6.py --name Comp_v6_KLD01 --est_length --repeat_time 3 --num_results 10 --ext default --gpu_id {gpu_id}
```
.


This is only needed for visualization, and is not needed for evaluating the quantitative performance.．




To visualize the output, enter the "Visualize" folder and execute the commands below:
```
python plotHuman.py textProposed 0
python plotHuman.py Comparative 1
```

## Editing with the proposed method
To see the edited motion, execute 
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python Comp.py {gpu_id} Proposed
```
in the "Edit_emb_example" folder with the environment for the proposed method first to obtain the edited motions.
Then the output folder "Orig_Proposed" and "Edit_Proposed" are generated.
These can be visualized by executing 
```
python plotHuman.py Orig_Proposed 0
python plotHuman.py Edit_Proposed 0
```
in the folder "ColorVisualize".
