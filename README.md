Start :

Install pip env by requirements.txt

Obtain the Weather, ETT, Traffic, and Electricity benchmarks from Google Drive 'https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy' provided in paper 'Autoformer'; obtain the Solar benchmark from 'https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-' provided in paper 'DLinear'; create a directory named 'dataset' and put them into 'dataset'

Train and test Latentformer by 'bash scripts/xxx.sh'. You can directly check the main results reported in the paper by the logs in the directory named 'results in paper'. Additionally, for different paradigms, please change the args.paradigm with 'informer/patchtst/itrm' in run_longExp.py.

If you want to reproduct the ablation study of  RQ1 in the paper, please employ the Hadamard product or addition version in 'ablation' folder to replace the original files in 'PatchTST/iTransformer' project.
