import os

result_path = "output/train"
output_file = "experiments.csv"

exps = os.listdir(result_path)
exps.sort()

with open(output_file, 'w') as fw:
    fw.write("model,epoch,train_loss,eval_loss,eval_top1,eval_top5,lr\n")
    for exp in exps:
        if exp[-5] == "1" or exp[-5] == "m": continue
        exp_path = os.path.join(result_path, exp)
        with open(os.path.join(exp_path, "summary.csv")) as fr:
            last_line = fr.readlines()[-1]
            fw.write(f"{exp}\n{last_line}\n")

