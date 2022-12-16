import os

model_root = 'run/wenyan'
try:
    checkpoints = [x for x in os.listdir(model_root) if x.endswith('.pt')]
except FileNotFoundError:
    os.mkdir(model_root)
    checkpoints = [x for x in os.listdir(model_root) if x.endswith('.pt')]

last_checkpoint = None
if len(checkpoints) > 0:
    checkpoints = sorted(checkpoints, key=lambda x: int(x[:-3].split('_')[-1]))
    last_checkpoint = checkpoints[-1]
    last_checkpoint = os.path.join(model_root, last_checkpoint)

if last_checkpoint is not None:
    print('last_checkpoint', last_checkpoint, os.path.exists(last_checkpoint))
    if isinstance(last_checkpoint, str) and os.path.exists(last_checkpoint):
        # 中断后继续训练使用
        os.system('onmt_train -config wenyan.yaml --train_from="%s"' % last_checkpoint)
    else:
        os.system('onmt_train -config wenyan.yaml')
else:
    os.system('onmt_train -config wenyan.yaml')
