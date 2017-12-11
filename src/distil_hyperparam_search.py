from subprocess import call
base_command = \
'python main.py --env-name=Cheatah --num-frames=1000000 --num-stack=1 --checkpoint='

losses = ['KL', 'MSE']
frac_student_rollouts = [0.0, 0.25, 0.50, 0.75, 1.0]

for i in losses:
    for f in frac_student_rollouts:
        command = 'python main.py '+\
                  '--checkpoint=
