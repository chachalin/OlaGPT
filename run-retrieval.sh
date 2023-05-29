# turbo
python baselines/gpt_turbo.py --is_eval=True --dataset=aqua --model_name=turbo --eval_full=True --n_split=30 --few_shot=2
python baselines/gpt_turbo.py --is_eval=True --dataset=ekar_chinese --model_name=turbo --eval_full=True --n_split=30 --few_shot=2

# at
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=at --eval_full=True --n_split=30 --few_shot=2
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=at --eval_full=True --n_split=30 --few_shot=2

# dt
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=dt --eval_full=True --n_split=30 --few_shot=2
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=dt --eval_full=True --n_split=30 --few_shot=2

# dst
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=dst --eval_full=True --n_split=30 --few_shot=2
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=dst --eval_full=True --n_split=30 --few_shot=2

# pt
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=pt --eval_full=True --n_split=30 --few_shot=2
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=pt --eval_full=True --n_split=30 --few_shot=2

# st
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=st --eval_full=True --n_split=30 --few_shot=2
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=st --eval_full=True --n_split=30 --few_shot=2
