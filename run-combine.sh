# turbo
python baselines/gpt_turbo.py --is_eval=True --dataset=aqua --model_name=turbo --eval_full=True --n_split=30 --few_shot=3
python baselines/gpt_turbo.py --is_eval=True --dataset=ekar_chinese --model_name=turbo --eval_full=True --n_split=30 --few_shot=3

# at
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=at --eval_full=True --n_split=30 --few_shot=3
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=at --eval_full=True --n_split=30 --few_shot=3

# dt
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=dt --eval_full=True --n_split=30 --few_shot=3
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=dt --eval_full=True --n_split=30 --few_shot=3

# dst
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=dst --eval_full=True --n_split=30 --few_shot=3
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=dst --eval_full=True --n_split=30 --few_shot=3

# pt
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=pt --eval_full=True --n_split=30 --few_shot=3
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=pt --eval_full=True --n_split=30 --few_shot=3

# st
python agents/single_action_agent.py --is_eval=True --dataset=aqua --model_name=st --eval_full=True --n_split=30 --few_shot=3
python agents/single_action_agent.py --is_eval=True --dataset=ekar_chinese --model_name=st --eval_full=True --n_split=30 --few_shot=3
