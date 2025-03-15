import numpy as np
from transformers import AutoProcessor

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# choose a random action data
action_horizon = 10
action_data = np.random.rand(1, action_horizon * 8)    # one batch of action chunks
print(f'{action_data=}')

summary = {}

# Tokenize & decode action chunks
print(f'first with shape (1, action_horizon * 8) try-----------------------------------')
tokens = tokenizer(action_data)              # tokens = list[int]
print(f'{tokens=}')
decoded_actions = tokenizer.decode(tokens, time_horizon=action_horizon, action_dim=8)
print(f'{decoded_actions=}')
assert decoded_actions.shape == (1, action_horizon, 8), f'{decoded_actions.shape=} {action_data.shape=}'
loss = np.mean((decoded_actions[0] - action_data.reshape(action_horizon, 8)) ** 2)
print(f'loss: {loss}')
summary['first with shape (1, action_horizon * 8)'] = {'loss': loss, 'np.array(tokens).shape': np.array(tokens).shape}
print()

# alt
print(f'alt with shape (action_horizon, 8) try-----------------------------------')
action_data = action_data.reshape(action_horizon, 8) 
print(f'{action_data=}')
tokens = tokenizer(action_data)              # tokens = list[int]
print(f'{tokens=}')
decoded_actions = tokenizer.decode(tokens, time_horizon=action_horizon, action_dim=8)
print(f'{decoded_actions=}')
assert decoded_actions.shape == (1, action_horizon, 8), f'{decoded_actions.shape=} {action_data.shape=}'
loss = np.mean((decoded_actions[0] - action_data) ** 2)
print(f'loss: {loss}')
summary['alt with shape (action_horizon, 8)'] = {'loss': loss, 'np.array(tokens).shape': np.array(tokens).shape}
print()

# alt 2
print(f'alt 2 with shape (1, action_horizon, 8) try-----------------------------------')
action_data = action_data.reshape(1, action_horizon, 8) 
print(f'{action_data=}')
tokens = tokenizer(action_data)              # tokens = list[int]
print(f'{tokens=}')
decoded_actions = tokenizer.decode(tokens, time_horizon=action_horizon, action_dim=8)
print(f'{decoded_actions=}')
assert decoded_actions.shape == (1, action_horizon, 8), f'{decoded_actions.shape=} {action_data.shape=}'
loss = np.mean((decoded_actions[0] - action_data[0]) ** 2)
print(f'loss: {loss}')
summary['alt 2 with shape (1, action_horizon, 8)'] = {'loss': loss, 'np.array(tokens).shape': np.array(tokens).shape}
print()

# alt 3
print(f'alt 3 with shape (1, 1, action_horizon * 8) try-----------------------------------')
action_data = action_data.reshape(1, 1, action_horizon * 8) 
print(f'{action_data=}')
tokens = tokenizer(action_data)              # tokens = list[int]
print(f'{tokens=}')
decoded_actions = tokenizer.decode(tokens, time_horizon=action_horizon, action_dim=8)
print(f'{decoded_actions=}')
assert decoded_actions.shape == (1, action_horizon, 8), f'{decoded_actions.shape=} {action_data.shape=}'
loss = np.mean((decoded_actions[0] - action_data.reshape(action_horizon, 8)) ** 2)
print(f'loss: {loss}')
summary['alt 3 with shape (1, 1, action_horizon * 8)'] = {'loss': loss, 'np.array(tokens).shape': np.array(tokens).shape}
print()

# print summary in a nice table 
print(f'-----------summary-----------')
for key, value in summary.items():
    print(f'\t{key}: {value}')
