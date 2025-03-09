import os
from datetime import datetime
import numpy as np
from openpi.shared.image_tools import resize_with_pad
import einops
import jax.numpy as jnp
from openpi.training import config
from openpi.policies import policy_config
from openpi.policies.policy import Policy
from openpi.shared import download

def get_time():
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def myprint(s):
	print(f'{get_time()}: {s}')

def process_inputs(inputs):
	# outputs of this function need to be jax arrays of shape (batch_size, 224, 224, 3) with values in the range [-1, 1]
	# inputs are expected to be np arrays of dtype uint8
	assert isinstance(inputs, np.ndarray)
	assert inputs.dtype == np.uint8
	# if inputs is a single image, add a batch dimension
	if len(inputs.shape) == 3:
		inputs = inputs[np.newaxis, ...]
	# convert to [-1, 1] float32
	inputs = inputs.astype(np.float32) / 255.0 * 2.0 - 1.0
	# if inputs is channel first, make it channel last
	if inputs.shape[1] == 3:
		inputs = einops.rearrange(inputs, 'b c h w -> b h w c')
	# convert to jax array
	inputs = jnp.asarray(inputs)
	# if resolution is not 224x224, change resolution to 224x224. The resize_with_pad function below takes a jax array as input.
	if inputs.shape[1:3] != (224, 224):
		inputs = resize_with_pad(inputs, 224, 224)
	return inputs

def load_policy(policy_name="pi0_fast_droid"):
	train_config = config.get_config(policy_name)
	checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{policy_name}")
	policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
	return policy

def embed(inputs, policy: Policy, return_bfloat16: bool = False):
	inputs = process_inputs(inputs)
	inputs_embeddings, _ = policy._model.PaliGemma.img(inputs, train=False) # jax array of shape (batch_size, 256, 2048), dtype bfloat16
	inputs_embeddings = np.asarray(inputs_embeddings)

	# We need to convert (batch_size, 256, 2048) to (batch_size, 16, 2048)
	# 256 patches = 16x16 grid of patches
	batch_size = inputs_embeddings.shape[0]
	inputs_embeddings = inputs_embeddings.reshape(batch_size, 16, 16, 2048)
	# Average over the second dimension to get (batch_size, 16, 2048)
	inputs_embeddings = inputs_embeddings.mean(axis=2)
	# Now reshape to (batch_size, 16*2048)
	inputs_embeddings = inputs_embeddings.reshape(batch_size, 16*2048)
	
	if not return_bfloat16:
		inputs_embeddings = inputs_embeddings.astype(np.float32) # convert to float32
	return inputs_embeddings

def embed_with_batches(inputs, policy: Policy, return_bfloat16: bool = False, batch_size: int = 256):
	all_inputs_embeddings = []
	for i in range(0, len(inputs), batch_size):
		inputs_batch = inputs[i:i+batch_size]
		inputs_embeddings = embed(inputs_batch, policy, return_bfloat16)
		all_inputs_embeddings.append(inputs_embeddings)
	return np.concatenate(all_inputs_embeddings, axis=0)

	