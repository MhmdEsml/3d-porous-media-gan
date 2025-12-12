import jax
import optax
from flax.training import train_state, checkpoints

class TrainState(train_state.TrainState):
    pass

def create_train_state(module, rng, learning_rate, beta1, beta2, *dummy_inputs):
    params = module.init(rng, *dummy_inputs)['params']
    tx = optax.adam(learning_rate, b1=beta1, b2=beta2)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)

def load_checkpoint_if_exists(save_dir, state, prefix):
    try:
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=save_dir,
            target=state,
            prefix=prefix
        )
        step = restored_state.step if hasattr(restored_state, 'step') else 0
        return restored_state, step
    except ValueError:
        return state, 0
