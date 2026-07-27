"""
Sparse RNN agent on Atari.

  python atari_run.py [ROM]

ROM defaults to ALE/Pong-v5. Any ALE rom works.
Press Ctrl+C to stop; the agent saves to sparse_agent.npz on exit.
"""

from sili.sparse_rnn import SparseRNNAgent

import os
import sys
import signal
import numpy as np
import gymnasium as gym

import faulthandler
faulthandler.enable()

# todo: init direct connections or o-[i-w,i+w] connections, in a diagonal line from 0,0 to max_i, max_o
# todo: actions should take from ALL state neurons, averaging chunks, NOT [:num_actions].
#   Could use another sparse net but averaging chunks is easier and does about the same
# todo: also try doing that with inputs. The state should be chunked with averages matching inputs, and thus should be larger than inputs.
# todo: add actor critic back in to action output and add direct-ish sparse connections to actor critic
#   While we may rarely have external feedback, we can route the energy aux_loss here for better motion learning
# ── Config ────────────────────────────────────────────────────────────────────

GAME      = sys.argv[1] if len(sys.argv) > 1 else "ALE/Pong-v5"
SAVE_FILE = "sparse_agent.npz"

# Full RGB Atari frame: 210 × 160 × 3 = 100,800 inputs.
# The sparse layer learns which pixels matter — no manual preprocessing.
OBS_SIZE = 210 * 160 * 3

# State size: how many recurrent neurons.
STATE_SIZE = 100_000

# Network capacity (per layer)
MAX_WEIGHTS = 2_000_000
NUM_CPUS    = 4

# Training hyperparameters
LR                = 1e-1
IMPORTANCE_CUTOFF = 0.01
# build_probes(k) selects top-k input neurons AND top-k output neurons,
# then outer-products them -- k^2 candidate synapses, not k. Kept small
# since this runs every online step (see below); a larger k can saturate
# a layer's connectivity in a handful of steps instead of growing
# gradually (measured: k=64 on a 1000x1000 layer reaches full density in
# ONE synap_step call). Raise only after checking per-step nnz growth
# against STATE_SIZE/MAX_WEIGHTS.
SYNAPTOGENESIS_K  = 4
# No synaptogenesis-cadence knob anymore -- build_probes/synap_step/
# equalizer_step run every step (see sili.sparse_rnn's module docstring:
# they're cheap and O(1)-ish by design, throttling them would reintroduce
# lag spikes). No importance-decay knob either -- doesn't exist in the
# current SparseLinearLayer API generation (see same docstring).


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env       = gym.make(GAME, render_mode="human")
    n_actions = env.action_space.n

    agent = SparseRNNAgent(
        n_inputs          = OBS_SIZE,
        n_actions         = n_actions,
        state_size        = STATE_SIZE,
        max_weights       = MAX_WEIGHTS,
        num_cpus          = NUM_CPUS,
        lr                = LR,
        importance_cutoff = IMPORTANCE_CUTOFF,
        synaptogenesis_k  = SYNAPTOGENESIS_K,
        percent_active    = 0.03,
    )

    if os.path.exists(SAVE_FILE):
        try:
            agent.load(SAVE_FILE)
            print(f"Resumed from {SAVE_FILE} at step {agent._step_count}")
        except Exception as e:
            print(f"Warning: could not load {SAVE_FILE}: {e}")

    running = [True]
    def _stop(sig, frame): running[0] = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"{GAME}  |  obs={OBS_SIZE}  state={STATE_SIZE}  actions={n_actions}")
    print(f"input_proj: {OBS_SIZE}→{STATE_SIZE}  recurrent: {STATE_SIZE}→{STATE_SIZE}")
    print("Ctrl+C to stop and save.\n")

    episode    = 0
    total_reward = 0.0
    ep_reward  = 0.0

    obs, _  = env.reset()
    agent.reset_state()

    while running[0]:
        # Flatten and normalise: uint8 [210,160,3] → float32 [100800] in [0,1]
        x = obs.ravel().astype(np.float32) / 255.0

        action = agent.train_step(x)  # detach, forward, aux_loss.backward(), step
        obs, reward, terminated, truncated, _ = env.step(action)

        ep_reward    += reward
        total_reward += reward

        if agent._step_count % 500 == 0:
            nnz_in = agent.cell.input_proj.nnz
            nnz_rc = agent.cell.recurrent.nnz
            e_loss = float(agent.cell.energy.aux_loss.data) if agent.cell.energy.aux_loss is not None else 0.0
            print(
                f"step {agent._step_count:>7}  "
                f"ep {episode:>4}  "
                f"nnz in={nnz_in:>7} rc={nnz_rc:>7}  "
                f"e_loss={e_loss:.4f}"
            )

        if terminated or truncated:
            print(f"  episode {episode:>4}  reward={ep_reward:>8.1f}  "
                  f"total={total_reward:.0f}")
            episode   += 1
            ep_reward  = 0.0
            obs, _     = env.reset()
            agent.reset_state()

    env.close()
    agent.save(SAVE_FILE)
    print(f"Saved to {SAVE_FILE}")


if __name__ == "__main__":
    main()