# Notes

## Experiments

- Utilizes `self.foul_count` enlarge the probability of survival at the initialization phase.
  - Significantly affects `explained variance` (Trailed: 2/2)
  - Higher `explained variance` gives possibility to improve the model from reward signal?

## Thoughts

- Due to retriable actions, average episode length does not reflect the model performance faithfully.
- PPOv5
  - tanh()
- PPOv6
  - PPOv4 + ReLU + 1 fc
- PPOv7
  - tanh + 1 fc
- PPOv8: 1x4 RotConv
    + Immediate reward (Corner)       // PPOv9
    + More updating iterations        // PPOv11
    + Higher weight
- PPOv13: Human-Design features
- PPOv15: Limited to upper-right corner
    + 0.3 x immediate reward          // PPOv16
    + 0.5 x immediate reward          // PPOv17
    + 1.0 x immediate reward          // PPOv18
    + Deeper MLP                      // PPOv19
+ PPOv20: Remove CNN
+ PPOv21: pow(x, 1.5) reward (Failed)
- PPOv22: CNN3x3 2x2 (Failed)
    - PPOv23 => Fine-tuen CNN param
    - PPOv24 => Fine-tune CNN param
    - TODO: Tune reward?
- A2Cv25:
- PPOv26:
    - Add reward for pinned row

## Code Architecture

1. Rollout Buffer is not over-writable in `PPO.__init__()`.
2. Parallel env skips dumping training log?