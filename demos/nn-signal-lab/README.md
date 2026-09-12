# Week 2/3 Demo - Signal Propagation Lab

Interactive demo for the signal-propagation lecture (ME2605 / DDA6204).

1. **Background (5 slides)** - what a fully connected network is, pre-activation vs
   post-activation, what happens to the scale with depth, and where the initialization
   gain enters. Follows *Optimization for Deep Learning*, Sec. 2.1.
2. **The Lab** - three dashboards, all computed live in the browser:
   - per-layer magnitude of pre-activation, post-activation and back-propagated error
   - the pre-activation distribution at a chosen layer, on a fixed axis
   - training / test **error** (not loss) on two moons, plus the decision boundary

Everything runs client-side: no server, no data leaves the page, and it works offline.
Deep links: `?g=0.7&L=12&d=48&act=relu&v=lab`.
