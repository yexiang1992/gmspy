r"""
.. _ref_ex_resample:

Resampling
--------------------------------------------------------

see :func:`gmspy.resample`.
"""
import numpy as np
import matplotlib.pyplot as plt
import gmspy as gm

ts, acc = gm.load_gm_examples('Imperial_Valley')
dt = ts[1] - ts[0]
dtis = [dt / 2, 2 * dt]

colors = ['b', 'r']
fig, ax = plt.subplots(figsize=(9, 4))
ax.hlines(0, np.min(ts), np.max(ts), lw=0.5, colors='gray')
ax.plot(ts, acc, c='gray', lw=1,
        label=f"origin, dt={dt}, num={len(ts)}", alpha=1)
for i, dti in enumerate(dtis):
    ts2, acc2 = gm.resample(dt, acc, dti)
    ax.plot(ts2, acc2, c=colors[i], lw=1, alpha=1,
            label=f"dti={dti}, num={len(ts2)}")
ax.set_xlim(np.min(ts), np.max(ts))
ax.grid(False)
ax.set_ylabel('acceleration', fontsize=12)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12)
plt.show()
