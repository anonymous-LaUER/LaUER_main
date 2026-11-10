"""Publication exports: ALBEF getAttMap overlay (gamma-boosted alpha) over original pixels.

Display follows ALBEF visualization.ipynb getAttMap line by line:
    attMap := (attMap - min) / max          # -> normalized heat in [0, 1]
    attMap := cubic-resize to (720, 1000)   # order=3, grid-constant
    attMap := gaussian_filter(attMap, 0.02 * max(H, W))   # display-only smoothing
    blend  := (1 - attMap**gamma) * image + attMap**gamma * jet(attMap)

The attMap**gamma term is the part ALBEF relies on and the previous revision dropped:
a plain alpha * heat ramp makes the overlay opacity fall off linearly, so every cell below
~25% of the peak disappears and a genuinely distributed map renders as isolated dots.
ALBEF instead raises the heat to 0.7 before blending at (near) full opacity, which keeps
roughly 9x more of the native 12 x 40 grid visible while preserving the peak.
"""
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

def render(path, data, heat, title, subtitle, kind, text='', alpha=1.0, dpi=200, pdf=False,
           cmap_name='jet', gamma=0.7):
    """kind in {'image', 'signal', 'matrix'}; heat is the normalized 12 x 40 CAM."""
    if not 0 < gamma <= 1:
        raise ValueError('heat gamma must be in (0, 1]')
    fig = plt.figure(figsize=(15, 11 if text else 9), layout='constrained')
    grid = fig.add_gridspec(2 if text else 1, 1, height_ratios=[8, 2] if text else [1])
    ax = fig.add_subplot(grid[0])
    cmap = plt.get_cmap(cmap_name)
    # ALBEF getAttMap: cubic resize followed by Gaussian blur (display only).
    heat = zoom(heat, (720 / heat.shape[0], 1000 / heat.shape[1]),
                order=3, mode='grid-constant', grid_mode=True)
    heat = np.clip(heat, 0, 1)
    heat = np.maximum(gaussian_filter(heat, .02 * 1000), 0)
    peak = float(heat.max())
    if peak > 0:
        heat = heat / peak
    # ALBEF blends with attMap**0.7, not with the raw intensity.
    weight = heat ** gamma
    if kind == 'matrix':
        # Same gamma applied through a norm so the colorbar still reports true values.
        ax.imshow(heat, cmap=cmap, norm=PowerNorm(gamma, vmin=0, vmax=1),
                  extent=(0, 10, 12, 0), interpolation='bilinear', aspect='auto')
        bar_norm = PowerNorm(gamma, vmin=0, vmax=1)
    else:
        rgba = cmap(heat)
        rgba[..., 3] = alpha * weight
        if kind == 'image':
            ax.imshow(data['original_image'], extent=(0, 10, 12, 0), aspect='auto')
        ax.imshow(rgba, extent=(0, 10, 12, 0), aspect='auto', interpolation='bilinear')
        if kind == 'signal':
            wave = data['waveform']
            scale = max(float(np.max(np.abs(wave))), .5)
            for lead in range(12):
                ax.plot(np.arange(5000) / 500, lead + .5 - wave[lead] * .43 / scale,
                        color='#17212b', linewidth=.55)
            ax.set_ylim(12, 0)
            # A numeric calibration bar makes amplitude legible despite stacked rows.
            bar_height = .43 / scale
            ax.plot([9.7, 9.7], [11.85, 11.85 - bar_height], color='#17212b', lw=2)
            ax.text(9.65, 11.85 - bar_height / 2, '1 mV', ha='right', va='center', fontsize=8)
            valid = data['valid_samples'] / 500
            if valid < 10:
                ax.axvspan(valid, 10, color='gray', alpha=.25, hatch='//')
                ax.text(valid, .2, ' zero padding', fontsize=8)
            ax.grid(axis='x', color='#aebbc5', alpha=.25, linewidth=.5)
        bar_norm = Normalize(0, 1)
    ax.set_yticks(np.arange(12) + .5, data['leads'])
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(11))
    ax.set_ylabel('Lead | actual waveform order' if kind != 'image' else 'Lead (image rows)')
    ax.set_xlabel('Time (s) | smoothed display; native attention grid: 12 x 40')
    ax.set_title('\n'.join(textwrap.wrap(subtitle, 135)), fontsize=9, loc='left', pad=12)
    bar = fig.colorbar(ScalarMappable(norm=bar_norm, cmap=cmap), ax=ax,
                       fraction=.025, pad=.015, ticks=[0, .25, .5, .75, 1])
    bar.set_label('Normalized positive attribution | low to high | display gamma %.2g' % gamma)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    if text:
        tx = fig.add_subplot(grid[1]); tx.axis('off')
        # Full text is always persisted in JSON; bound figure text to keep layout legible.
        shown = text if len(text) < 1300 else text[:1290] + '... [full text in JSON]'
        tx.text(0, 1, '\n'.join(textwrap.wrap(shown, 145)), va='top', fontsize=9,
                transform=tx.transAxes)
    fig.savefig(path, dpi=dpi, facecolor='white')
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
