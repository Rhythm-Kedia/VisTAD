import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(14, 8), dpi=300)

gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], hspace=0.35, wspace=0.25)

# Panel (a): Physical Invariant Breakdown in SCADA
ax0 = fig.add_subplot(gs[:, 0])

t = np.linspace(0, 100, 201)
# Nominal trajectory for tank level: fills up then stabilizes
lit = 750 + 40 * np.sin(t[:111] / 15)
# At t = 55, attack starts: pump is commanded OFF, but level continues draining rapidly
t_attack = t[110:]
lit_attack = lit[-1] - 0.8 * (t_attack - 55) - 3 * np.sin((t_attack - 55) / 5)
lit_full = np.concatenate([lit, lit_attack[1:]])

pump = np.zeros_like(t)
pump[:111] = 680 # ON
pump[111:] = 600 # OFF (Commanded idle)

ax0.plot(t, lit_full, color='#1f77b4', lw=3.0, label='Tank Level LIT101 (mm)')
ax0.plot(t, pump, color='#2ca02c', lw=3.0, label='Pump P101 Status (Active / Idle)')
ax0.axvspan(55, 100, color='#d62728', alpha=0.15, label='Coordinated Invariant Attack')
ax0.axhline(680, color='gray', ls='--', alpha=0.5)

ax0.annotate('P101 commanded OFF but Level Drains!\nMass conservation invariant violated',
             xy=(75, 625), xytext=(45, 695),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=2, headwidth=8),
             fontsize=12, fontweight='bold', color='#b2182b',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff2f2', edgecolor='#d62728', alpha=0.9))

ax0.set_title('(a) Physical Invariant Breakdown in Industrial SCADA', fontsize=14, fontweight='bold', pad=12)
ax0.set_xlabel('Time Steps (seconds)', fontsize=12, fontweight='bold')
ax0.set_ylabel('Sensor / Actuator Reading', fontsize=12, fontweight='bold')
ax0.legend(loc='upper right', frameon=True, fontsize=11)
ax0.set_ylim(590, 800)
ax0.tick_params(labelsize=11)

# Panel (b): Multi-Domain Telemetry Dimensions
ax1 = fig.add_subplot(gs[0, 1])
domains = ['SWaT\n(Water 51D)', 'SMD\n(Cloud 38D)', 'MSL\n(Space 55D)', 'GECCO\n(Quality 9D)']
channels = [51, 38, 55, 9]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

bars = ax1.bar(domains, channels, color=colors, edgecolor='black', lw=1.2, width=0.55)
for bar, ch in zip(bars, channels):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{ch} Ch',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_title('(b) Multi-Domain High-Dimensional Telemetry', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Channel Count (C)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 68)
ax1.tick_params(labelsize=10)

# Panel (c): Latency & Edge Compression
ax2 = fig.add_subplot(gs[1, 1])
models = ['VLM Teacher\n(Qwen / DINOv2)\n[Cloud GPU]', 'VisTAD Student\n(Causal TCN)\n[Edge CPU / PLC]']
latencies = [1208.0, 0.44] # in ms
colors_c = ['#ff7f0e', '#17becf']

bars_c = ax2.bar(models, latencies, color=colors_c, edgecolor='black', lw=1.2, width=0.45)
ax2.set_yscale('log')
ax2.set_title('(c) 2,745x - 45,454x Edge Compression for PLCs', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('Inference Latency (ms) [Log]', fontsize=11, fontweight='bold')
ax2.set_ylim(0.01, 3000)

ax2.text(0, 1400, '1,208 ms\n(Cloud A100)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d62728')
ax2.text(1, 0.6, '0.44 ms\n(Edge Jetson)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#008b8b')
ax2.tick_params(labelsize=10)

plt.tight_layout()
fig.savefig('submissions/novelty_assessment/poster/figures/Intro.png', dpi=300, bbox_inches='tight')
fig.savefig('submissions/novelty_assessment/poster/figures/Intro.pdf', bbox_inches='tight')
print('Successfully generated clean Intro.png and Intro.pdf')
