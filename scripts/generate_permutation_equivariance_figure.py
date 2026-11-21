import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Publication-grade styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 6.2), dpi=300)

# Colors matching VisTAD 2.0 theme
NAVY = '#0a2d5f'
GREEN = '#146e2d'
RED = '#a01414'
BADGE_BG = '#e4eefa'
GRAY_BG = '#f8f9fa'

C1 = '#1f77b4' # Sensor 1: LIT101 (Blue)
C2 = '#ff7f0e' # Sensor 2: P101 (Orange)
C3 = '#2ca02c' # Sensor 3: FIT101 (Green)
C4 = '#9467bd' # Sensor 4: MV101 (Purple)

gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.05], wspace=0.18)

# ==============================================================================
# PANEL (A): COMMUTATIVE DIAGRAM (THE MATHEMATICAL EQUIVARIANCE PROOF)
# ==============================================================================
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(0, 100)
ax0.set_ylim(0, 100)
ax0.axis('off')

# Panel header
ax0.text(50, 97, r"(a) Commutative Equivariance: $\mathbf{MHA}(\mathbf{P}\mathbf{H}) = \mathbf{P} \cdot \mathbf{MHA}(\mathbf{H})$", 
         ha='center', va='top', fontsize=13, fontweight='bold', color=NAVY)

# --- TOP PATH: Nominal Input -> MHA -> Permute Output ---
# Input Tokens H
ax0.text(12.5, 87, r"Nominal Tokens $\mathbf{H}$", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=NAVY)
toks_top = [(r'$s_1$', 'LIT101', C1), (r'$s_2$', 'P101', C2), (r'$s_3$', 'FIT101', C3), (r'$s_4$', 'MV101', C4)]
for i, (sym, name, col) in enumerate(toks_top):
    rect = patches.FancyBboxPatch((2.5 + i*5.0, 72), 4.4, 11, boxstyle="round,pad=0.2",
                                  facecolor=col, edgecolor='black', alpha=0.9, lw=1.2)
    ax0.add_patch(rect)
    ax0.text(4.7 + i*5.0, 79.5, sym, ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
    ax0.text(4.7 + i*5.0, 74.5, name, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#ffffff')

# Arrow H -> MHA
ax0.annotate('', xy=(32, 77.5), xytext=(23.5, 77.5),
             arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=2.0, headwidth=6, shrink=0.05))

# MHA Box Top
mha_top = patches.FancyBboxPatch((32.5, 67), 24, 21, boxstyle="round,pad=0.4",
                                facecolor=BADGE_BG, edgecolor=NAVY, lw=2.0)
ax0.add_patch(mha_top)
ax0.text(44.5, 83, "Multi-Head Attention", ha='center', va='center', fontsize=10.5, fontweight='bold', color=NAVY)
ax0.text(44.5, 76.5, r"$\mathbf{A} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$", 
         ha='center', va='center', fontsize=9.5, color=NAVY)
ax0.text(44.5, 70.5, "(No Sensor Pos-Encodings)", ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#444444')

# Arrow MHA -> Z
ax0.annotate('', xy=(66, 77.5), xytext=(57, 77.5),
             arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=2.0, headwidth=6, shrink=0.05))

# Output Tokens Z
ax0.text(76.5, 87, r"Contextual $\mathbf{Z} = \mathbf{MHA}(\mathbf{H})$", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=NAVY)
z_top = [(r'$z_1$', C1), (r'$z_2$', C2), (r'$z_3$', C3), (r'$z_4$', C4)]
for i, (sym, col) in enumerate(z_top):
    rect = patches.FancyBboxPatch((66.5 + i*5.0, 72), 4.4, 11, boxstyle="round,pad=0.2",
                                  facecolor=col, edgecolor='black', alpha=0.9, lw=1.2)
    ax0.add_patch(rect)
    ax0.text(68.7 + i*5.0, 77.5, sym, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Downward Permutation Arrow P on RHS
ax0.annotate('', xy=(76.5, 43), xytext=(76.5, 68),
             arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, width=2.2, headwidth=7, shrink=0.05))
ax0.text(82.5, 55.5, r"Permute $\mathbf{P}$" + "\n" + r"(Swap $s_1 \leftrightarrow s_2$)", ha='left', va='center', 
         fontsize=9.5, fontweight='bold', color=GREEN)

# Result after Permute: P · MHA(H)
z_perm = [(r'$z_2$', C2), (r'$z_1$', C1), (r'$z_3$', C3), (r'$z_4$', C4)]
for i, (sym, col) in enumerate(z_perm):
    rect = patches.FancyBboxPatch((66.5 + i*5.0, 30), 4.4, 11, boxstyle="round,pad=0.2",
                                  facecolor=col, edgecolor='black', alpha=0.9, lw=1.2)
    ax0.add_patch(rect)
    ax0.text(68.7 + i*5.0, 35.5, sym, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax0.text(76.5, 26, r"Result: $\mathbf{P} \cdot \mathbf{MHA}(\mathbf{H})$", ha='center', va='top', fontsize=10.5, fontweight='bold', color=GREEN)


# --- BOTTOM PATH: Permute Input First -> MHA -> Identical Output ---
# Downward Permutation Arrow P on LHS
ax0.annotate('', xy=(12.5, 43), xytext=(12.5, 68),
             arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, width=2.2, headwidth=7, shrink=0.05))
ax0.text(6.0, 55.5, r"Permute $\mathbf{P}$" + "\n" + r"(Swap $s_1 \leftrightarrow s_2$)", ha='right', va='center', 
         fontsize=9.5, fontweight='bold', color=GREEN)

# Permuted Input Tokens PH
toks_bot = [(r'$s_2$', 'P101', C2), (r'$s_1$', 'LIT101', C1), (r'$s_3$', 'FIT101', C3), (r'$s_4$', 'MV101', C4)]
for i, (sym, name, col) in enumerate(toks_bot):
    rect = patches.FancyBboxPatch((2.5 + i*5.0, 30), 4.4, 11, boxstyle="round,pad=0.2",
                                  facecolor=col, edgecolor='black', alpha=0.9, lw=1.2)
    ax0.add_patch(rect)
    ax0.text(4.7 + i*5.0, 37.5, sym, ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
    ax0.text(4.7 + i*5.0, 32.5, name, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#ffffff')
ax0.text(12.5, 26, r"Permuted Input $\mathbf{P}\mathbf{H}$", ha='center', va='top', fontsize=10.5, fontweight='bold', color=GREEN)

# Arrow PH -> MHA
ax0.annotate('', xy=(32, 35.5), xytext=(23.5, 35.5),
             arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, width=2.0, headwidth=6, shrink=0.05))

# MHA Box Bottom
mha_bot = patches.FancyBboxPatch((32.5, 25), 24, 21, boxstyle="round,pad=0.4",
                                facecolor=BADGE_BG, edgecolor=GREEN, lw=2.0)
ax0.add_patch(mha_bot)
ax0.text(44.5, 41, "Multi-Head Attention", ha='center', va='center', fontsize=10.5, fontweight='bold', color=GREEN)
ax0.text(44.5, 34.5, r"$\mathbf{A}' = \mathbf{P}\mathbf{A}\mathbf{P}^\top$", 
         ha='center', va='center', fontsize=9.5, color=GREEN)
ax0.text(44.5, 28.5, r"$\mathbf{A}'\mathbf{V}' = \mathbf{P}\mathbf{A}\mathbf{V}$", 
         ha='center', va='center', fontsize=8.5, fontstyle='italic', color=GREEN)

# Arrow MHA -> Output
ax0.annotate('', xy=(66, 35.5), xytext=(57, 35.5),
             arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, width=2.0, headwidth=6, shrink=0.05))

# Center Identity Equality Box
id_box = patches.FancyBboxPatch((12, 4.0), 76, 14.5, boxstyle="round,pad=0.4",
                               facecolor='#eafaf1', edgecolor=GREEN, lw=2.2)
ax0.add_patch(id_box)
ax0.text(50, 13, r"$\mathbf{MHA}(\mathbf{P}\mathbf{H}, \mathbf{P}\mathbf{H}, \mathbf{P}\mathbf{H}) \equiv \mathbf{P} \cdot \mathbf{MHA}(\mathbf{H}, \mathbf{H}, \mathbf{H})$",
         ha='center', va='center', fontsize=11.5, fontweight='bold', color=GREEN)
ax0.text(50, 7.5, r"Exact Commutativity $\rightarrow$ Inter-sensor physical interactions invariant to channel ordering",
         ha='center', va='center', fontsize=8.5, color=GREEN)


# ==============================================================================
# PANEL (B): ARCHITECTURAL CONTRAST (STATIC ADJACENCY VS. PERMUTATION-EQUIVARIANT)
# ==============================================================================
ax1 = fig.add_subplot(gs[1])
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.axis('off')

ax1.text(50, 97, "(b) Architectural Contrast: Static vs. Universal", 
         ha='center', va='top', fontsize=13, fontweight='bold', color=NAVY)

# --- Box 1: Classical Static Adjacency (Failure Mode) ---
fail_box = patches.FancyBboxPatch((2, 53), 96, 38, boxstyle="round,pad=0.5",
                                 facecolor='#fff5f5', edgecolor=RED, lw=1.8)
ax1.add_patch(fail_box)

ax1.text(5, 86, r"Classical Baselines: Fixed Adjacency $\mathbf{A} \in \mathbb{R}^{C \times C}$",
         fontsize=10.2, fontweight='bold', color=RED)
ax1.text(5, 79.5, "• Models: MTAD-GAT, GDN, Static Graph Neural Networks", fontsize=8.8, color='#333333')
ax1.text(5, 73, "• Fatal Flaw 1 (Channel Permutation Breakdown):", fontsize=8.8, fontweight='bold', color=RED)
ax1.text(8, 67.5, r"$\mathbf{A}\mathbf{P} \neq \mathbf{P}\mathbf{A} \rightarrow$ Permuting channels destroys learned edges.", fontsize=8.0, color='#444444')
ax1.text(5, 61.5, "• Fatal Flaw 2 (Dimension Inflexibility):", fontsize=8.8, fontweight='bold', color=RED)
ax1.text(8, 56.5, r"Fixed to $C_{\mathrm{train}}$ (e.g. 51D). Incompatible with GECCO (9D) / SMD (38D).", fontsize=8.0, color='#444444')

# --- Box 2: VisTAD 2.0 Permutation-Equivariant MHA (Success) ---
pass_box = patches.FancyBboxPatch((2, 5), 96, 43, boxstyle="round,pad=0.5",
                                 facecolor='#f0f8ff', edgecolor=NAVY, lw=2.0)
ax1.add_patch(pass_box)

ax1.text(5, 43, "VisTAD 2.0: Permutation-Equivariant MHA",
         fontsize=10.2, fontweight='bold', color=NAVY)
ax1.text(5, 36.5, "• Dynamic Pairwise Relational Geometry:", fontsize=8.8, fontweight='bold', color=NAVY)
ax1.text(8, 31, r"Attention $\mathbf{A}_{i,j} \propto \exp(\mathbf{q}_i \mathbf{k}_j^\top / \sqrt{d})$ re-computes dynamically.", fontsize=8.0, color='#333333')
ax1.text(5, 25, r"• Universal Foundation Transfer ($C \in [1, \infty)$):", fontsize=8.8, fontweight='bold', color=NAVY)
ax1.text(8, 19.5, r"Zero-shot transfer: SWaT 51D $\rightarrow$ GECCO 9D (AUROC 0.8355).", fontsize=8.0, color='#333333')
ax1.text(5, 14, "• Sensor Dropout Robustness:", fontsize=8.8, fontweight='bold', color=NAVY)
ax1.text(8, 8.5, r"Survives 75% telemetry loss (13/51 surviving channels: AUROC 0.9018).", fontsize=8.0, color='#333333')

fig.savefig('submissions/novelty_assessment/poster/figures/fig_permutation_equivariance.png', dpi=300, bbox_inches='tight')
fig.savefig('submissions/novelty_assessment/poster/figures/fig_permutation_equivariance.pdf', bbox_inches='tight')
print("Successfully generated refined fig_permutation_equivariance.png and .pdf!")
