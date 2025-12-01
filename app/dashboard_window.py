# ============================================================
# dashboard_window.py
# EngageSense AI - Post-Session Analytics Dashboard
# Works with engagesense_live.py
# ============================================================

import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def show_dashboard(data):
    df = pd.DataFrame(data)

    root = tk.Tk()
    root.title("EngageSense AI - Analytics Dashboard")
    root.geometry("1100x750")

    style = ttk.Style()
    style.theme_use("clam")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ============================================================
    # TAB 1 — SUMMARY REPORT
    # ============================================================

    summary_tab = ttk.Frame(notebook)
    notebook.add(summary_tab, text="📋 Summary")

    total = len(df)
    engaged = (df['engagement'] == 'Engaged').sum()
    partial = (df['engagement'] == 'Partially Engaged').sum()
    not_eng = (df['engagement'] == 'Not Engaged').sum()

    engaged_pct = (engaged / total) * 100
    partial_pct = (partial / total) * 100
    not_pct = (not_eng / total) * 100

    summary_text = f"""
📊 SESSION SUMMARY
──────────────────────────────────────────
🟢 Engaged: {engaged_pct:.1f}% ({engaged} frames)
🟡 Partially Engaged: {partial_pct:.1f}% ({partial} frames)
🔴 Not Engaged: {not_pct:.1f}% ({not_eng} frames)

⏱ Total Duration: {df['timestamp'].max():.1f} seconds
──────────────────────────────────────────
💡 INTERPRETATION
• >70% Engaged → Excellent focus.
• 40–70% → Moderate.
• <40% → Low attention — needs improvement.
"""

    txt = tk.Text(summary_tab, wrap=tk.WORD, height=18, font=("Consolas", 11),
                  bg="#1A1A1A", fg="#F5F5F5")
    txt.insert(tk.END, summary_text)
    txt.config(state=tk.DISABLED)
    txt.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    # ============================================================
    # TAB 2 — ENGAGEMENT TIMELINE
    # ============================================================

    line_tab = ttk.Frame(notebook)
    notebook.add(line_tab, text="📈 Timeline")

    fig, ax = plt.subplots(figsize=(10, 5))
    eng_map = {"Not Engaged": 0, "Partially Engaged": 1, "Engaged": 2}
    df["eng_val"] = df["engagement"].map(eng_map)

    ax.plot(df["timestamp"], df["eng_val"], color="#00BFFF", linewidth=2)
    ax.fill_between(df["timestamp"], df["eng_val"], color="#00BFFF", alpha=0.25)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Not", "Partial", "Engaged"])
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Engagement Level")
    ax.set_title("Engagement Level Over Time", fontweight="bold")
    ax.grid(alpha=0.3)

    canvas = FigureCanvasTkAgg(fig, master=line_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # TAB 3 — PIE CHART DISTRIBUTION
    # ============================================================

    pie_tab = ttk.Frame(notebook)
    notebook.add(pie_tab, text="🥧 Distribution")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    counts = df['engagement'].value_counts()
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']

    ax2.pie(counts.values, labels=counts.index,
            autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 11})
    ax2.set_title("Overall Engagement Distribution", fontweight="bold")

    canvas2 = FigureCanvasTkAgg(fig2, master=pie_tab)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # TAB 4 — GAZE & HEAD POSE BARS
    # ============================================================

    bar_tab = ttk.Frame(notebook)
    notebook.add(bar_tab, text="📊 Gaze / Head")

    fig3, axs = plt.subplots(1, 2, figsize=(10, 5))

    df["gaze"].value_counts().plot(kind='bar', ax=axs[0], color="#00CED1")
    axs[0].set_title("Gaze Direction Frequency")
    axs[0].set_xlabel("Direction")
    axs[0].set_ylabel("Count")
    axs[0].grid(axis='y', alpha=0.3)

    df["head"].value_counts().plot(kind='bar', ax=axs[1], color="#9370DB")
    axs[1].set_title("Head Pose Frequency")
    axs[1].set_xlabel("Direction")
    axs[1].set_ylabel("Count")
    axs[1].grid(axis='y', alpha=0.3)

    canvas3 = FigureCanvasTkAgg(fig3, master=bar_tab)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # TAB 5 — SEGMENT HEAT OVERVIEW
    # ============================================================

    combo_tab = ttk.Frame(notebook)
    notebook.add(combo_tab, text="🔥 Heat Segments")

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    # Divide timeline into 10 segments and compute avg engagement
    segments = df.groupby(pd.cut(df["timestamp"], bins=10))["eng_val"].mean()

    segments.plot(kind='bar', ax=ax4, color="#32CD32")
    ax4.set_title("Average Engagement per Time Segment")
    ax4.set_xlabel("Time Segments")
    ax4.set_ylabel("Avg Engagement Level")
    ax4.set_ylim(0, 2)
    ax4.grid(alpha=0.3)

    canvas4 = FigureCanvasTkAgg(fig4, master=combo_tab)
    canvas4.draw()
    canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ============================================================
    root.mainloop()
