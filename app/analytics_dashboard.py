import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


def show_dashboard(root, data):
    df = pd.DataFrame(data)
    root.title("EngageSense AI - Analytics Dashboard")

    for widget in root.winfo_children():
        widget.destroy()

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ===============================
    # 📋 TAB 1: Summary Report
    # ===============================
    summary_tab = ttk.Frame(notebook)
    notebook.add(summary_tab, text="📋 Summary")

    total = len(df)
    engaged = (df['engagement'] == 'Engaged').sum()
    partial = (df['engagement'] == 'Partially Engaged').sum()
    noteng = (df['engagement'] == 'Not Engaged').sum()

    engaged_pct = (engaged / total) * 100
    partial_pct = (partial / total) * 100
    noteng_pct = (noteng / total) * 100

    summary_text = f"""
📊 SESSION SUMMARY
───────────────────────────────
✅ Engaged: {engaged_pct:.1f}% ({engaged} frames)
⚠️ Partially Engaged: {partial_pct:.1f}% ({partial} frames)
❌ Not Engaged: {noteng_pct:.1f}% ({noteng} frames)

⏱ Total Duration: {df['timestamp'].max():.1f} seconds
───────────────────────────────
💡 RECOMMENDATION:
- >70% → Excellent engagement
- 40-70% → Moderate engagement
- <40% → Low focus — consider re-engagement
"""
    text_box = tk.Text(summary_tab, wrap=tk.WORD, height=18, font=("Consolas", 11), bg="#101820", fg="#F5F5F5")
    text_box.insert(tk.END, summary_text)
    text_box.config(state=tk.DISABLED)
    text_box.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    # ===============================
    # 📈 TAB 2: Line Chart Over Time
    # ===============================
    line_tab = ttk.Frame(notebook)
    notebook.add(line_tab, text="📈 Engagement Over Time")

    fig, ax = plt.subplots(figsize=(10, 5))
    eng_map = {"Not Engaged": 0, "Partially Engaged": 1, "Engaged": 2}
    df["eng_val"] = df["engagement"].map(eng_map)
    ax.plot(df["timestamp"], df["eng_val"], color="#00BFFF", linewidth=2)
    ax.fill_between(df["timestamp"], df["eng_val"], color="#00BFFF", alpha=0.3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Not", "Partial", "Engaged"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Engagement Level")
    ax.set_title("Engagement Level Over Time", fontweight="bold")
    ax.grid(True, alpha=0.3)

    canvas = FigureCanvasTkAgg(fig, master=line_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ===============================
    # 🥧 TAB 3: Engagement Distribution
    # ===============================
    pie_tab = ttk.Frame(notebook)
    notebook.add(pie_tab, text="🥧 Engagement Distribution")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    engagement_counts = df['engagement'].value_counts()
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    ax2.pie(engagement_counts.values, labels=engagement_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 11})
    ax2.set_title("Engagement Distribution", fontweight="bold")

    canvas2 = FigureCanvasTkAgg(fig2, master=pie_tab)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ===============================
    # 📊 TAB 4: Vertical Bar Charts
    # ===============================
    bar_tab = ttk.Frame(notebook)
    notebook.add(bar_tab, text="📊 Gaze & Head Analysis")

    fig3, axs = plt.subplots(1, 2, figsize=(10, 5))
    df['gaze'].value_counts().plot(kind='bar', color="#00CED1", ax=axs[0])
    axs[0].set_title("Gaze Direction Frequency")
    axs[0].set_xlabel("Direction")
    axs[0].set_ylabel("Count")
    axs[0].grid(axis="y", alpha=0.3)

    df['head'].value_counts().plot(kind='bar', color="#9370DB", ax=axs[1])
    axs[1].set_title("Head Pose Frequency")
    axs[1].set_xlabel("Direction")
    axs[1].set_ylabel("Count")
    axs[1].grid(axis="y", alpha=0.3)

    fig3.tight_layout()
    canvas3 = FigureCanvasTkAgg(fig3, master=bar_tab)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ===============================
    # 🔁 TAB 5: Combined Heat Overview
    # ===============================
    combo_tab = ttk.Frame(notebook)
    notebook.add(combo_tab, text="🔥 Combined Overview")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    combined = df.groupby(pd.cut(df['timestamp'], bins=10))['eng_val'].mean()
    combined.plot(kind="bar", color="#32CD32", ax=ax4)
    ax4.set_title("Average Engagement per Time Interval")
    ax4.set_xlabel("Time Segments")
    ax4.set_ylabel("Avg Engagement Level")
    ax4.set_ylim(0, 2)
    ax4.grid(True, alpha=0.3)

    canvas4 = FigureCanvasTkAgg(fig4, master=combo_tab)
    canvas4.draw()
    canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
