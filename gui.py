import customtkinter as ctk
from tkinter import ttk
import tkinter as tk

import cursor
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

from create_myData import calc_similarity_score

# Set appearance mode and color theme
ctk.set_appearance_mode("light")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  # "blue", "green", or "dark-blue"

# === Konfiguration: Dateipfade ===
mydata_path = "data/mydata.csv"
sim_score_matrix_path = "data/similarity_score_matrix.csv"

# === Daten-Variablen ===
mydata = pd.DataFrame()
sim_score_matrix = pd.DataFrame()
sort_state = {}
matrix_sort_state = {}
matched_rows_tree_select = pd.DataFrame()
matched_rows_matrix_tree_select = pd.DataFrame()
radar_canvas = None
current_weights = None


def update_table(df):
    tree.delete(*tree.get_children())

    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col, command=lambda _col=col: sort_column(_col))
        tree.column(col, anchor="w", width=150, stretch=False)

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))


def sort_column(col):
    global mydata
    ascending = sort_state.get(col, True)
    mydata_sorted = mydata.sort_values(by=col, ascending=ascending)
    sort_state[col] = not ascending
    update_table(mydata_sorted)


def sort_matrix_column(col):
    items = [(matrix_tree.set(k, col), k) for k in matrix_tree.get_children('')]

    # Automatisch erkennen ob Zahl oder Text
    try:
        items.sort(key=lambda t: float(t[0]), reverse=matrix_sort_state.get(col, False))
    except ValueError:
        items.sort(key=lambda t: t[0], reverse=matrix_sort_state.get(col, False))

    for index, (_, k) in enumerate(items):
        matrix_tree.move(k, '', index)

    matrix_sort_state[col] = not matrix_sort_state.get(col, False)

def update_matrix_view(index):
    matrix_tree.delete(*matrix_tree.get_children())
    col = sim_score_matrix[index].drop(index)
    for i, val in col.items():
        matrix_tree.insert("", "end", values=(mydata.loc[i, 'player_name'], mydata.loc[i, 'player_position'], round(val, 4)))

def on_row_select(event):
    global matched_rows_tree_select
    global radar_canvas
    if radar_canvas is not None:
        radar_canvas.get_tk_widget().destroy()
        radar_canvas = None
    selected_item = tree.focus()
    values = tree.item(selected_item, 'values')

    if not values or len(values) < 2:
        return  # Kein valider Eintrag ausgewählt

    try:
        # Spieler anhand der ID suchen
        player_id = int(values[1])  # Sicherstellen, dass das wirklich 'player_id' ist
        matched_rows_tree_select = mydata[mydata['player_id'] == player_id]
        if not matched_rows_tree_select.empty:
            update_matrix_view(matched_rows_tree_select.index[0])
    except Exception as e:
        print("Fehler bei Auswahl:", e)


def on_row_select_matrix_tree(event):
    global matched_rows_matrix_tree_select
    global radar_canvas
    if radar_canvas is not None:
        radar_canvas.get_tk_widget().destroy()
        radar_canvas = None
    selected_item = matrix_tree.focus()
    values = matrix_tree.item(selected_item, 'values')

    if not values or len(values) < 2:
        return  # Kein valider Eintrag ausgewählt

    try:
        # Spieler anhand des Namens suchen
        player_name = values[0]  # Sicherstellen, dass das wirklich 'player_name' ist
        matched_rows_matrix_tree_select = mydata[mydata['player_name'] == player_name]
        if not matched_rows_matrix_tree_select.empty:
            draw_radar_chart()
    except Exception as e:
        print("Fehler bei Auswahl:", e)

def draw_radar_chart():
    for widget in right_panel.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    # Angenommen: df ist dein DataFrame
    labels = matched_rows_tree_select.columns[-5:].tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Werte für Spieler 1
    values1 = matched_rows_tree_select.iloc[:, -5:].values.flatten().tolist()
    values1 += values1[:1]

    # Werte für Spieler 2
    values2 = matched_rows_matrix_tree_select.iloc[:, -5:].values.flatten().tolist()
    values2 += values2[:1]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    ax.plot(angles, values1, color='blue', linewidth=2, label=matched_rows_tree_select['player_name'])
    line1, = ax.plot(angles, values1, color='blue', linewidth=2)
    ax.fill(angles, values1, color='skyblue', alpha=0.3)

    ax.plot(angles, values2, color='red', linewidth=2, label=matched_rows_matrix_tree_select['player_name'])
    line2, = ax.plot(angles, values2, color='red', linewidth=2)
    ax.fill(angles, values2, color='salmon', alpha=0.3)

    cursor = mplcursors.cursor([line1, line2], hover=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # In Tkinter anzeigen
    global radar_canvas
    radar_canvas = FigureCanvasTkAgg(fig, master=diagram_frame)
    radar_canvas.draw()
    radar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @cursor.connect("add")
    def on_add(sel):
        radius = sel.target[1]  # [0]=angle, [1]=radius
        sel.annotation.set_text(f"{radius:.2f}")


# === GUI ===
root = ctk.CTk()
root.title("Similarity Score")
root.geometry("1000x600")

# === Fenster in 2 Zeilen aufteilen (je 50%) ===
root.grid_rowconfigure(0, weight=1)  # obere Hälfte
root.grid_rowconfigure(1, weight=1)  # untere Hälfte
root.grid_columnconfigure(0, weight=1)

# === Obere Hälfte mit Tabelle ===
top_half = ctk.CTkFrame(root)
top_half.grid(row=0, column=0, sticky="nsew")

top_frame = ctk.CTkFrame(top_half)
top_frame.pack(fill=tk.X, padx=10, pady=5)
label = ctk.CTkLabel(top_frame, text="myData:")

filter_var = tk.StringVar()

filter_entry = ctk.CTkEntry(top_frame, textvariable=filter_var, width=300)
filter_entry.pack(side=tk.RIGHT, padx=(10, 0))

def on_filter_change(*args):
    query = filter_var.get().lower()
    if query == "":
        update_table(mydata)
    else:
        filtered_df = mydata[mydata.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)]
        update_table(filtered_df)

filter_var.trace_add("write", on_filter_change)

reset_button = ctk.CTkButton(top_frame, text="Reset", command=lambda: filter_var.set(""), width=80)
reset_button.pack(side=tk.RIGHT, padx=(5, 0))

label.pack(side=tk.LEFT)

table_frame = ctk.CTkFrame(top_half)
table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Using tkinter Scrollbar and Treeview since CustomTkinter doesn't have direct equivalents
vsb = tk.Scrollbar(table_frame, orient="vertical")
hsb = tk.Scrollbar(table_frame, orient="horizontal")

tree = ttk.Treeview(table_frame, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
vsb.config(command=tree.yview)
hsb.config(command=tree.xview)

tree.grid(row=0, column=0, sticky='nsew')
vsb.grid(row=0, column=1, sticky='ns')
hsb.grid(row=1, column=0, sticky='ew')
table_frame.grid_rowconfigure(0, weight=1)
table_frame.grid_columnconfigure(0, weight=1)

tree.bind("<<TreeviewSelect>>", on_row_select)

# === Untere Hälfte in zwei Spalten aufteilen ===
bottom_half = ctk.CTkFrame(root)
bottom_half.grid(row=1, column=0, sticky="nsew")
bottom_half.grid_rowconfigure(0, weight=1)
bottom_half.grid_columnconfigure(0, weight=1)
bottom_half.grid_columnconfigure(1, weight=5)

# Linke Seite
left_panel = ctk.CTkFrame(bottom_half)
left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

left_label = ctk.CTkLabel(left_panel, text="Similarity Score of selected player")
left_label.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

matrix_filter_var = tk.StringVar()

matrix_filter_frame = ctk.CTkFrame(left_panel)
matrix_filter_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
matrix_filter_frame.grid_columnconfigure(0, weight=1)

matrix_filter_entry = ctk.CTkEntry(matrix_filter_frame, textvariable=matrix_filter_var, width=150)
matrix_filter_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

matrix_reset_button = ctk.CTkButton(matrix_filter_frame, text="Reset", command=lambda: matrix_filter_var.set(""), width=80)
matrix_reset_button.grid(row=0, column=1, sticky="e")

def on_matrix_filter_change(*args):
    query = matrix_filter_var.get().lower()
    index = matched_rows_tree_select.index[0] if not matched_rows_tree_select.empty else None
    if index is None:
        return
    col = sim_score_matrix[index].drop(index)
    filtered = []
    for i, val in col.items():
        name = mydata.loc[i, 'player_name']
        position = mydata.loc[i, 'player_position']
        score = round(val, 4)
        if (query in name.lower()) or (query in position.lower()) or (query in str(score)):
            filtered.append((name, position, score))

    matrix_tree.delete(*matrix_tree.get_children())
    for row in filtered:
        matrix_tree.insert("", "end", values=row)

matrix_filter_var.trace_add("write", on_matrix_filter_change)

matrix_frame = ctk.CTkFrame(left_panel)
matrix_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

left_panel.grid_rowconfigure(0, weight=0)  # Label
left_panel.grid_rowconfigure(1, weight=0)  # Filter input
left_panel.grid_rowconfigure(2, weight=1)  # Matrix view (main content area)
left_panel.grid_columnconfigure(0, weight=1)

matrix_scrollbar = tk.Scrollbar(matrix_frame, orient="vertical")
matrix_tree = ttk.Treeview(matrix_frame, columns=("name", "position", "score"), show="headings", yscrollcommand=matrix_scrollbar.set)
matrix_scrollbar.config(command=matrix_tree.yview)

matrix_tree.heading("name", text="Name", command=lambda: sort_matrix_column("name"))
matrix_tree.heading("position", text="Position", command=lambda: sort_matrix_column("position"))
matrix_tree.heading("score", text="Score", command=lambda: sort_matrix_column("score"))

matrix_tree.grid(row=0, column=0, sticky="nsew")
matrix_scrollbar.grid(row=0, column=1, sticky="ns")

matrix_frame.grid_rowconfigure(0, weight=1)
matrix_frame.grid_columnconfigure(0, weight=1)

matrix_tree.bind("<<TreeviewSelect>>", on_row_select_matrix_tree)

# === Button für Gewichtungs-Array ===
def open_weight_popup():
    global current_weights

    index = matched_rows_tree_select.index[0] if not matched_rows_tree_select.empty else None
    if index is None:
        return

    filtered_data = mydata.iloc[:, 3:]
    filtered_data = filtered_data.iloc[:, :-5]
    labels = filtered_data.columns.tolist()
    num_entries = filtered_data.shape[1]

    # Wenn aktuelle Gewichte existieren, verwende sie. Sonst mit Einsen initialisieren
    if current_weights is not None and len(current_weights) == num_entries:
        weight_array = current_weights
    else:
        weight_array = np.ones((num_entries,))

    popup = ctk.CTkToplevel(root)
    popup.title("Change Weights")
    popup.geometry("600x700")

    container = ctk.CTkFrame(popup)
    container.pack(fill="both", expand=True)

    # Using tkinter Canvas for scrollable frame since CTk doesn't have direct equivalent
    canvas = tk.Canvas(container)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ctk.CTkFrame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    entries = []

    def save_weights():
        nonlocal weight_array
        new_weights = []
        try:
            for entry in entries:
                val = float(entry.get())
                new_weights.append(val)
            weight_array = np.array(new_weights)
            global current_weights
            current_weights = weight_array  # Gewichte merken

            calc_similarity_score(filtered_data, new_weights, True)
            global sim_score_matrix
            sim_score_matrix = pd.read_csv(sim_score_matrix_path, encoding="utf8", delimiter=';', decimal=',', header=None)
            update_matrix_view(matched_rows_tree_select.index[0])
            popup.destroy()
        except ValueError:
            error_label.configure(text="Not a valid number!")

    def reset_weights():
        for entry in entries:
            entry.delete(0, tk.END)
            entry.insert(0, "1.0")

    for i in range(num_entries):
        row = ctk.CTkFrame(scrollable_frame)
        row.pack(fill="x", padx=10, pady=2)
        label = ctk.CTkLabel(row, text=f"{labels[i]}:", width=300, anchor="w")
        label.pack(side="left")
        entry = ctk.CTkEntry(row)
        entry.insert(0, str(weight_array[i]))
        entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
        entries.append(entry)

    error_label = ctk.CTkLabel(scrollable_frame, text="", text_color="red")
    error_label.pack(pady=(10, 0))

    button_frame = ctk.CTkFrame(scrollable_frame)
    button_frame.pack(pady=10)

    save_button = ctk.CTkButton(button_frame, text="Save", command=save_weights)
    save_button.pack(side="left", padx=5)

    reset_button = ctk.CTkButton(button_frame, text="Reset", command=reset_weights)
    reset_button.pack(side="left", padx=5)

# Button under similarity score
weight_button = ctk.CTkButton(left_panel, text="change weights", command=open_weight_popup)
weight_button.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

# Rechte Seite
right_panel = ctk.CTkFrame(bottom_half)
right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
right_panel.grid_propagate(False)  # Verhindert automatische Größenanpassung

diagram_frame = ctk.CTkFrame(right_panel, width=400, height=400)
diagram_frame.pack_propagate(False)  # Inhalt bestimmt nicht die Größe
diagram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# === CSV-Dateien beim Start laden ===
try:
    mydata = pd.read_csv(mydata_path, encoding="utf8", delimiter=';', decimal=',')
    sim_score_matrix = pd.read_csv(sim_score_matrix_path, encoding="utf8", delimiter=';', decimal=',', header=None)
    update_table(mydata)
except Exception as e:
    print("Fehler beim Laden der CSV-Dateien:", e)

root.mainloop()