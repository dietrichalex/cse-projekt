import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import io

# Try to import cairosvg, but make it optional
try:
    import cairosvg

    HAS_CAIROSVG = True
    print("cairosvg imported successfully")
except ImportError:
    HAS_CAIROSVG = False
    print("cairosvg not available - SVG support disabled")

import cursor
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

from create_myData import calc_similarity_score

# === Konfiguration: Dateipfade ===
mydata_path = "data/mydata.csv"
sim_score_matrix_path = "data/similarity_score_matrix.csv"
logo_path = "pictures/Logo_FC_Augsburg.svg"

# === Daten-Variablen ===
mydata = pd.DataFrame()
sim_score_matrix = pd.DataFrame()
sort_state = {}
matrix_sort_state = {}
matched_rows_tree_select = pd.DataFrame()
matched_rows_matrix_tree_select = pd.DataFrame()
radar_canvas = None
bar_canvas = None
current_weights = None

# === Color Scheme ===
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F8F9FA"
ALTERNATE_GRAY = "#E9ECEF"
DARK_GRAY = "#6C757D"
RED_ACCENT = "#DC3545"
GREEN_ACCENT = "#198754"
BLUE_ACCENT = "#0D6EFD"
LIGHT_RED = "#F8D7DA"
LIGHT_GREEN = "#D1E7DD"


def load_svg_as_icon(svg_path, size=(64, 64)):
    """Convert SVG to PhotoImage for window icon with multiple fallback methods"""
    print(f"Attempting to load icon from: {svg_path}")

    # Check if file exists
    if not os.path.exists(svg_path):
        print(f"SVG file not found: {svg_path}")
        return None

    if not HAS_CAIROSVG:
        print("cairosvg not available - cannot load SVG icon")
        return None

    try:
        # Convert SVG to PNG bytes with higher quality
        png_bytes = cairosvg.svg2png(
            url=svg_path,
            output_width=size[0],
            output_height=size[1],
            background_color='white'  # Ensure white background
        )
        print(f"SVG converted to PNG, size: {len(png_bytes)} bytes")

        # Create PIL Image from PNG bytes
        pil_image = Image.open(io.BytesIO(png_bytes))
        print(f"PIL image created: {pil_image.size}, mode: {pil_image.mode}")

        # Ensure the image is in the right format
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        # Convert to PhotoImage
        photo_image = ImageTk.PhotoImage(pil_image)
        print("PhotoImage created successfully")
        return photo_image
    except Exception as e:
        print(f"Error loading SVG icon: {e}")
        return None


def load_svg_as_background(svg_path, size=(400, 400), alpha=1.0):
    """Convert SVG to background image with transparency"""
    print(f"Attempting to load background from: {svg_path}")

    # Check if file exists
    if not os.path.exists(svg_path):
        print(f"SVG file not found: {svg_path}")
        return None

    if not HAS_CAIROSVG:
        print("cairosvg not available - cannot load SVG background")
        return None

    try:
        # Convert SVG to PNG bytes with higher quality
        png_bytes = cairosvg.svg2png(
            url=svg_path,
            output_width=size[0],
            output_height=size[1],
            background_color='white'  # White background
        )
        print(f"Background SVG converted to PNG, size: {len(png_bytes)} bytes")

        # Create PIL Image from PNG bytes
        pil_image = Image.open(io.BytesIO(png_bytes))
        print(f"Background PIL image created: {pil_image.size}, mode: {pil_image.mode}")

        # Add transparency
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        # Apply alpha to the image
        data = pil_image.getdata()
        new_data = []
        for item in data:
            # item is (r, g, b, a)
            if item[3] > 0:  # If not fully transparent
                # Make it very transparent for background
                new_data.append((item[0], item[1], item[2], int(255 * alpha)))
            else:
                new_data.append(item)

        pil_image.putdata(new_data)
        print(f"Alpha applied: {alpha}")

        # Convert to PhotoImage
        photo_image = ImageTk.PhotoImage(pil_image)
        print("Background PhotoImage created successfully")
        return photo_image
    except Exception as e:
        print(f"Error loading SVG background: {e}")
        return None


def try_load_alternative_icon():
    """Try to load icon from alternative formats"""
    alternative_paths = [
        "Logo_FC_Augsburg.png",
        "Logo_FC_Augsburg.ico",
        "pictures/Logo_FC_Augsburg.png",
        "pictures/Logo_FC_Augsburg.ico",
        "images/Logo_FC_Augsburg.png",
        "images/Logo_FC_Augsburg.ico"
    ]

    for path in alternative_paths:
        if os.path.exists(path):
            try:
                print(f"Trying alternative icon: {path}")
                if path.endswith('.png'):
                    return tk.PhotoImage(file=path)
                elif path.endswith('.ico'):
                    # For ICO files, we need to use iconbitmap directly
                    return path  # Return path for iconbitmap
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

    return None


def clear_all_charts():
    global radar_canvas, bar_canvas

    # Close matplotlib figures before clearing
    if radar_canvas is not None:
        plt.close('all')
    if bar_canvas is not None:
        plt.close('all')

    # Clear radar chart
    for widget in radar_frame.winfo_children():
        widget.destroy()

    radar_placeholder = tk.Label(radar_frame, text="Select two players to view radar chart comparison",
                                 fg=DARK_GRAY, font=("Arial", 14), bg=WHITE)
    radar_placeholder.pack(expand=True)

    if hasattr(root, 'background_logo') and root.background_logo:
        radar_logo = tk.Label(radar_frame, image=root.background_logo, bg=WHITE, bd=0)
        radar_logo.place(relx=0.5, rely=0.5, anchor='center')  # CENTER
        radar_logo.lower()
        radar_placeholder.lift()

    # Clear bar chart
    for widget in bar_frame.winfo_children():
        widget.destroy()

    bar_placeholder = tk.Label(bar_frame, text="Select two players to view bar chart comparison",
                               fg=DARK_GRAY, font=("Arial", 14), bg=WHITE)
    bar_placeholder.pack(expand=True)

    if hasattr(root, 'background_logo') and root.background_logo:
        bar_logo = tk.Label(bar_frame, image=root.background_logo, bg=WHITE, bd=0)
        bar_logo.place(relx=0.5, rely=0.5, anchor='center')
        bar_logo.lower()
        bar_placeholder.lift()

    # Clear statistics
    for widget in stats_frame.winfo_children():
        widget.destroy()

    stats_placeholder = tk.Label(stats_frame, text="Select two players to view statistical comparison",
                                 fg=DARK_GRAY, font=("Arial", 14), bg=WHITE)
    stats_placeholder.pack(expand=True)

    if hasattr(root, 'background_logo') and root.background_logo:
        stats_logo = tk.Label(stats_frame, image=root.background_logo, bg=WHITE, bd=0)
        stats_logo.place(relx=0.5, rely=0.5, anchor='center')
        stats_logo.lower()
        stats_placeholder.lift()

    radar_canvas = None
    bar_canvas = None



def clear_matrix_view():
    """Clear the similarity matrix view"""
    matrix_tree.delete(*matrix_tree.get_children())


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
        matrix_tree.insert("", "end",
                           values=(mydata.loc[i, 'player_name'], mydata.loc[i, 'player_position'], round(val, 4)))


def on_row_select(event):
    global matched_rows_tree_select, matched_rows_matrix_tree_select

    selected_item = tree.focus()
    values = tree.item(selected_item, 'values')

    if not values or len(values) < 2:
        return  # Kein valider Eintrag ausgewählt

    try:
        # Spieler anhand der ID suchen
        player_id = int(values[1])  # Sicherstellen, dass das wirklich 'player_id' ist
        matched_rows_tree_select = mydata[mydata['player_id'] == player_id]

        # Clear second player selection when first player changes
        matched_rows_matrix_tree_select = pd.DataFrame()

        # Clear matrix tree selection
        matrix_tree.selection_remove(matrix_tree.selection())

        # Clear all charts since we only have one player selected
        clear_all_charts()

        if not matched_rows_tree_select.empty:
            update_matrix_view(matched_rows_tree_select.index[0])
    except Exception as e:
        print("Fehler bei Auswahl:", e)


def on_row_select_matrix_tree(event):
    global matched_rows_matrix_tree_select

    selected_item = matrix_tree.focus()
    values = matrix_tree.item(selected_item, 'values')

    if not values or len(values) < 2:
        return  # Kein valider Eintrag ausgewählt

    try:
        # Spieler anhand des Namens suchen
        player_name = values[0]  # Sicherstellen, dass das wirklich 'player_name' ist
        matched_rows_matrix_tree_select = mydata[mydata['player_name'] == player_name]

        # Only update charts if we have both players selected
        if not matched_rows_tree_select.empty and not matched_rows_matrix_tree_select.empty:
            # Update the currently selected tab
            current_tab = notebook.index(notebook.select())
            if current_tab == 0:  # Radar Chart
                draw_radar_chart()
            elif current_tab == 1:  # Bar Chart
                draw_bar_chart()
            elif current_tab == 2:  # Statistics
                update_statistics()
    except Exception as e:
        print("Fehler bei Auswahl:", e)


def draw_radar_chart():
    global radar_canvas

    if matched_rows_tree_select.empty or matched_rows_matrix_tree_select.empty:
        return

    # Clear existing content and close any existing figure
    for widget in radar_frame.winfo_children():
        widget.destroy()

    # Close any existing matplotlib figure to free memory
    if radar_canvas is not None:
        radar_canvas.get_tk_widget().destroy()
        plt.close('all')  # Close all figures to prevent memory leak

    radar_canvas = None  # Reset canvas reference

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

    # Plot erstellen with white background and red/green colors
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(angles, values1, color=RED_ACCENT, linewidth=2, label=matched_rows_tree_select['player_name'].iloc[0])
    line1, = ax.plot(angles, values1, color=RED_ACCENT, linewidth=2)
    ax.fill(angles, values1, color=RED_ACCENT, alpha=0.2)

    ax.plot(angles, values2, color=GREEN_ACCENT, linewidth=2,
            label=matched_rows_matrix_tree_select['player_name'].iloc[0])
    line2, = ax.plot(angles, values2, color=GREEN_ACCENT, linewidth=2)
    ax.fill(angles, values2, color=GREEN_ACCENT, alpha=0.2)

    cursor = mplcursors.cursor([line1, line2], hover=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # In Tkinter anzeigen
    radar_canvas = FigureCanvasTkAgg(fig, master=radar_frame)
    radar_canvas.draw()
    radar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @cursor.connect("add")
    def on_add(sel):
        radius = sel.target[1]  # [0]=angle, [1]=radius
        sel.annotation.set_text(f"{radius:.2f}")


def draw_bar_chart():
    global bar_canvas

    if matched_rows_tree_select.empty or matched_rows_matrix_tree_select.empty:
        return

    # Clear existing content and close any existing figure
    for widget in bar_frame.winfo_children():
        widget.destroy()

    # Close any existing matplotlib figure to free memory
    if bar_canvas is not None:
        bar_canvas.get_tk_widget().destroy()
        plt.close('all')  # Close all figures to prevent memory leak

    bar_canvas = None  # Reset canvas reference

    # Get the last 5 columns (stats)
    labels = matched_rows_tree_select.columns[-5:].tolist()
    values1 = matched_rows_tree_select.iloc[:, -5:].values.flatten()
    values2 = matched_rows_matrix_tree_select.iloc[:, -5:].values.flatten()

    # Create bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, values1, width, label=matched_rows_tree_select['player_name'].iloc[0],
                   color=RED_ACCENT, alpha=0.8)
    bars2 = ax.bar(x + width / 2, values2, width, label=matched_rows_matrix_tree_select['player_name'].iloc[0],
                   color=GREEN_ACCENT, alpha=0.8)

    ax.set_xlabel('Statistics')
    ax.set_ylabel('Values')
    ax.set_title('Player Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Display in Tkinter
    bar_canvas = FigureCanvasTkAgg(fig, master=bar_frame)
    bar_canvas.draw()
    bar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def update_statistics():
    # Clear existing content
    for widget in stats_frame.winfo_children():
        widget.destroy()

    if matched_rows_tree_select.empty or matched_rows_matrix_tree_select.empty:
        no_data_label = tk.Label(stats_frame, text="Select two players to compare statistics",
                                 fg=DARK_GRAY, font=("Arial", 14), bg=WHITE)
        no_data_label.pack(expand=True)
        return

    # Create scrollable frame for statistics
    canvas = tk.Canvas(stats_frame, bg=WHITE)
    scrollbar = tk.Scrollbar(stats_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=WHITE)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Player names
    player1_name = matched_rows_tree_select['player_name'].iloc[0]
    player2_name = matched_rows_matrix_tree_select['player_name'].iloc[0]

    # Title
    title_label = tk.Label(scrollable_frame, text="Statistical Comparison",
                           fg=DARK_GRAY, font=("Arial", 16, "bold"))
    title_label.pack(pady=(0, 20))

    # Header frame
    header_frame = tk.Frame(scrollable_frame, bg=LIGHT_GRAY)
    header_frame.pack(fill=tk.X, pady=(0, 10))

    tk.Label(header_frame, text="Statistic", fg=DARK_GRAY, font=("Arial", 12, "bold"), bg=LIGHT_GRAY).grid(
        row=0, column=0, padx=10, pady=5, sticky="w")
    tk.Label(header_frame, text=player1_name, fg=RED_ACCENT, font=("Arial", 12, "bold"), bg=LIGHT_GRAY).grid(
        row=0, column=1, padx=10, pady=5)
    tk.Label(header_frame, text=player2_name, fg=GREEN_ACCENT, font=("Arial", 12, "bold"), bg=LIGHT_GRAY).grid(
        row=0, column=2, padx=10, pady=5)
    tk.Label(header_frame, text="Difference", fg=DARK_GRAY, font=("Arial", 12, "bold"), bg=LIGHT_GRAY).grid(
        row=0, column=3, padx=10, pady=5)

    # Get last 5 columns (stats)
    stats_columns = matched_rows_tree_select.columns[-5:].tolist()

    for i, stat in enumerate(stats_columns):
        value1 = matched_rows_tree_select[stat].iloc[0]
        value2 = matched_rows_matrix_tree_select[stat].iloc[0]
        difference = value1 - value2

        # Alternate row colors
        row_color = WHITE if i % 2 == 0 else ALTERNATE_GRAY
        row_frame = tk.Frame(scrollable_frame, bg=row_color)
        row_frame.pack(fill=tk.X, pady=1)

        tk.Label(row_frame, text=stat, fg=DARK_GRAY, font=("Arial", 11), bg=row_color).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")
        tk.Label(row_frame, text=f"{value1:.2f}", fg=RED_ACCENT, font=("Arial", 11), bg=row_color).grid(
            row=0, column=1, padx=10, pady=5)
        tk.Label(row_frame, text=f"{value2:.2f}", fg=GREEN_ACCENT, font=("Arial", 11), bg=row_color).grid(
            row=0, column=2, padx=10, pady=5)

        # Color code the difference
        diff_color = GREEN_ACCENT if difference > 0 else RED_ACCENT if difference < 0 else DARK_GRAY
        diff_text = f"+{difference:.2f}" if difference > 0 else f"{difference:.2f}"
        tk.Label(row_frame, text=diff_text, fg=diff_color, font=("Arial", 11), bg=row_color).grid(
            row=0, column=3, padx=10, pady=5)


def on_tab_change(event):
    """Handle tab change events"""
    if matched_rows_tree_select.empty or matched_rows_matrix_tree_select.empty:
        return

    current_tab = notebook.index(notebook.select())
    if current_tab == 0:  # Radar Chart
        draw_radar_chart()
    elif current_tab == 1:  # Bar Chart
        draw_bar_chart()
    elif current_tab == 2:  # Statistics
        update_statistics()


def reset_all_selections():
    """Reset all selections and clear all views"""
    global matched_rows_tree_select, matched_rows_matrix_tree_select

    # Clear dataframes
    matched_rows_tree_select = pd.DataFrame()
    matched_rows_matrix_tree_select = pd.DataFrame()

    # Clear tree selections
    tree.selection_remove(tree.selection())
    matrix_tree.selection_remove(matrix_tree.selection())

    # Clear matrix view
    clear_matrix_view()

    # Clear all charts
    clear_all_charts()

    # Reset filter
    filter_var.set("")


# === GUI ===
root = tk.Tk()
root.title("Similarity Score - FC Augsburg")
root.geometry("1200x700")
root.configure(bg=WHITE)

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for logo at: {os.path.abspath(logo_path)}")

# Create main container frame that will hold the background
main_container = tk.Frame(root, bg=WHITE)
main_container.pack(fill=tk.BOTH, expand=True)

# Load and set window icon - comprehensive approach
icon_loaded = False
try:
    # First try: SVG icon
    icon_image = load_svg_as_icon(logo_path, size=(64, 64))
    if icon_image is not None:
        root.iconphoto(True, icon_image)
        root.icon_image = icon_image  # Keep reference
        print("SVG icon loaded and set successfully")
        icon_loaded = True
    else:
        print("SVG icon loading failed, trying alternatives...")

        # Try alternative formats
        alt_icon = try_load_alternative_icon()
        if alt_icon is not None:
            if isinstance(alt_icon, str):  # ICO file path
                root.iconbitmap(alt_icon)
                print(f"ICO icon loaded: {alt_icon}")
            else:  # PhotoImage
                root.iconphoto(True, alt_icon)
                root.alt_icon = alt_icon  # Keep reference
                print("Alternative icon loaded successfully")
            icon_loaded = True

except Exception as e:
    print(f"Error setting window icon: {e}")

if not icon_loaded:
    print("Warning: No icon could be loaded")

# Load logo for use in placeholder background
try:
    root.background_logo = load_svg_as_background(logo_path, size=(400, 400), alpha=0.5)
    print("Logo for placeholders loaded successfully")
except Exception as e:
    print("Could not load logo for placeholders:", e)
    root.background_logo = None

# === Configure grid weights ===
main_container.grid_rowconfigure(0, weight=1)  # obere Hälfte
main_container.grid_rowconfigure(1, weight=1)  # untere Hälfte
main_container.grid_columnconfigure(0, weight=1)

# === Obere Hälfte mit Tabelle ===
top_half = tk.Frame(main_container, relief=tk.FLAT, bd=0, highlightthickness=0)
top_half.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

top_frame = tk.Frame(top_half, bg=LIGHT_GRAY)
top_frame.pack(fill=tk.X, padx=10, pady=5)

label = tk.Label(top_frame, text="Player Data:", fg=DARK_GRAY, font=("Arial", 14, "bold"), bg=LIGHT_GRAY)
label.pack(side=tk.LEFT)

filter_var = tk.StringVar()

# Filter label on the left side
filter_label = tk.Label(top_frame, text="Filter:", fg=DARK_GRAY, font=("Arial", 12, "bold"), bg=LIGHT_GRAY)
filter_label.pack(side=tk.LEFT, padx=(20, 5))

filter_entry = tk.Entry(top_frame, textvariable=filter_var, width=30, bg=WHITE, fg="black", bd=2, relief=tk.SOLID)
filter_entry.pack(side=tk.LEFT, padx=(0, 5))

# Reset button
reset_button = tk.Button(top_frame, text="Reset", command=reset_all_selections, width=10,
                         bg=RED_ACCENT, fg=WHITE, activebackground="#B02A30", bd=0)
reset_button.pack(side=tk.LEFT, padx=(5, 0))


def on_filter_change(*args):
    query = filter_var.get().lower()
    if query == "":
        update_table(mydata)
    else:
        filtered_df = mydata[mydata.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)]
        update_table(filtered_df)


filter_var.trace_add("write", on_filter_change)

# Table frame
table_frame = tk.Frame(top_half, bg=WHITE)
table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Scrollbars for table
vsb = tk.Scrollbar(table_frame, orient="vertical")
hsb = tk.Scrollbar(table_frame, orient="horizontal")

# Configure treeview style
style = ttk.Style()
style.theme_use('clam')
style.configure("Treeview", background=WHITE, foreground=DARK_GRAY, fieldbackground=WHITE)
style.configure("Treeview.Heading", background=LIGHT_GRAY, foreground=DARK_GRAY, font=("Arial", 10, "bold"))
style.map("Treeview", background=[('selected', LIGHT_GREEN)], foreground=[('selected', 'black')])

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
bottom_half = tk.Frame(main_container, highlightthickness=0)
bottom_half.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
bottom_half.grid_rowconfigure(0, weight=1)
bottom_half.grid_columnconfigure(0, weight=1)
bottom_half.grid_columnconfigure(1, weight=5)

# === Linke Seite ===
left_panel = tk.Frame(bottom_half, relief=tk.FLAT, bd=0, highlightthickness=1, highlightcolor=LIGHT_GRAY,
                      highlightbackground=LIGHT_GRAY)
left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

left_label = tk.Label(left_panel, text="Similarity Score of selected player",
                      fg=DARK_GRAY, font=("Arial", 12, "bold"))
left_label.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

# Matrix filter
matrix_filter_var = tk.StringVar()

matrix_filter_frame = tk.Frame(left_panel, bg=LIGHT_GRAY)
matrix_filter_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

matrix_filter_frame.grid_columnconfigure(1, weight=1)

# Filter label for matrix
matrix_filter_label = tk.Label(matrix_filter_frame, text="Filter:", fg=DARK_GRAY,
                               font=("Arial", 11, "bold"), bg=LIGHT_GRAY)
matrix_filter_label.grid(row=0, column=0, sticky="w", padx=(5, 5))

matrix_filter_entry = tk.Entry(matrix_filter_frame, textvariable=matrix_filter_var, width=20,
                               bg=WHITE, fg="black", bd=2, relief=tk.SOLID)
matrix_filter_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

matrix_reset_button = tk.Button(matrix_filter_frame, text="Reset", command=lambda: matrix_filter_var.set(""),
                                width=8, bg=RED_ACCENT, fg=WHITE, activebackground="#B02A30", bd=0)
matrix_reset_button.grid(row=0, column=2, sticky="e", padx=(5, 5))


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

# Matrix tree view
matrix_frame = tk.Frame(left_panel, bg=WHITE)
matrix_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

left_panel.grid_rowconfigure(0, weight=0)  # Label
left_panel.grid_rowconfigure(1, weight=0)  # Filter input
left_panel.grid_rowconfigure(2, weight=1)  # Matrix view (main content area)
left_panel.grid_columnconfigure(0, weight=1)

matrix_scrollbar = tk.Scrollbar(matrix_frame, orient="vertical")
matrix_tree = ttk.Treeview(matrix_frame, columns=("name", "position", "score"), show="headings",
                           yscrollcommand=matrix_scrollbar.set)
matrix_scrollbar.config(command=matrix_tree.yview)

matrix_tree.heading("name", text="Name", command=lambda: sort_matrix_column("name"))
matrix_tree.heading("position", text="Position", command=lambda: sort_matrix_column("position"))
matrix_tree.heading("score", text="Score", command=lambda: sort_matrix_column("score"))

matrix_tree.grid(row=0, column=0, sticky="nsew")
matrix_scrollbar.grid(row=0, column=1, sticky="ns")

matrix_frame.grid_rowconfigure(0, weight=1)
matrix_frame.grid_columnconfigure(0, weight=1)

# Apply same style to matrix tree for consistency
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

    popup = tk.Toplevel(root)
    popup.title("Change Weights")
    popup.geometry("600x700")
    popup.configure(bg=WHITE)

    # Fix window focus and visibility issues
    popup.transient(root)  # Make popup a transient window
    popup.grab_set()  # Make popup modal
    popup.focus_set()  # Set focus to popup
    popup.lift()  # Bring popup to front
    popup.attributes('-topmost', True)  # Keep popup on top temporarily
    popup.after(100, lambda: popup.attributes('-topmost', False))  # Remove topmost after 100ms

    container = tk.Frame(popup, bg=WHITE)
    container.pack(fill="both", expand=True)

    # Using tkinter Canvas for scrollable frame
    canvas = tk.Canvas(container, bg=WHITE, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=WHITE)

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
            sim_score_matrix = pd.read_csv(sim_score_matrix_path, encoding="utf8", delimiter=';', decimal=',',
                                           header=None)
            update_matrix_view(matched_rows_tree_select.index[0])
            popup.destroy()
        except ValueError:
            error_label.configure(text="Not a valid number!")

    def reset_weights():
        for entry in entries:
            entry.delete(0, tk.END)
            entry.insert(0, "1.0")

    for i in range(num_entries):
        # Alternate row colors: white for even rows, darker gray for odd rows
        row_color = WHITE if i % 2 == 0 else ALTERNATE_GRAY
        row = tk.Frame(scrollable_frame, bg=row_color)
        row.pack(fill="x", padx=10, pady=2)

        label = tk.Label(row, text=f"{labels[i]}:", width=30, anchor="w", fg=DARK_GRAY, bg=row_color)
        label.pack(side="left")

        entry = tk.Entry(row, bg=WHITE, fg="black", bd=2, relief=tk.SOLID)
        entry.insert(0, str(weight_array[i]))
        entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
        entries.append(entry)

    error_label = tk.Label(scrollable_frame, text="", fg=RED_ACCENT, bg=WHITE)
    error_label.pack(pady=(10, 0))

    button_frame = tk.Frame(scrollable_frame, bg=WHITE)
    button_frame.pack(pady=10)

    save_button = tk.Button(button_frame, text="Save", command=save_weights,
                            bg=GREEN_ACCENT, fg=WHITE, activebackground="#146C43", bd=0, padx=20)
    save_button.pack(side="left", padx=5)

    reset_button = tk.Button(button_frame, text="Reset", command=reset_weights,
                             bg=RED_ACCENT, fg=WHITE, activebackground="#B02A30", bd=0, padx=20)
    reset_button.pack(side="left", padx=5)


# Button under similarity score
weight_button = tk.Button(left_panel, text="change weights", command=open_weight_popup,
                          bg=GREEN_ACCENT, fg=WHITE, activebackground="#146C43", bd=0)
weight_button.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

# === Rechte Seite mit Tabs ===
right_panel = tk.Frame(bottom_half, relief=tk.FLAT, bd=0, highlightthickness=1, highlightcolor=LIGHT_GRAY,
                       highlightbackground=LIGHT_GRAY)
right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

# Create notebook (tabbed view)
notebook = ttk.Notebook(right_panel)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Create tab frames with transparent appearance
radar_frame = tk.Frame(notebook)
bar_frame = tk.Frame(notebook)
stats_frame = tk.Frame(notebook)

# Configure the frames to be transparent
radar_frame.configure(highlightthickness=0, bd=0)
bar_frame.configure(highlightthickness=0, bd=0)
stats_frame.configure(highlightthickness=0, bd=0)

# Add tabs to notebook
notebook.add(radar_frame, text="Radar Chart")
notebook.add(bar_frame, text="Bar Chart")
notebook.add(stats_frame, text="Statistics")

# Bind tab change event
notebook.bind("<<NotebookTabChanged>>", on_tab_change)

# Add initial placeholder labels
radar_placeholder = tk.Label(radar_frame, text="Select two players to view radar chart comparison",
                             fg=DARK_GRAY, font=("Arial", 14), highlightthickness=0, bd=0)
radar_placeholder.pack(expand=True)

bar_placeholder = tk.Label(bar_frame, text="Select two players to view bar chart comparison",
                           fg=DARK_GRAY, font=("Arial", 14), highlightthickness=0, bd=0)
bar_placeholder.pack(expand=True)

stats_placeholder = tk.Label(stats_frame, text="Select two players to view statistical comparison",
                             fg=DARK_GRAY, font=("Arial", 14), highlightthickness=0, bd=0)
stats_placeholder.pack(expand=True)

# Add background logos to tab frames if available
if hasattr(root, 'background_logo') and root.background_logo:
    radar_bg_label = tk.Label(radar_frame, image=root.background_logo, bg=WHITE)
    radar_bg_label.place(relx=0.5, rely=0.5, anchor='center')
    radar_placeholder.lift()  # Bring text to front

    bar_bg_label = tk.Label(bar_frame, image=root.background_logo, bg=WHITE)
    bar_bg_label.place(relx=0.5, rely=0.5, anchor='center')
    bar_placeholder.lift()  # Bring text to front

    stats_bg_label = tk.Label(stats_frame, image=root.background_logo, bg=WHITE)
    stats_bg_label.place(relx=0.5, rely=0.5, anchor='center')
    stats_placeholder.lift()  # Bring text to front

# === CSV-Dateien beim Start laden ===
try:
    mydata = pd.read_csv(mydata_path, encoding="utf8", delimiter=';', decimal=',')
    sim_score_matrix = pd.read_csv(sim_score_matrix_path, encoding="utf8", delimiter=';', decimal=',', header=None)
    update_table(mydata)
except Exception as e:
    print("Fehler beim Laden der CSV-Dateien:", e)

root.mainloop()