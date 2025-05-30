# src/fem_app/ui/main_window.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys
import traceback # Для детального виводу помилок

# --- Імпорти з пакету fem_app ---
try:
    from fem_app.core import mesh, derivatives, stiffness, boundary, solver
    from fem_app.ui import visualization, array_viewer
    # from fem_app.utils import helpers # Якщо знадобиться
except ModuleNotFoundError as e:
     print(f"Помилка імпорту в main_window.py: {e}. Перевірте структуру проекту та PYTHONPATH.")
     try:
         root_err = tk.Tk()
         root_err.withdraw()
         messagebox.showerror("Помилка запуску", f"Не вдалося завантажити компоненти програми: {e}")
         root_err.destroy()
     except tk.TclError:
         pass
     sys.exit(1)


class FEMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("МСЕ Аналіз Паралелепіпеда v3.2 (з деформацією)")
        self.geometry("1200x800")
        self._init_state()
        self._create_ui()

    def _init_state(self):
        """Ініціалізує змінні стану програми."""
        self.nx_var = tk.IntVar(value=1)
        self.ny_var = tk.IntVar(value=1)
        self.nz_var = tk.IntVar(value=1)
        self.ax_var = tk.DoubleVar(value=1.0)
        self.ay_var = tk.DoubleVar(value=1.0)
        self.az_var = tk.DoubleVar(value=1.0)
        self.E_var = tk.DoubleVar(value=1.0)
        self.nu_var = tk.DoubleVar(value=0.3)
        self.zu_text = tk.StringVar(value="")
        self.zp_text = tk.StringVar(value="")
        self.p_val_for_zp_default = 0.5 # Змінено знак для стискаючого тиску

        self.AKT, self.NT, self.node_map, self.nqp, self.nel = None, None, None, 0, 0
        self.deformed_AKT = None # Для деформованих координат
        self.ZU, self.ZP = None, None
        self.DFIABG, self.MG, self.F, self.U = None, None, None, None
        self.SIGMA, self.SIGMA_P = None, None
        self.DJ, self.DFIXYZ, self.MGE, self.FE = {}, {}, {}, {}

        self.alpha_g_flat = derivatives.alpha_flat_g
        self.beta_g_flat = derivatives.beta_flat_g
        self.gamma_g_flat = derivatives.gamma_flat_g
        self.gauss_point_coords_list = list(zip(self.alpha_g_flat, self.beta_g_flat, self.gamma_g_flat))
        self.gauss_point_display_list = [f"GP {i+1}: ({a:.6f}, {b:.6f}, {g:.6f})"
                                         for i, (a,b,g) in enumerate(self.gauss_point_coords_list)]
        self.selected_gauss_point_display_str = tk.StringVar()
        if self.gauss_point_display_list:
            self.selected_gauss_point_display_str.set(self.gauss_point_display_list[0])
        self.selected_element_for_dfixyz_idx = tk.IntVar(value=1)

        # Змінні для керування візуалізацією деформації
        self.show_deformation_var = tk.BooleanVar(value=False)
        self.deformation_scale_var = tk.DoubleVar(value=1.0) # Початковий масштаб 1.0

        self.after(50, self._initial_setup_from_image)

    def _initial_setup_from_image(self):
        print("Виконується початкове налаштування...")
        self._generate_grid_callback(show_confirmation=False)
        if self.AKT is not None and self.nel > 0:
            print("Встановлення ГУ за замовчуванням...")
            self._set_default_zu()
            self._set_default_zp_from_image()
            self._apply_bcs_callback(show_info=False)
            print("ГУ за замовчуванням встановлено та застосовано.")
        else:
            messagebox.showwarning("Початкове налаштування", "Не вдалося згенерувати сітку за замовчуванням.")
            print("Помилка початкового налаштування: сітка не була згенерована належним чином.")

    def _create_ui(self):
        # ... (код _create_setup_panel, _create_bc_panel, _create_solve_panel без змін) ...
        settings_frame = ttk.Frame(self, padding="10", width=380)
        settings_frame.pack(side="left", fill="y", padx=(0, 5), anchor='nw')
        settings_frame.pack_propagate(False)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="right", expand=True, fill="both", padx=(5,0))

        self._create_setup_panel(settings_frame)
        self._create_bc_panel(settings_frame)
        self._create_solve_panel(settings_frame)

        self.tab_vis = ttk.Frame(self.notebook, padding="5")
        self.tab_arrays = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_vis, text="Візуалізація")
        self.notebook.add(self.tab_arrays, text="Масиви")
        self._create_vis_tab() # Буде змінено
        self._create_arrays_tab()


    def _create_setup_panel(self, parent_frame):
        # ... (код без змін) ...
        frame = ttk.LabelFrame(parent_frame, text="Параметри сітки та матеріалу", padding="10")
        frame.pack(fill="x", pady=(0, 10), expand=False)
        labels_params = ["Nx:", "Ny:", "Nz:", "Ax:", "Ay:", "Az:", "E:", "nu:"]
        vars_params = [self.nx_var, self.ny_var, self.nz_var, self.ax_var, self.ay_var, self.az_var, self.E_var, self.nu_var]
        for i, (label, var) in enumerate(zip(labels_params, vars_params)):
             r, c_pair = divmod(i, 4)
             col_label = c_pair * 2
             col_entry = c_pair * 2 + 1
             ttk.Label(frame, text=label).grid(row=r, column=col_label, sticky='w', padx=(0,2), pady=2)
             ttk.Entry(frame, textvariable=var, width=7).grid(row=r, column=col_entry, sticky='ew', padx=(0,5), pady=2)
        for col_idx in [1, 3, 5, 7]:
             frame.columnconfigure(col_idx, weight=1)
        btn_generate = ttk.Button(frame, text="Згенерувати / Оновити сітку", command=self._generate_grid_callback)
        btn_generate.grid(row=2, column=0, columnspan=8, pady=(10,0), sticky='ew')

    def _create_bc_panel(self, parent_frame):
        # ... (код без змін) ...
        frame = ttk.LabelFrame(parent_frame, text="Граничні умови", padding="10")
        frame.pack(fill="x", pady=(0, 10))
        ttk.Label(frame, text="Закріплені вузли ZU (номери через кому):").grid(row=0, column=0, columnspan=2, sticky='w')
        self.zu_entry = ttk.Entry(frame, textvariable=self.zu_text); self.zu_entry.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        
        btn_def_zu = ttk.Button(frame, text="Низ", width=5, command=self._set_default_zu); btn_def_zu.grid(row=2, column=0, sticky='w')
        btn_clr_zu = ttk.Button(frame, text="Очист.", width=6, command=lambda: self.zu_text.set("")); btn_clr_zu.grid(row=2, column=1, sticky='w', padx=(5,0))
        
        ttk.Label(frame, text="Навантаження ZP (Елем, Грань, Тиск):").grid(row=3, column=0, columnspan=2, sticky='w', pady=(10, 0))
        self.zp_entry = scrolledtext.ScrolledText(frame, height=3, width=30); self.zp_entry.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        self.zp_entry.bind("<FocusOut>", lambda e: self.zp_text.set(self.zp_entry.get("1.0", tk.END).strip()))
        
        btn_def_zp = ttk.Button(frame, text="Верх (P)", width=8, command=self._set_default_zp_from_image); btn_def_zp.grid(row=5, column=0, sticky='w')
        btn_clr_zp = ttk.Button(frame, text="Очист.", width=6, command=lambda: (self.zp_entry.delete("1.0", tk.END), self.zp_text.set(""))); btn_clr_zp.grid(row=5, column=1, sticky='w', padx=(5,0))
        
        btn_apply_bc = ttk.Button(frame, text="Застосувати введені ГУ", command=self._apply_bcs_callback); btn_apply_bc.grid(row=6, column=0, columnspan=2, pady=(10,0), sticky='ew')
        frame.columnconfigure(0, weight=1); frame.columnconfigure(1, weight=1)

    def _create_solve_panel(self, parent_frame):
         # ... (код без змін) ...
         frame = ttk.LabelFrame(parent_frame, text="Розрахунок", padding="10"); frame.pack(fill="x", pady=(0, 10))
         btn_solve = ttk.Button(frame, text="Зібрати систему та Розв'язати", command=self._solve_callback); btn_solve.pack(pady=(5, 10), fill='x')
         ttk.Label(frame, text="Результати:").pack(anchor='w')
         self.results_label = ttk.Label(frame, text="Розрахунок не виконано.", justify=tk.LEFT, wraplength=340); self.results_label.pack(anchor='w', pady=(0, 5), fill='x')


    def _create_vis_tab(self):
        frame = self.tab_vis
        
        # Фрейм для опцій візуалізації
        options_vis_frame = ttk.Frame(frame)
        options_vis_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))

        self.show_nodes_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.show_outline_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_vis_frame, text="Вузли", variable=self.show_nodes_var, command=self._update_visualization).pack(side='left', padx=5)
        ttk.Checkbutton(options_vis_frame, text="Номери", variable=self.show_labels_var, command=self._update_visualization).pack(side='left', padx=5)
        ttk.Checkbutton(options_vis_frame, text="Контур", variable=self.show_outline_var, command=self._update_visualization).pack(side='left', padx=5)
        
        # Додаємо елементи для керування відображенням деформації
        ttk.Checkbutton(options_vis_frame, text="Показати деформацію", variable=self.show_deformation_var, command=self._update_visualization).pack(side='left', padx=10)
        
        ttk.Label(options_vis_frame, text="Масштаб деф.:").pack(side='left', padx=(5,0))
        self.deformation_scale_spinbox = tk.Spinbox(options_vis_frame, from_=0.1, to=1000.0, increment=0.1, textvariable=self.deformation_scale_var, width=7, command=self._update_visualization)
        self.deformation_scale_spinbox.pack(side='left', padx=5)

        # Графік Matplotlib
        self.vis_fig = plt.figure(figsize=(6, 5))
        self.vis_ax = self.vis_fig.add_subplot(111, projection='3d')
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=frame)
        canvas_widget = self.vis_canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.vis_canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_arrays_tab(self):
        # ... (код залишається без змін) ...
        frame = self.tab_arrays
        select_frame = ttk.Frame(frame)
        select_frame.pack(pady=5, fill='x')
        ttk.Label(select_frame, text="Вибрати масив:").pack(side='left', padx=(0, 5))
        self.array_names = ["AKT", "NT"] 
        self.selected_array_var = tk.StringVar(value=self.array_names[0] if self.array_names else "")
        self.array_combo = ttk.Combobox(select_frame, textvariable=self.selected_array_var, values=self.array_names, width=30, state="readonly")
        self.array_combo.pack(side='left', padx=(0, 10))
        self.array_combo.bind("<<ComboboxSelected>>", self._on_array_selected)

        self.params_frame = ttk.Frame(frame)
        ttk.Label(self.params_frame, text="Точка Гаусса:").pack(side='left', padx=(10, 2))
        self.gauss_point_combo = ttk.Combobox(self.params_frame, 
                                              textvariable=self.selected_gauss_point_display_str, 
                                              values=self.gauss_point_display_list, 
                                              width=30, state="readonly")
        self.gauss_point_combo.pack(side='left', padx=(0, 10))
        if self.gauss_point_display_list: self.gauss_point_combo.current(0)

        self.element_label = ttk.Label(self.params_frame, text="Елемент (1-Nel):")
        self.element_spinbox = tk.Spinbox(self.params_frame, from_=1, to=1, textvariable=self.selected_element_for_dfixyz_idx, width=7)
        
        action_frame = ttk.Frame(frame)
        action_frame.pack(pady=5, fill='x')
        ttk.Button(action_frame, text="Показати масив", command=self._show_array_callback).pack(side='left')
        
        self._update_array_params_visibility()

    def _on_array_selected(self, event=None): 
        self._update_array_params_visibility()

    def _update_array_params_visibility(self):
        # ... (код залишається без змін) ...
        selected_name = self.selected_array_var.get()
        is_dfi_type = selected_name.startswith("DFIABG") or selected_name.startswith("DFIXYZ")
        is_mge_or_fe_type = selected_name.startswith("MGE (elem") or selected_name.startswith("FE (elem")

        # Спочатку ховаємо всі опціональні елементи
        # Перевіряємо, чи існують віджети перед pack_forget
        if hasattr(self, 'gauss_point_combo') and self.gauss_point_combo.winfo_exists():
            if self.gauss_point_combo.master.winfo_exists() and self.gauss_point_combo.master.winfo_children():
                 if self.gauss_point_combo.master.winfo_children()[0].winfo_exists():
                    self.gauss_point_combo.master.winfo_children()[0].pack_forget() # Label
            self.gauss_point_combo.pack_forget()
        
        if hasattr(self, 'element_label') and self.element_label.winfo_exists():
            self.element_label.pack_forget()
        if hasattr(self, 'element_spinbox') and self.element_spinbox.winfo_exists():
            self.element_spinbox.pack_forget()
        if hasattr(self, 'params_frame') and self.params_frame.winfo_exists():
            self.params_frame.pack_forget()


        if is_dfi_type or is_mge_or_fe_type:
            if hasattr(self, 'params_frame'): # Переконуємось, що фрейм існує
                self.params_frame.pack(pady=(0, 5), fill='x', after=self.array_combo.master)
            
            if selected_name.startswith("DFIABG") or selected_name.startswith("DFIXYZ"):
                if hasattr(self, 'gauss_point_combo') and self.gauss_point_combo.winfo_exists():
                     if self.gauss_point_combo.master.winfo_exists() and self.gauss_point_combo.master.winfo_children():
                        if self.gauss_point_combo.master.winfo_children()[0].winfo_exists():
                            self.gauss_point_combo.master.winfo_children()[0].pack(side='left', padx=(10, 2)) # Label
                     self.gauss_point_combo.pack(side='left', padx=(0, 10))
            
            if selected_name.startswith("DFIXYZ") or selected_name.startswith("MGE (elem") or selected_name.startswith("FE (elem"):
                if hasattr(self, 'element_label') and self.element_label.winfo_exists():
                    self.element_label.pack(side='left', padx=(10, 2))
                if hasattr(self, 'element_spinbox') and self.element_spinbox.winfo_exists():
                    self.element_spinbox.pack(side='left', padx=(0, 10))
                    max_elem = self.nel if self.nel > 0 else 1
                    self.element_spinbox.config(to=max_elem)
                    current_elem_val = self.selected_element_for_dfixyz_idx.get()
                    if not (1 <= current_elem_val <= max_elem):
                        self.selected_element_for_dfixyz_idx.set(min(max(1, current_elem_val), max_elem))


    def _generate_grid_callback(self, show_confirmation=True):
        # ... (код майже без змін, але додаємо скидання deformed_AKT) ...
        try:
            nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
            ax_, ay_, az_ = self.ax_var.get(), self.ay_var.get(), self.az_var.get()
            if not (nx>0 and ny>0 and nz>0 and ax_>0 and ay_>0 and az_>0):
                raise ValueError("Nx, Ny, Nz, Ax, Ay, Az мають бути позитивними.")

            self.AKT, self.node_map, self.nqp = mesh.generate_mesh(nx, ny, nz, ax_, ay_, az_)
            self.NT, self.nel = mesh.generate_connectivity(nx, ny, nz, self.node_map)
            
            self.deformed_AKT = None # Скидаємо деформовані координати при новій генерації
            self.DFIABG, self.MG, self.F, self.U, self.SIGMA, self.SIGMA_P = None, None, None, None, None, None
            self.DJ, self.DFIXYZ, self.MGE, self.FE = {}, {}, {}, {}
            self.ZU, self.ZP = None, None
            
            self.results_label.config(text="Нову сітку згенеровано. Розрахунок не виконано.")
            self._update_array_combo_list() 
            self._update_visualization() # Покаже оригінальну сітку
            
            if show_confirmation:
                self._set_default_zu() 
                self._set_default_zp_from_image() 
                self._apply_bcs_callback(show_info=False)
                messagebox.showinfo("Сітка", f"Згенеровано: {self.nqp} вузлів, {self.nel} елементів.")
            
        except ValueError as e: messagebox.showerror("Помилка введення", f"{e}")
        except Exception as e: messagebox.showerror("Помилка генерації", f"{e}", detail=traceback.format_exc())


    def _update_visualization(self): # Тепер не приймає аргумент show_deformed напряму
        try:
            ax_dimensions = (self.ax_var.get(), self.ay_var.get(), self.az_var.get())
            
            akt_to_show_deformed = None
            scale = 1.0
            if self.show_deformation_var.get() and self.deformed_AKT is not None and self.U is not None:
                akt_to_show_deformed = self.deformed_AKT
                scale = self.deformation_scale_var.get()
            
            visualization.draw_mesh(self.vis_ax, self.AKT, self.NT,
                                    ax_dimensions,
                                    show_nodes=self.show_nodes_var.get(),
                                    show_node_labels=self.show_labels_var.get(),
                                    show_outline=self.show_outline_var.get(),
                                    AKT_deformed=akt_to_show_deformed, # Передаємо деформовані координати
                                    deformation_scale=scale, # Передаємо масштаб
                                    show_orig_wireframe=True if akt_to_show_deformed is not None else False
                                    )
            self.vis_canvas.draw_idle()
        except Exception as e: 
            print(f"Помилка візуалізації: {e}")
            traceback.print_exc()

    def _set_default_zu(self):
        # ... (код без змін) ...
        if self.AKT is None or self.nqp == 0:
            print("Неможливо встановити ZU: сітку не згенеровано.")
            self.zu_text.set("")
            return
        try:
            min_z = np.min(self.AKT[2, :])
            bottom_nodes_indices = np.where(np.abs(self.AKT[2, :] - min_z) < 1e-9)[0]
            if bottom_nodes_indices.size > 0:
                self.zu_text.set(", ".join([str(i + 1) for i in bottom_nodes_indices]))
                print(f"Встановлено ZU за замовчуванням: {self.zu_text.get()}")
            else:
                self.zu_text.set("")
                print("Не знайдено вузлів для ZU за замовчуванням.")
        except Exception as e:
            print(f"Помилка автоматичного встановлення ZU: {e}")
            self.zu_text.set("")

    def _set_default_zp_from_image(self):
        # ... (код без змін, але використовуємо self.p_val_for_zp_default) ...
        if self.nel == 0:
            print("Неможливо встановити ZP: сітку не згенеровано або немає елементів.")
            self.zp_entry.delete("1.0", tk.END)
            self.zp_text.set("")
            return

        zp_lines = []
        max_z_coord = self.az_var.get() 

        for elem_id in range(1, self.nel + 1):
            elem_nodes_global_0based = self.NT[:, elem_id - 1] - 1
            valid_nodes_mask = elem_nodes_global_0based >= 0
            if not np.all(valid_nodes_mask): continue 
            try:
                akt_elem_nodes = self.AKT[:, elem_nodes_global_0based]
            except IndexError:
                continue 

            upper_face_local_indices = [4, 5, 6, 7, 16, 17, 18, 19] # Локальні індекси 3D вузлів
            is_top_element_face = False
            for loc_idx in upper_face_local_indices:
                node_global_idx = elem_nodes_global_0based[loc_idx]
                if node_global_idx < self.nqp and np.abs(self.AKT[2, node_global_idx] - max_z_coord) < 1e-9:
                    is_top_element_face = True
                    break
            if is_top_element_face:
                # Припускаємо, що тиск стискаючий, тому передаємо від'ємне значення
                zp_lines.append(f"{elem_id}, 6, {-self.p_val_for_zp_default}") 
        if zp_lines:
            zp_default_str = "\n".join(zp_lines)
            self.zp_entry.delete("1.0", tk.END)
            self.zp_entry.insert("1.0", zp_default_str)
            self.zp_text.set(zp_default_str) 
            print(f"Встановлено ZP за замовчуванням: {zp_default_str}")
        else:
            self.zp_entry.delete("1.0", tk.END)
            self.zp_text.set("")
            print("Не знайдено верхніх граней для ZP за замовчуванням (або nel=0).")


    def _apply_bcs_callback(self, show_info=True):
        # ... (код без змін) ...
        try:
            zu_str = self.zu_text.get().strip()
            if zu_str:
                self.ZU = np.array([int(n.strip()) for n in zu_str.split(',') if n.strip()], dtype=int)
                if self.nqp > 0 and (np.any(self.ZU <= 0) or np.any(self.ZU > self.nqp)):
                    raise ValueError(f"Некоректні номери вузлів в ZU (мають бути від 1 до {self.nqp}).")
            else:
                self.ZU = np.array([], dtype=int)
            
            zp_str_from_widget = self.zp_entry.get("1.0", tk.END).strip()
            self.zp_text.set(zp_str_from_widget)
            
            zp_list_parsed = []
            if zp_str_from_widget:
                for i, line in enumerate(zp_str_from_widget.split('\n')):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                    if len(parts) != 3:
                        raise ValueError(f"Неправильний формат ZP в рядку {i+1}. Очікується 'Елем, Грань, Тиск'.")
                    try:
                        elem_id, face_id, load_val = int(parts[0]), int(parts[1]), float(parts[2])
                    except ValueError:
                        raise ValueError(f"Неправильний числовий формат в ZP, рядок {i+1}.")
                    if self.nel > 0 and (elem_id <= 0 or elem_id > self.nel):
                        raise ValueError(f"Некоректний номер елемента {elem_id} в ZP (має бути від 1 до {self.nel}).")
                    if not (1 <= face_id <= 6):
                        raise ValueError(f"Некоректний номер грані {face_id} в ZP (має бути 1-6).")
                    zp_list_parsed.append([elem_id, face_id, load_val])
            self.ZP = np.array(zp_list_parsed, dtype=np.float64) if zp_list_parsed else np.empty((0, 3), dtype=np.float64)

            if show_info:
                messagebox.showinfo("Граничні умови", "Граничні умови застосовано.")
            print(f"Застосовано ZU: {self.ZU}")
            print(f"Застосовано ZP: {self.ZP}")
            self._update_array_combo_list()
        except ValueError as e:
            messagebox.showerror("Помилка введення ГУ", f"{e}")
            self.ZU, self.ZP = None, None
        except Exception as e:
            messagebox.showerror("Помилка застосування ГУ", f"Загальна помилка: {e}", detail=traceback.format_exc())
            self.ZU, self.ZP = None, None


    def _get_selected_gauss_point_index_from_str(self):
        # ... (код без змін) ...
        selected_display_str = self.selected_gauss_point_display_str.get()
        try:
            idx = self.gauss_point_display_list.index(selected_display_str)
            return idx
        except ValueError:
            if self.gauss_point_display_list:
                self.selected_gauss_point_display_str.set(self.gauss_point_display_list[0])
            return 0

    def _show_array_callback(self):
        # ... (код без змін, але використовує нові заголовки для U) ...
        array_name = self.selected_array_var.get()
        data_for_viewer, title_for_viewer = None, array_name
        custom_headers_for_viewer = None
        
        gp_idx = 0 
        if array_name.startswith("DFIABG") or array_name.startswith("DFIXYZ"):
            gp_idx = self._get_selected_gauss_point_index_from_str()

        try:
            if array_name == "AKT": 
                data_for_viewer = self.AKT.T if self.AKT is not None else None
                title_for_viewer = "AKT (Координати вузлів, Транспоновано)"
                custom_headers_for_viewer = ["X", "Y", "Z"]
            elif array_name == "NT": 
                data_for_viewer = self.NT.T if self.NT is not None else None
                title_for_viewer = "NT (Таблиця зв'язності, Транспоновано)"
                if data_for_viewer is not None:
                    custom_headers_for_viewer = [f"Елем {i+1}" for i in range(data_for_viewer.shape[1])]
            elif array_name == "ZU": 
                data_for_viewer = self.ZU[:,np.newaxis] if self.ZU is not None and self.ZU.size > 0 else np.array([[]])
                title_for_viewer = "ZU (Закріплені вузли, 1-based)"
                custom_headers_for_viewer = ["Номер вузла"]
            elif array_name == "ZP": 
                data_for_viewer = self.ZP if self.ZP is not None and self.ZP.size > 0 else np.array([[]])
                title_for_viewer = "ZP (Навантаження: Елем, Грань, Тиск)"
                custom_headers_for_viewer = ["Елемент", "Грань", "Тиск"]
            elif array_name == "DFIABG":
                if self.DFIABG is not None:
                    if 0 <= gp_idx < self.DFIABG.shape[0]:
                        data_for_viewer = self.DFIABG[gp_idx, :, :].T 
                        title_for_viewer = f"DFIABG ({self.selected_gauss_point_display_str.get()}, Вузол x (da,db,dg))"
                        custom_headers_for_viewer = ["d/d\u03B1", "d/d\u03B2", "d/d\u03B3"]
                    else: messagebox.showwarning("Показ DFIABG", f"Некоректний індекс точки Гаусса: {gp_idx}."); return
                else: data_for_viewer = None
            elif array_name.startswith("DFIXYZ (elem"):
                elem_id_view = self.selected_element_for_dfixyz_idx.get() 
                if elem_id_view in self.DFIXYZ and self.DFIXYZ[elem_id_view] is not None:
                    dfixyz_for_elem = self.DFIXYZ[elem_id_view]
                    if 0 <= gp_idx < dfixyz_for_elem.shape[0]:
                        data_for_viewer = dfixyz_for_elem[gp_idx, :, :] 
                        title_for_viewer = f"DFIXYZ (Елем #{elem_id_view}, {self.selected_gauss_point_display_str.get()}, Вузол x (dX,dY,dZ))"
                        custom_headers_for_viewer = ["d/dX", "d/dY", "d/dZ"]
                    else: messagebox.showwarning("Показ DFIXYZ", f"Некоректний індекс точки Гаусса: {gp_idx}."); return
                else: data_for_viewer = None
            elif array_name.startswith("MGE (elem"):
                elem_id_view = self.selected_element_for_dfixyz_idx.get()
                if elem_id_view in self.MGE and self.MGE[elem_id_view] is not None:
                    data_for_viewer = self.MGE[elem_id_view]
                    title_for_viewer = f"MGE (Елем #{elem_id_view}, 60x60)"
                else: data_for_viewer = None
            elif array_name.startswith("FE (elem"):
                elem_id_view = self.selected_element_for_dfixyz_idx.get()
                if elem_id_view in self.FE and self.FE[elem_id_view] is not None:
                    data_for_viewer = self.FE[elem_id_view][:, np.newaxis] 
                    title_for_viewer = f"FE (Елем #{elem_id_view}, Вектор навантажень, 60x1)"
                    custom_headers_for_viewer = ["Значення сили"]
                else: data_for_viewer = None
            elif array_name == "MG (Global Stiffness)": 
                data_for_viewer = self.MG
                title_for_viewer = "MG (Глобальна матриця жорсткості)"
            elif array_name == "F (Global Load)": 
                data_for_viewer = self.F[:, np.newaxis] if self.F is not None else None
                title_for_viewer = "F (Глобальний вектор навантажень)"
                custom_headers_for_viewer = ["Значення сили"]
            elif array_name == "U (Displacements)":
                if self.U is not None and self.nqp > 0:
                    display_data_list = []
                    axes = ["X", "Y", "Z"]
                    num_total_dofs = self.U.shape[0]
                    for i in range(num_total_dofs):
                        node_id_0based = i // 3
                        node_id_1based = node_id_0based + 1
                        if node_id_1based > self.nqp:
                             display_data_list.append([f"Вузол {node_id_1based} (поза nqp?)", axes[i % 3], self.U[i]])
                        else:
                            display_data_list.append([node_id_1based, axes[i % 3], self.U[i]])
                    if display_data_list:
                        data_for_viewer = np.array(display_data_list, dtype=object)
                        custom_headers_for_viewer = ["Вузол (глоб.)", "Вісь", "Переміщення"]
                        title_for_viewer = "U (Вектор переміщень)"
                    else: data_for_viewer = None
                else: data_for_viewer = None
            else: 
                data_for_viewer = getattr(self, array_name, None)
                
            if data_for_viewer is None or (isinstance(data_for_viewer, np.ndarray) and data_for_viewer.size == 0) :
                 messagebox.showwarning("Показ масиву", f"Масив '{array_name}' ще не обчислено, порожній або не вибрано відповідний елемент/точку Гаусса.");
                 return
            array_viewer.create_array_viewer_window(self, data_for_viewer, title_for_viewer, column_headers=custom_headers_for_viewer)
        except AttributeError:
             messagebox.showwarning("Показ масиву", f"Атрибут для масиву '{array_name}' не знайдено.")
        except Exception as e:
            messagebox.showerror("Помилка показу", f"Помилка при відображенні '{array_name}': {e}", detail=traceback.format_exc())


    def _solve_callback(self):
        try:
            if self.AKT is None or self.NT is None or self.nel == 0:
                raise ValueError("Спочатку згенеруйте сітку (Nx, Ny, Nz > 0).")
            if self.ZU is None or self.ZP is None:
                self._apply_bcs_callback(show_info=False)
                if self.ZU is None:
                     raise ValueError("Не визначено граничні умови закріплення (ZU). Застосуйте ГУ.")
            
            self.results_label.config(text="Запуск розрахунку...")
            self.update_idletasks()

            print("--- Початок розрахунку МСЕ ---")
            if self.DFIABG is None:
                print("Обчислення DFIABG...")
                self.DFIABG = derivatives.compute_DFIABG(
                    self.alpha_g_flat, self.beta_g_flat, self.gamma_g_flat
                )
                if np.isnan(self.DFIABG).any(): raise ValueError("Помилка в DFIABG (NaN).")
            print("DFIABG обчислено/використано попереднє.")

            full_size = 3 * self.nqp
            if full_size == 0: raise ValueError("Кількість вузлів nqp = 0.")
            
            if full_size > 2000 and not messagebox.askyesno("Увага", f"Розмір системи {full_size}x{full_size}. Продовжити?"):
                self.results_label.config(text="Розрахунок скасовано.")
                return
            
            print(f"Ініціалізація MG ({full_size}x{full_size}) та F ({full_size})...")
            self.MG = np.zeros((full_size, full_size))
            self.F = np.zeros(full_size)
            self.DJ, self.DFIXYZ, self.MGE, self.FE = {}, {}, {}, {}

            print("Обчислення параметрів матеріалу...")
            E = self.E_var.get(); nu = self.nu_var.get()
            lambda_, mu_ = stiffness.calculate_lambda_mu(E, nu)
            if np.isnan(lambda_) or np.isnan(mu_):
                raise ValueError(f"Не вдалося обчислити параметри Ламе для E={E}, nu={nu}.")
            gauss_weights_3D = stiffness.gauss_weights_3D
            print(f"Параметри: E={E}, nu={nu} => lambda={lambda_:.3e}, mu={mu_:.3e}")

            print("Збирання глобальної системи...")
            singular_elements_jacobian = []
            for elem_id in range(1, self.nel + 1):
                if elem_id % max(1, self.nel // 10) == 0 or elem_id == self.nel or self.nel < 10:
                     print(f" Обробка елемента {elem_id}/{self.nel}...")
                
                dfixyz_elem, dj_dets_elem, dets_ok = derivatives.compute_DFIXYZ_for_element(
                    elem_id, self.AKT, self.NT, self.DFIABG
                )
                self.DFIXYZ[elem_id] = dfixyz_elem
                self.DJ[elem_id] = dj_dets_elem
                if not dets_ok: singular_elements_jacobian.append(elem_id)

                mge_elem_current = np.zeros((60,60))
                if dets_ok and not np.isnan(dfixyz_elem).any() and not np.any(np.abs(dj_dets_elem) < 1e-9):
                    mge_elem_current = stiffness.compute_element_stiffness_MGE(
                        dfixyz_elem, dj_dets_elem, lambda_, mu_, gauss_weights_3D
                    )
                self.MGE[elem_id] = mge_elem_current
                
                fe_elem_current = np.zeros(60)
                if self.ZP is not None and self.ZP.shape[0] > 0:
                    loads_for_current_element = self.ZP[self.ZP[:, 0] == elem_id]
                    if loads_for_current_element.shape[0] > 0:
                        elem_nodes_global_0based_for_akt = self.NT[:, elem_id - 1] - 1
                        valid_nodes_mask = elem_nodes_global_0based_for_akt >= 0
                        if not np.all(valid_nodes_mask):
                             print(f"  Попередження: Елемент {elem_id} має невалідні вузли в NT, пропуск розрахунку FE.")
                        else:
                            try:
                                akt_elem_nodes = self.AKT[:, elem_nodes_global_0based_for_akt]
                                for load_data in loads_for_current_element:
                                    face_id_to_load = int(load_data[1])
                                    pressure_val = float(load_data[2]) # Залишаємо як є, знак буде враховано в boundary
                                    # print(f"    Застосування навантаження: Елем {elem_id}, Грань {face_id_to_load}, Тиск {pressure_val}")
                                    fe_face = boundary.compute_element_load_vector(
                                        pressure_val, face_id_to_load, akt_elem_nodes
                                    )
                                    fe_elem_current += fe_face
                            except IndexError as e_idx:
                                print(f"  ПОМИЛКА Індексації при розрахунку FE для елемента {elem_id}: {e_idx}")
                self.FE[elem_id] = fe_elem_current

                elem_nodes_global_0based = self.NT[:, elem_id - 1] - 1
                for r_loc in range(20):
                     g_r = elem_nodes_global_0based[r_loc]
                     if g_r < 0 or g_r >= self.nqp : continue
                     for c_loc in range(20):
                          g_c = elem_nodes_global_0based[c_loc]
                          if g_c < 0 or g_c >= self.nqp : continue
                          for dof_r in range(3):
                              mg_r_idx = 3*g_r+dof_r; mge_r_idx = r_loc + 20*dof_r
                              for dof_c in range(3):
                                   mg_c_idx = 3*g_c+dof_c; mge_c_idx = c_loc + 20*dof_c
                                   if 0 <= mg_r_idx < full_size and 0 <= mg_c_idx < full_size and \
                                      0 <= mge_r_idx < 60 and 0 <= mge_c_idx < 60:
                                        self.MG[mg_r_idx, mg_c_idx] += mge_elem_current[mge_r_idx, mge_c_idx]
                boundary.assemble_global_load(self.F, fe_elem_current, elem_nodes_global_0based, self.nqp)

            if singular_elements_jacobian:
                messagebox.showwarning("Попередження про Якобіан", f"Сингулярний Якобіан: {singular_elements_jacobian}.")
            print("Збирання завершено.")

            print("Врахування ГУ (ZU)...")
            if self.ZU is None:
                print("  Попередження: ZU не визначено.")
            else:
                self.MG, self.F = boundary.apply_boundary_conditions(self.MG, self.F, self.ZU, self.nqp)
            print("ГУ закріплення враховано.")

            print("Розв'язування СЛАР...")
            self.U = solver.solve_system(self.MG, self.F)
            
            if self.U is None or np.isnan(self.U).any():
                 self.results_label.config(text="Помилка розв'язку СЛАР.")
                 messagebox.showerror("Помилка розв'язку", "Не вдалося розв'язати СЛАР.")
                 self.deformed_AKT = None # Немає деформованих координат
                 self._update_visualization() # Оновити, щоб показати тільки оригінал
                 return

            print("СЛАР розв'язано.")
            # Обчислення деформованих координат
            print("Обчислення деформованих координат...")
            self.deformed_AKT = self.AKT.copy()
            num_nodes_to_process = min(self.nqp, self.U.shape[0] // 3)
            for node_idx_0based in range(num_nodes_to_process):
                u_x_idx, u_y_idx, u_z_idx = 3*node_idx_0based, 3*node_idx_0based+1, 3*node_idx_0based+2
                if u_z_idx < self.U.shape[0]:
                    self.deformed_AKT[0, node_idx_0based] += self.U[u_x_idx]
                    self.deformed_AKT[1, node_idx_0based] += self.U[u_y_idx]
                    self.deformed_AKT[2, node_idx_0based] += self.U[u_z_idx]
            print("Деформовані координати обчислено.")

            print("Обчислення напружень (ЗАГЛУШКА)...")
            self.SIGMA = np.zeros((self.nqp, 6))
            self.SIGMA_P = np.zeros((self.nqp, 3))

            self._update_array_combo_list()
            self._update_visualization() # Тепер _update_visualization сама вирішить, що показувати
            
            max_U_abs = np.max(np.abs(self.U)) if self.U is not None else 0.0
            results_text = f"Розрахунок успішно завершено.\nМакс. абс. переміщення: {max_U_abs:.6e}"
            self.results_label.config(text=results_text)
            if max_U_abs < 1e-12 and np.any(self.F != 0): # Якщо є навантаження, а переміщення нульові
                messagebox.showwarning("Результат розрахунку", "Розрахунок завершено, але переміщення близькі до нуля. Перевірте навантаження та ГУ.")
            else:
                messagebox.showinfo("Розрахунок завершено", results_text)

        except ValueError as e:
            self.results_label.config(text=f"Помилка даних: {e}")
            messagebox.showerror("Помилка даних", f"{e}")
            print(f"ПОМИЛКА ДАНИХ: {e}\n{traceback.format_exc()}")
        except MemoryError as e:
            self.results_label.config(text=f"Помилка пам'яті: {e}")
            messagebox.showerror("Помилка пам'яті", f"Недостатньо пам'яті: {e}.")
            print(f"ПОМИЛКА ПАМ'ЯТІ: {e}\n{traceback.format_exc()}")
        except Exception as e:
            self.results_label.config(text=f"Загальна помилка: {e}")
            messagebox.showerror("Помилка розрахунку", f"Непередбачена помилка: {e}", detail=traceback.format_exc())
            print(f"ЗАГАЛЬНА ПОМИЛКА: {e}\n{traceback.format_exc()}")
        finally:
            print("--- Розрахунок МСЕ завершено (або перервано) ---")

    def _update_array_combo_list(self):
        # ... (код без змін) ...
         base_list = ["AKT", "NT"]
         if self.deformed_AKT is not None: # Додаємо деформовані координати, якщо є
             base_list.append("deformed_AKT")
         prev_selection = self.selected_array_var.get()
         
         if self.ZU is not None and self.ZU.size > 0 : base_list.append("ZU")
         if self.ZP is not None and self.ZP.size > 0 : base_list.append("ZP")
         
         if self.DFIABG is not None: base_list.append("DFIABG")
         if len(self.DFIXYZ) > 0 : base_list.append("DFIXYZ (elem ...)") 

         mge_keys = sorted([k for k in self.MGE.keys() if self.MGE[k] is not None and self.MGE[k].size > 0])
         fe_keys = sorted([k for k in self.FE.keys() if self.FE[k] is not None and self.FE[k].size > 0])

         for eid in mge_keys: base_list.append(f"MGE (elem {eid})")
         for eid in fe_keys: base_list.append(f"FE (elem {eid})")

         if self.MG is not None: base_list.append("MG (Global Stiffness)")
         if self.F is not None: base_list.append("F (Global Load)")
         if self.U is not None: base_list.append("U (Displacements)")
         
         self.array_combo['values'] = base_list
         if prev_selection in base_list: self.selected_array_var.set(prev_selection)
         elif base_list: self.selected_array_var.set(base_list[0])
         else: self.selected_array_var.set("")
         self._update_array_params_visibility()