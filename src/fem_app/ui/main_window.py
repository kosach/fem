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
        self.title("МСЕ Аналіз Паралелепіпеда v3.6 (Ручне керування знаком тиску)")
        self.geometry("1250x850") 
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
        
        self.default_pressure_magnitude_var = tk.DoubleVar(value=0.5)
        # self.pressure_direction_sign = tk.IntVar(value=-1) # Більше не використовується
        # self.pressure_direction_label_var = tk.StringVar(value="Напрям тиску: Стискаючий (-)") # Більше не використовується


        self.AKT, self.NT, self.node_map, self.nqp, self.nel = None, None, None, 0, 0
        self.deformed_AKT = None 
        self.ZU = None 
        self.ZP = None 
        
        self.DFIABG, self.MG, self.F, self.U = None, None, None, None
        self.SIGMA, self.SIGMA_P = None, None
        self.DJ, self.DFIXYZ, self.MGE, self.FE = {}, {}, {}, {}
        self.loaded_faces_info_for_vis = [] 

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

        self.show_deformation_var = tk.BooleanVar(value=False)
        self.deformation_scale_var = tk.DoubleVar(value=1.0)
        
        self.show_nodes_var = tk.BooleanVar(value=True)
        self.show_node_labels_var = tk.BooleanVar(value=True)
        self.show_element_outline_var = tk.BooleanVar(value=True)
        self.show_face_numbers_var = tk.BooleanVar(value=False) 
        self.show_pressure_arrows_var = tk.BooleanVar(value=True)

        self.after(50, self._initial_setup_from_image)

    def _initial_setup_from_image(self):
        print("Виконується початкове налаштування...")
        self._generate_grid_callback(show_confirmation=False)
        if self.AKT is not None and self.nel > 0:
            print("Встановлення ГУ за замовчуванням...")
            self._set_default_zu()
            self._set_default_zp() 
            self._apply_bcs_callback(show_info=False) 
            print("ГУ за замовчуванням встановлено та застосовано.")
        else:
            messagebox.showwarning("Початкове налаштування", "Не вдалося згенерувати сітку за замовчуванням.")
            print("Помилка початкового налаштування: сітка не була згенерована належним чином.")

    def _create_ui(self):
        settings_frame = ttk.Frame(self, padding="10", width=400) 
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
        self._create_vis_tab()
        self._create_arrays_tab()

    def _create_setup_panel(self, parent_frame):
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
        frame = ttk.LabelFrame(parent_frame, text="Граничні умови", padding="10")
        frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(frame, text="Закріплені вузли ZU (номери через кому):").grid(row=0, column=0, columnspan=3, sticky='w')
        self.zu_entry = ttk.Entry(frame, textvariable=self.zu_text)
        self.zu_entry.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 5))
        
        btn_def_zu = ttk.Button(frame, text="Низ (закр.)", command=self._set_default_zu)
        btn_def_zu.grid(row=2, column=0, sticky='ew')
        btn_clr_zu = ttk.Button(frame, text="Очист. ZU", command=lambda: self.zu_text.set(""))
        btn_clr_zu.grid(row=2, column=1, sticky='ew', padx=(5,0))

        ttk.Label(frame, text="Навантаження ZP (Елем, Грань, Тиск зі знаком):").grid(row=3, column=0, columnspan=3, sticky='w', pady=(10,0)) # Змінено текст
        self.zp_entry = scrolledtext.ScrolledText(frame, height=4, width=35) 
        self.zp_entry.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(0, 5))
        
        btn_def_zp = ttk.Button(frame, text="Верх (тиск за замовч.)", command=self._set_default_zp) # Змінено текст кнопки
        btn_def_zp.grid(row=5, column=0, sticky='ew')
        btn_clr_zp = ttk.Button(frame, text="Очист. ZP", command=lambda: self.zp_entry.delete("1.0", tk.END))
        btn_clr_zp.grid(row=5, column=1, sticky='ew', padx=(5,0))

        # self.pressure_dir_status_label = ttk.Label(frame, textvariable=self.pressure_direction_label_var) # Видалено
        # self.pressure_dir_status_label.grid(row=6, column=0, columnspan=2, sticky='w', pady=(5,0)) # Видалено
        
        btn_toggle_pressure_dir = ttk.Button(frame, text="Інвертувати знаки тиску в ZP", command=self._toggle_pressure_direction) # Змінено текст кнопки
        btn_toggle_pressure_dir.grid(row=6, column=0, columnspan=3, sticky='ew', pady=(5,0)) # Розширено на всі стовпці

        ttk.Label(frame, text="Велич. тиску (для замовч.):").grid(row=7, column=0, sticky='w', pady=(5,0)) # Змінено текст
        self.default_pressure_magnitude_entry = ttk.Entry(frame, textvariable=self.default_pressure_magnitude_var, width=10)
        self.default_pressure_magnitude_entry.grid(row=7, column=1, sticky='ew', pady=(5,0), columnspan=2) # Розширено

        btn_apply_bc = ttk.Button(frame, text="Застосувати введені ГУ", command=self._apply_bcs_callback)
        btn_apply_bc.grid(row=8, column=0, columnspan=3, pady=(10,0), sticky='ew')
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def _toggle_pressure_direction(self):
        print("--- Debug: Entering _toggle_pressure_direction ---")
        current_zp_text = self.zp_entry.get("1.0", tk.END).strip()
        if not current_zp_text:
            messagebox.showinfo("Інверсія тиску", "Поле ZP порожнє. Немає значень для інверсії.")
            return

        modified_zp_lines = []
        had_errors = False
        for i, line_str in enumerate(current_zp_text.split('\n')):
            line_str = line_str.strip()
            if not line_str or line_str.startswith('#'):
                modified_zp_lines.append(line_str) # Зберігаємо коментарі та порожні рядки
                continue
            
            parts = [p.strip() for p in line_str.split(',') if p.strip()]
            if len(parts) == 3:
                try:
                    elem_id_str, face_id_str, pressure_val_str = parts
                    pressure_val = float(pressure_val_str)
                    inverted_pressure_val = -pressure_val
                    modified_zp_lines.append(f"{elem_id_str}, {face_id_str}, {inverted_pressure_val:.4g}")
                except ValueError:
                    modified_zp_lines.append(line_str) # Залишаємо рядок без змін, якщо помилка парсингу
                    print(f"Помилка парсингу значення тиску в рядку ZP: '{line_str}'")
                    had_errors = True
            else:
                modified_zp_lines.append(line_str) # Залишаємо рядок без змін, якщо неправильний формат
                print(f"Неправильний формат рядка ZP: '{line_str}'")
                had_errors = True
        
        if had_errors:
            messagebox.showwarning("Інверсія тиску", "Деякі рядки в ZP не вдалося обробити. Вони залишені без змін.")

        self.zp_entry.delete("1.0", tk.END)
        self.zp_entry.insert("1.0", "\n".join(modified_zp_lines))
        print("Знаки тиску в полі ZP інвертовано.")

        try:
            print("Автоматичне оновлення ГУ після інверсії знаків тиску...")
            self._apply_bcs_callback(show_info=False) 
            messagebox.showinfo("Інверсія тиску", "Знаки тиску в ZP інвертовано. Граничні умови оновлено.")
        except Exception as e:
            messagebox.showerror("Помилка оновлення ГУ", f"Помилка під час автоматичного оновлення ГУ після інверсії: {e}\n{traceback.format_exc()}")
            print(f"ПОМИЛКА під час автоматичного оновлення ГУ після інверсії: {e}")
            traceback.print_exc()
        print("--- Debug: Exiting _toggle_pressure_direction ---")


    def _create_solve_panel(self, parent_frame):
         frame = ttk.LabelFrame(parent_frame, text="Розрахунок", padding="10")
         frame.pack(fill="x", pady=(0, 10))
         btn_solve = ttk.Button(frame, text="Зібрати систему та Розв'язати", command=self._solve_callback)
         btn_solve.pack(pady=(5, 10), fill='x')
         ttk.Label(frame, text="Результати:").pack(anchor='w')
         self.results_label = ttk.Label(frame, text="Розрахунок не виконано.", justify=tk.LEFT, wraplength=380) 
         self.results_label.pack(anchor='w', pady=(0, 5), fill='x')

    def _create_vis_tab(self):
        frame = self.tab_vis
        
        options_vis_frame = ttk.Frame(frame)
        options_vis_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))

        ttk.Checkbutton(options_vis_frame, text="Вузли", variable=self.show_nodes_var, command=self._update_visualization).pack(side='left', padx=2)
        ttk.Checkbutton(options_vis_frame, text="Номери вузлів", variable=self.show_node_labels_var, command=self._update_visualization).pack(side='left', padx=2)
        ttk.Checkbutton(options_vis_frame, text="Контур елементів", variable=self.show_element_outline_var, command=self._update_visualization).pack(side='left', padx=2)
        ttk.Checkbutton(options_vis_frame, text="Номери граней", variable=self.show_face_numbers_var, command=self._update_visualization).pack(side='left', padx=2) 
        ttk.Checkbutton(options_vis_frame, text="Напрям тиску", variable=self.show_pressure_arrows_var, command=self._update_visualization).pack(side='left', padx=2) 
        
        ttk.Checkbutton(options_vis_frame, text="Деформація", variable=self.show_deformation_var, command=self._update_visualization).pack(side='left', padx=(10,2))
        ttk.Label(options_vis_frame, text="Масштаб деф.:").pack(side='left', padx=(5,0))
        self.deformation_scale_spinbox = tk.Spinbox(options_vis_frame, from_=0.1, to=10000.0, increment=0.5, textvariable=self.deformation_scale_var, width=7, command=self._update_visualization)
        self.deformation_scale_spinbox.pack(side='left', padx=2)

        self.vis_fig = plt.figure(figsize=(7, 6)) 
        self.vis_ax = self.vis_fig.add_subplot(111, projection='3d')
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=frame)
        canvas_widget = self.vis_canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.vis_canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_arrays_tab(self):
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
        if self.gauss_point_display_list: self.gauss_point_combo.current(0)
        self.gauss_point_combo.pack(side='left', padx=(0, 10))


        self.element_label = ttk.Label(self.params_frame, text="Елемент (1-Nel):")
        self.element_spinbox = tk.Spinbox(self.params_frame, from_=1, to=1, textvariable=self.selected_element_for_dfixyz_idx, width=7)
        self.element_label.pack(side='left', padx=(10,2)) 
        self.element_spinbox.pack(side='left', padx=(0,10))
        
        action_frame = ttk.Frame(frame)
        action_frame.pack(pady=5, fill='x')
        ttk.Button(action_frame, text="Показати масив", command=self._show_array_callback).pack(side='left')
        
        self._update_array_params_visibility()


    def _on_array_selected(self, event=None): 
        self._update_array_params_visibility()

    def _update_array_params_visibility(self):
        selected_name = self.selected_array_var.get()
        
        self.params_frame.pack_forget()
        if hasattr(self.gauss_point_combo, 'master') and self.gauss_point_combo.master.winfo_exists() and \
           len(self.gauss_point_combo.master.winfo_children()) > 0 and \
           hasattr(self.gauss_point_combo.master.winfo_children()[0], 'winfo_exists') and \
           self.gauss_point_combo.master.winfo_children()[0].winfo_exists():
            self.gauss_point_combo.master.winfo_children()[0].pack_forget() 
        
        if hasattr(self.gauss_point_combo, 'winfo_exists') and self.gauss_point_combo.winfo_exists():
            self.gauss_point_combo.pack_forget()
        if hasattr(self.element_label, 'winfo_exists') and self.element_label.winfo_exists():
            self.element_label.pack_forget()
        if hasattr(self.element_spinbox, 'winfo_exists') and self.element_spinbox.winfo_exists():
            self.element_spinbox.pack_forget()

        needs_gauss_point = selected_name.startswith("DFIABG") or selected_name.startswith("DFIXYZ (elem")
        needs_element_id = selected_name.startswith("DFIXYZ (elem") or \
                           selected_name.startswith("MGE (elem") or \
                           selected_name.startswith("FE (elem") or \
                           selected_name.startswith("DJ (elem")


        if needs_gauss_point or needs_element_id:
            self.params_frame.pack(pady=(0, 5), fill='x', after=self.array_combo.master)
            if needs_gauss_point:
                if hasattr(self.gauss_point_combo, 'master') and self.gauss_point_combo.master.winfo_exists() and \
                   len(self.gauss_point_combo.master.winfo_children()) > 0 and \
                   hasattr(self.gauss_point_combo.master.winfo_children()[0], 'winfo_exists') and \
                   self.gauss_point_combo.master.winfo_children()[0].winfo_exists():
                     self.gauss_point_combo.master.winfo_children()[0].pack(side='left', padx=(10, 2)) 
                if hasattr(self.gauss_point_combo, 'winfo_exists'):
                    self.gauss_point_combo.pack(side='left', padx=(0, 10))
            if needs_element_id:
                if hasattr(self.element_label, 'winfo_exists'):
                    self.element_label.pack(side='left', padx=(10, 2))
                if hasattr(self.element_spinbox, 'winfo_exists'):
                    self.element_spinbox.pack(side='left', padx=(0, 10))
                    max_elem = self.nel if self.nel > 0 else 1
                    self.element_spinbox.config(to=max_elem)
                    current_elem_val = self.selected_element_for_dfixyz_idx.get()
                    if not (1 <= current_elem_val <= max_elem):
                        self.selected_element_for_dfixyz_idx.set(min(max(1, current_elem_val), max_elem))

    def _generate_grid_callback(self, show_confirmation=True):
        try:
            nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
            ax_, ay_, az_ = self.ax_var.get(), self.ay_var.get(), self.az_var.get()
            if not (nx>0 and ny>0 and nz>0 and ax_>0 and ay_>0 and az_>0):
                raise ValueError("Nx, Ny, Nz, Ax, Ay, Az мають бути позитивними.")

            self.AKT, self.node_map, self.nqp = mesh.generate_mesh(nx, ny, nz, ax_, ay_, az_)
            self.NT, self.nel = mesh.generate_connectivity(nx, ny, nz, self.node_map)
            
            self.deformed_AKT = None 
            self.DFIABG, self.MG, self.F, self.U, self.SIGMA, self.SIGMA_P = None, None, None, None, None, None
            self.DJ, self.DFIXYZ, self.MGE, self.FE = {}, {}, {}, {}
            self.ZU, self.ZP = None, None 
            self.loaded_faces_info_for_vis = [] 
            
            self.results_label.config(text="Нову сітку згенеровано. Розрахунок не виконано.")
            self._update_array_combo_list() 
            
            if show_confirmation:
                self._set_default_zu() 
                self._set_default_zp() 
                self._apply_bcs_callback(show_info=False) 
                messagebox.showinfo("Сітка", f"Згенеровано: {self.nqp} вузлів, {self.nel} елементів.")
            
            self._update_visualization() 
            
        except ValueError as e: messagebox.showerror("Помилка введення", f"{e}")
        except Exception as e: messagebox.showerror("Помилка генерації", f"{e}", detail=traceback.format_exc())


    def _update_visualization(self):
        try:
            ax_dimensions = (self.ax_var.get(), self.ay_var.get(), self.az_var.get())
            
            akt_to_show_deformed = None
            scale = 1.0
            if self.show_deformation_var.get() and self.deformed_AKT is not None and self.U is not None:
                akt_to_show_deformed = self.deformed_AKT
                scale = self.deformation_scale_var.get()
            
            visualization.draw_mesh(
                ax=self.vis_ax, 
                AKT_orig=self.AKT, 
                NT=self.NT,
                ax_dims=ax_dimensions,
                show_nodes=self.show_nodes_var.get(),
                show_node_labels=self.show_node_labels_var.get(),
                show_element_outline=self.show_element_outline_var.get(),
                AKT_deformed=akt_to_show_deformed,
                deformation_scale=scale,
                show_orig_wireframe=True if akt_to_show_deformed is not None else False,
                show_face_numbers=self.show_face_numbers_var.get(), 
                loaded_faces_data=self.loaded_faces_info_for_vis if self.show_pressure_arrows_var.get() else None 
            )
            self.vis_canvas.draw_idle()
        except Exception as e: 
            print(f"Помилка візуалізації: {e}")
            traceback.print_exc()

    def _set_default_zu(self):
        if self.AKT is None or self.nqp == 0:
            print("Неможливо встановити ZU: сітку не згенеровано.")
            self.zu_text.set("")
            return
        try:
            min_z_coord = 0.0 
            bottom_nodes_indices = np.where(np.abs(self.AKT[2, :] - min_z_coord) < 1e-9)[0]

            if bottom_nodes_indices.size > 0:
                self.zu_text.set(", ".join([str(i + 1) for i in bottom_nodes_indices]))
                print(f"Встановлено ZU за замовчуванням (вузли на z=0): {self.zu_text.get()}")
            else:
                self.zu_text.set("")
                print("Не знайдено вузлів на z=0 для ZU за замовчуванням.")
        except Exception as e:
            print(f"Помилка автоматичного встановлення ZU: {e}")
            self.zu_text.set("")


    def _set_default_zp(self):
        if self.nel == 0 or self.AKT is None:
            print("Неможливо встановити ZP: сітку не згенеровано або немає елементів.")
            self.zp_entry.delete("1.0", tk.END)
            return

        zp_lines = []
        max_z_coord = self.az_var.get() 
        pressure_magnitude = self.default_pressure_magnitude_var.get()
        # За замовчуванням встановлюємо стискаючий тиск (від'ємний)
        signed_pressure_for_default = -pressure_magnitude 

        for elem_id_1based in range(1, self.nel + 1):
            elem_nodes_global_0based = self.NT[:, elem_id_1based - 1] - 1
            
            if np.any(elem_nodes_global_0based < 0) or np.any(elem_nodes_global_0based >= self.nqp):
                continue
            
            is_top_element_face = False
            upper_face_id = 6 
            if upper_face_id in boundary.FACE_NODES_MAP:
                upper_face_local_indices = boundary.FACE_NODES_MAP[upper_face_id]["nodes_3d_indices"]
                
                for loc_idx in upper_face_local_indices:
                    if loc_idx < len(elem_nodes_global_0based):
                        node_global_idx = elem_nodes_global_0based[loc_idx]
                        if 0 <= node_global_idx < self.AKT.shape[1] and \
                           np.abs(self.AKT[2, node_global_idx] - max_z_coord) < 1e-9:
                            is_top_element_face = True
                            break
            
            if is_top_element_face:
                # Записуємо тиск зі знаком у поле вводу
                zp_lines.append(f"{elem_id_1based}, {upper_face_id}, {signed_pressure_for_default:.4g}") 
        
        self.zp_entry.delete("1.0", tk.END) 
        if zp_lines:
            zp_default_str = "\n".join(zp_lines)
            self.zp_entry.insert("1.0", zp_default_str)
            print(f"Встановлено ZP за замовчуванням (тиск зі знаком): {zp_default_str}")
        else:
            print("Не знайдено верхніх граней для ZP за замовчуванням (або nel=0).")

    def _apply_bcs_callback(self, show_info=True):
        print("--- Debug: Entering _apply_bcs_callback ---")
        try:
            if self.ZU is None:
                print("Debug: self.ZU was None, initializing to empty array.")
                self.ZU = np.array([], dtype=int)
            if self.ZP is None:
                print("Debug: self.ZP was None, initializing to empty array.")
                self.ZP = np.empty((0,3), dtype=np.float64)

            print(f"Debug: Initial self.ZU (at start of _apply_bcs_callback): {self.ZU}")
            print(f"Debug: Initial self.ZP (at start of _apply_bcs_callback): {self.ZP}")

            zu_str = self.zu_text.get().strip()
            print(f"Debug: zu_str from UI: '{zu_str}'")
            if zu_str:
                parsed_zu_nodes = np.array([int(n.strip()) for n in zu_str.split(',') if n.strip()], dtype=int)
                print(f"Debug: parsed_zu_nodes: {parsed_zu_nodes}")
                if self.nqp is not None and self.nqp > 0:
                    if np.any(parsed_zu_nodes <= 0) or np.any(parsed_zu_nodes > self.nqp):
                        print("Debug: Invalid ZU nodes detected based on nqp.")
                        raise ValueError(f"Некоректні номери вузлів в ZU (мають бути від 1 до {self.nqp}).")
                elif parsed_zu_nodes.size > 0: 
                     print("Debug: ZU nodes provided, but nqp is not valid (0 or None).")
                     raise ValueError("Неможливо перевірити ZU: кількість вузлів сітки (nqp) не визначена або нульова.")
                self.ZU = parsed_zu_nodes
            else:
                self.ZU = np.array([], dtype=int)
            print(f"Debug: self.ZU after parsing: {self.ZU}")
            
            zp_str_from_widget = self.zp_entry.get("1.0", tk.END).strip()
            print(f"Debug: zp_str_from_widget:\n{zp_str_from_widget}")
            zp_list_parsed = []
            if zp_str_from_widget:
                for i, line in enumerate(zp_str_from_widget.split('\n')):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    print(f"Debug: Parsing ZP line {i+1}: '{line}'")
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                    if len(parts) != 3:
                        print(f"Debug: Invalid ZP format line {i+1}.")
                        raise ValueError(f"Неправильний формат ZP в рядку {i+1}. Очікується 'Елем, Грань, Тиск зі знаком'.")
                    try:
                        # Тепер третій елемент - це signed_pressure_value
                        elem_id, face_id, signed_pressure_value = int(parts[0]), int(parts[1]), float(parts[2])
                    except ValueError:
                        print(f"Debug: Invalid ZP numeric format line {i+1}.")
                        raise ValueError(f"Неправильний числовий формат в ZP, рядок {i+1}.")
                    
                    if self.nel is not None and self.nel > 0:
                        if elem_id <= 0 or elem_id > self.nel:
                            print(f"Debug: Invalid ZP element ID {elem_id} line {i+1} (nel={self.nel}).")
                            raise ValueError(f"Некоректний номер елемента {elem_id} в ZP (має бути від 1 до {self.nel}).")
                    elif elem_id > 0 : 
                        print(f"Debug: ZP element ID {elem_id} provided, but nel is not valid (0 or None).")
                        raise ValueError("Неможливо перевірити ZP: кількість елементів сітки (nel) не визначена або нульова.")

                    if not (1 <= face_id <= 6): 
                        print(f"Debug: Invalid ZP face ID {face_id} line {i+1}.")
                        raise ValueError(f"Некоректний номер грані {face_id} в ZP (має бути 1-6).")
                    
                    # signed_load_val = self.pressure_direction_sign.get() * load_magnitude # БІЛЬШЕ НЕ ПОТРІБНО
                    zp_list_parsed.append([elem_id, face_id, signed_pressure_value]) # Зберігаємо значення тиску як є (зі знаком)
            
            self.ZP = np.array(zp_list_parsed, dtype=np.float64) if zp_list_parsed else np.empty((0, 3), dtype=np.float64)
            print(f"Debug: self.ZP after parsing: {self.ZP}")

            if show_info:
                messagebox.showinfo("Граничні умови", "Граничні умови застосовано.")
            
            print(f"Застосовано ZU: {self.ZU if self.ZU is not None and self.ZU.size > 0 else 'Немає'}")
            print(f"Застосовано ZP (зі знаком з поля вводу): {self.ZP if self.ZP is not None and self.ZP.size > 0 else 'Немає'}")
            
            print("Debug: Calling _prepare_loaded_faces_for_vis...")
            self._prepare_loaded_faces_for_vis() 
            print("Debug: Calling _update_array_combo_list...")
            self._update_array_combo_list()
            print("Debug: Calling _update_visualization...")
            self._update_visualization() 
            print("--- Debug: Exiting _apply_bcs_callback successfully ---")

        except ValueError as e:
            print(f"--- Debug: ValueError in _apply_bcs_callback: {e} ---")
            traceback.print_exc() 
            messagebox.showerror("Помилка введення ГУ", f"{e}")
            self.ZU, self.ZP = None, None 
            self.loaded_faces_info_for_vis = []
            self._update_visualization() 
        except Exception as e:
            print(f"--- Debug: Exception in _apply_bcs_callback: {e} ---")
            traceback.print_exc() 
            messagebox.showerror("Помилка застосування ГУ", f"Загальна помилка: {e}", detail=traceback.format_exc())
            self.ZU, self.ZP = None, None
            self.loaded_faces_info_for_vis = []
            self._update_visualization()

    def _get_selected_gauss_point_index_from_str(self):
        selected_display_str = self.selected_gauss_point_display_str.get()
        try:
            idx = self.gauss_point_display_list.index(selected_display_str)
            return idx
        except ValueError:
            if self.gauss_point_display_list:
                self.selected_gauss_point_display_str.set(self.gauss_point_display_list[0])
            return 0

    def _show_array_callback(self):
        array_name = self.selected_array_var.get()
        data_for_viewer, title_for_viewer = None, array_name
        custom_headers_for_viewer = None
        
        gp_idx = 0 
        elem_id_view = 1 
        if array_name.startswith("DFIABG") or array_name.startswith("DFIXYZ (elem") or \
           array_name.startswith("DJ (elem"):
            gp_idx = self._get_selected_gauss_point_index_from_str()
        
        if array_name.startswith("DFIXYZ (elem") or array_name.startswith("MGE (elem") or \
           array_name.startswith("FE (elem") or array_name.startswith("DJ (elem"):
            elem_id_view = self.selected_element_for_dfixyz_idx.get()


        try:
            if array_name == "AKT": 
                data_for_viewer = self.AKT.T if self.AKT is not None else None
                title_for_viewer = "AKT (Координати вузлів, Транспоновано)"
                custom_headers_for_viewer = ["X", "Y", "Z"]
            elif array_name == "deformed_AKT":
                data_for_viewer = self.deformed_AKT.T if self.deformed_AKT is not None else None
                title_for_viewer = "deformed_AKT (Деформовані координати, Транспоновано)"
                custom_headers_for_viewer = ["X_def", "Y_def", "Z_def"]
            elif array_name == "NT": 
                data_for_viewer = self.NT.T if self.NT is not None else None
                title_for_viewer = "NT (Таблиця зв'язності, Транспоновано)"
                if data_for_viewer is not None and data_for_viewer.ndim == 2: 
                     custom_headers_for_viewer = [f"Вузол {i+1}" for i in range(data_for_viewer.shape[1])] 
                elif data_for_viewer is not None and data_for_viewer.ndim == 1 and self.nel == 1: 
                     custom_headers_for_viewer = [f"Вузол {i+1}" for i in range(data_for_viewer.shape[0])]

            elif array_name == "ZU": 
                data_for_viewer = self.ZU[:,np.newaxis] if self.ZU is not None and self.ZU.size > 0 else np.array([[]])
                title_for_viewer = "ZU (Закріплені вузли, 1-based)"
                custom_headers_for_viewer = ["Номер вузла"]
            elif array_name == "ZP": 
                data_for_viewer = self.ZP if self.ZP is not None and self.ZP.size > 0 else np.array([[]])
                title_for_viewer = "ZP (Навантаження: Елем, Грань, Тиск зі знаком)" # Змінено
                custom_headers_for_viewer = ["Елемент", "Грань", "Тиск (зі знаком)"]
            elif array_name == "DFIABG":
                if self.DFIABG is not None:
                    if 0 <= gp_idx < self.DFIABG.shape[0]:
                        data_for_viewer = self.DFIABG[gp_idx, :, :].T 
                        title_for_viewer = f"DFIABG ({self.selected_gauss_point_display_str.get()}, Вузол x (da,db,dg))"
                        custom_headers_for_viewer = ["d/d\u03B1", "d/d\u03B2", "d/d\u03B3"]
                    else: messagebox.showwarning("Показ DFIABG", f"Некоректний індекс точки Гаусса: {gp_idx}."); return
                else: data_for_viewer = None
            elif array_name.startswith("DFIXYZ (elem"):
                if elem_id_view in self.DFIXYZ and self.DFIXYZ[elem_id_view] is not None:
                    dfixyz_for_elem = self.DFIXYZ[elem_id_view] 
                    if 0 <= gp_idx < dfixyz_for_elem.shape[0]:
                        data_for_viewer = dfixyz_for_elem[gp_idx, :, :] 
                        title_for_viewer = f"DFIXYZ (Елем #{elem_id_view}, {self.selected_gauss_point_display_str.get()}, Вузол x (dX,dY,dZ))"
                        custom_headers_for_viewer = ["d/dX", "d/dY", "d/dZ"]
                    else: messagebox.showwarning("Показ DFIXYZ", f"Некоректний індекс точки Гаусса: {gp_idx}."); return
                else: data_for_viewer = None
            elif array_name.startswith("DJ (elem"): 
                if elem_id_view in self.DJ and self.DJ[elem_id_view] is not None:
                    dj_for_elem = self.DJ[elem_id_view] 
                    if 0 <= gp_idx < dj_for_elem.shape[0]:
                        data_for_viewer = np.array([[dj_for_elem[gp_idx]]]) 
                        title_for_viewer = f"DJ (det(J)) (Елем #{elem_id_view}, {self.selected_gauss_point_display_str.get()})"
                        custom_headers_for_viewer = ["det(J)"]
                    else: messagebox.showwarning("Показ DJ", f"Некоректний індекс точки Гаусса: {gp_idx}."); return
                else: data_for_viewer = None
            elif array_name.startswith("MGE (elem"):
                if elem_id_view in self.MGE and self.MGE[elem_id_view] is not None:
                    data_for_viewer = self.MGE[elem_id_view]
                    title_for_viewer = f"MGE (Елем #{elem_id_view}, 60x60)"
                else: data_for_viewer = None
            elif array_name.startswith("FE (elem"):
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

    def _prepare_loaded_faces_for_vis(self):
        self.loaded_faces_info_for_vis = []
        if self.ZP is not None and self.ZP.shape[0] > 0 and self.NT is not None and self.nel > 0: 
            for elem_id_1based, face_id, signed_pressure_val in self.ZP:
                elem_idx_0based = int(elem_id_1based) - 1
                face_id_int = int(face_id)
                
                if 0 <= elem_idx_0based < self.nel and face_id_int in boundary.FACE_NODES_MAP:
                    normal_vec_from_map = boundary.FACE_NODES_MAP[face_id_int]["normal_vector_direction"]
                    normal_vec = np.array(normal_vec_from_map, dtype=float) if not isinstance(normal_vec_from_map, np.ndarray) else normal_vec_from_map.astype(float)
                    
                    self.loaded_faces_info_for_vis.append(
                        (elem_idx_0based, face_id_int, signed_pressure_val, normal_vec)
                    )
        print(f"Підготовлено для візуалізації тиску: {len(self.loaded_faces_info_for_vis)} граней.")


    def _solve_callback(self):
        try:
            if self.AKT is None or self.NT is None or self.nel == 0:
                raise ValueError("Спочатку згенеруйте сітку (Nx, Ny, Nz > 0).")
            if self.ZU is None or self.ZP is None: 
                print("Debug: ZU or ZP is None in _solve_callback, attempting to apply BCs.")
                self._apply_bcs_callback(show_info=False) 
                if self.ZU is None: 
                     print("Debug: self.ZU is still None after _apply_bcs_callback in _solve_callback.")
                     raise ValueError("Не визначено граничні умови закріплення (ZU). Застосуйте ГУ.")
            
            self.results_label.config(text="Запуск розрахунку...")
            self.update_idletasks() 

            print("--- Початок розрахунку МСЕ ---")
            if self.DFIABG is None: 
                print("Обчислення DFIABG...")
                self.DFIABG = derivatives.compute_DFIABG(
                    self.alpha_g_flat, self.beta_g_flat, self.gamma_g_flat
                )
                if np.any(np.isnan(self.DFIABG)): raise ValueError("Помилка в DFIABG (NaN).") 
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
            E_val = self.E_var.get(); nu_val = self.nu_var.get()
            lambda_val, mu_val = stiffness.calculate_lambda_mu(E_val, nu_val)
            if np.isnan(lambda_val) or np.isnan(mu_val):
                raise ValueError(f"Не вдалося обчислити параметри Ламе для E={E_val}, nu={nu_val}.")
            
            gauss_weights_3D_local = stiffness.gauss_weights_3D 
            print(f"Параметри: E={E_val}, nu={nu_val} => lambda={lambda_val:.3e}, mu={mu_val:.3e}")

            print("Збирання глобальної системи...")
            singular_elements_jacobian = []
            for elem_id_1based in range(1, self.nel + 1):
                if elem_id_1based % max(1, self.nel // 20) == 0 or elem_id_1based == self.nel or self.nel < 20:
                     print(f" Обробка елемента {elem_id_1based}/{self.nel}...")
                
                dfixyz_elem, dj_dets_elem, dets_ok = derivatives.compute_DFIXYZ_for_element(
                    elem_id_1based, self.AKT, self.NT, self.DFIABG
                )
                self.DFIXYZ[elem_id_1based] = dfixyz_elem
                self.DJ[elem_id_1based] = dj_dets_elem 
                if not dets_ok: 
                    singular_elements_jacobian.append(elem_id_1based)

                mge_elem_current = np.zeros((60,60)) 
                if dets_ok and not np.any(np.isnan(dfixyz_elem)) and not np.any(np.abs(dj_dets_elem) < 1e-12):
                    mge_elem_current = stiffness.compute_element_stiffness_MGE(
                        dfixyz_elem, dj_dets_elem, lambda_val, mu_val, gauss_weights_3D_local
                    )
                else:
                    print(f"  ПОПЕРЕДЖЕННЯ: Пропуск MGE для елемента {elem_id_1based} через проблеми з Якобіаном або DFIXYZ.")

                self.MGE[elem_id_1based] = mge_elem_current
                
                fe_elem_current = np.zeros(60)
                if self.ZP is not None and self.ZP.shape[0] > 0:
                    loads_for_current_element = self.ZP[self.ZP[:, 0] == elem_id_1based]
                    
                    if loads_for_current_element.shape[0] > 0:
                        elem_nodes_global_0based_for_akt = self.NT[:, elem_id_1based - 1] - 1
                        
                        if np.any(elem_nodes_global_0based_for_akt < 0) or \
                           np.any(elem_nodes_global_0based_for_akt >= self.AKT.shape[1]):
                             print(f"  Попередження: Елемент {elem_id_1based} має невалідні глобальні індекси вузлів в NT, пропуск розрахунку FE.")
                        else:
                            try:
                                akt_elem_nodes = self.AKT[:, elem_nodes_global_0based_for_akt]
                                for load_data_row in loads_for_current_element:
                                    face_id_to_load = int(load_data_row[1])
                                    signed_pressure_val = float(load_data_row[2]) # Тепер це значення вже зі знаком
                                    
                                    fe_face_contribution = boundary.compute_element_load_vector(
                                        signed_pressure_val, face_id_to_load, akt_elem_nodes
                                    )
                                    fe_elem_current += fe_face_contribution
                            except IndexError as e_idx:
                                print(f"  ПОМИЛКА Індексації при розрахунку FE для елемента {elem_id_1based}: {e_idx}")
                                traceback.print_exc()
                self.FE[elem_id_1based] = fe_elem_current

                elem_nodes_global_0based = self.NT[:, elem_id_1based - 1] - 1 
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
                messagebox.showwarning("Попередження про Якобіан", 
                                       f"Сингулярний або проблемний Якобіан для елементів: {singular_elements_jacobian[:20]}"
                                       f"{'...' if len(singular_elements_jacobian)>20 else ''}."
                                       f"\nРезультати можуть бути некоректними.")
            print("Збирання завершено.")

            print("Врахування ГУ закріплення (ZU)...")
            if self.ZU is None or self.ZU.size == 0: 
                print("  Попередження: ZU не визначено або порожній. Модель може бути не закріплена.")
            
            self.MG, self.F = boundary.apply_boundary_conditions(self.MG, self.F, self.ZU, self.nqp)
            print("ГУ закріплення враховано.")

            print("Розв'язування СЛАР...")
            self.U = solver.solve_system(self.MG, self.F)
            
            if self.U is None or np.any(np.isnan(self.U)): 
                 self.results_label.config(text="Помилка розв'язку СЛАР. Перевірте ГУ та сітку.")
                 messagebox.showerror("Помилка розв'язку", "Не вдалося розв'язати СЛАР. Можливі причини: недостатнє закріплення, вироджена сітка.")
                 self.deformed_AKT = None 
                 self._update_visualization() 
                 return

            print("СЛАР розв'язано.")
            
            print("Обчислення деформованих координат...")
            self.deformed_AKT = self.AKT.copy()
            num_nodes_to_process = min(self.nqp, self.U.shape[0] // 3)
            for node_idx_0based in range(num_nodes_to_process):
                u_x_idx, u_y_idx, u_z_idx = 3*node_idx_0based, 3*node_idx_0based+1, 3*node_idx_0based+2
                if u_z_idx < self.U.shape[0]: 
                    self.deformed_AKT[0, node_idx_0based] += -self.U[u_x_idx] 
                    self.deformed_AKT[1, node_idx_0based] += -self.U[u_y_idx] 
                    self.deformed_AKT[2, node_idx_0based] += -self.U[u_z_idx] 
            print("Деформовані координати обчислено (зі зміною знаку U).")

            print("Обчислення напружень (ЗАГЛУШКА)...") 
            self.SIGMA = np.zeros((self.nqp, 6))    
            self.SIGMA_P = np.zeros((self.nqp, 3))  

            print("Debug: Calling _update_array_combo_list from _solve_callback (before results)...")
            self._update_array_combo_list() 
            print("Debug: Calling _update_visualization from _solve_callback (before results)...")
            self._update_visualization()    
            
            max_U_abs = np.max(np.abs(self.U)) if self.U is not None and self.U.size > 0 else 0.0
            results_text = f"Розрахунок успішно завершено.\nМакс. абс. переміщення: {max_U_abs:.6e}"
            self.results_label.config(text=results_text)
            
            if max_U_abs < 1e-12 and (self.F is not None and np.any(np.abs(self.F) > 1e-9)): 
                messagebox.showwarning("Результат розрахунку", "Розрахунок завершено, але переміщення близькі до нуля. Перевірте навантаження, ГУ та параметри матеріалу.")
            else:
                messagebox.showinfo("Розрахунок завершено", results_text)

        except ValueError as e:
            self.results_label.config(text=f"Помилка даних: {e}")
            messagebox.showerror("Помилка даних", f"{e}")
            print(f"ПОМИЛКА ДАНИХ в _solve_callback: {e}")
            traceback.print_exc()
        except MemoryError as e:
            self.results_label.config(text=f"Помилка пам'яті: {e}")
            messagebox.showerror("Помилка пам'яті", f"Недостатньо пам'яті: {e}.")
            print(f"ПОМИЛКА ПАМ'ЯТІ в _solve_callback: {e}")
            traceback.print_exc()
        except Exception as e: 
            self.results_label.config(text=f"Загальна помилка: {e}")
            messagebox.showerror("Помилка розрахунку", f"Непередбачена помилка: {e}", detail=traceback.format_exc())
            print(f"ЗАГАЛЬНА ПОМИЛКА в _solve_callback: {e}")
            traceback.print_exc()
        finally:
            print("--- Розрахунок МСЕ завершено (або перервано) ---")


    def _update_array_combo_list(self):
         print("Debug: Entering _update_array_combo_list")
         base_list = ["AKT", "NT"]
         if self.deformed_AKT is not None:
             base_list.append("deformed_AKT")
         prev_selection = self.selected_array_var.get()
         
         if self.ZU is not None and self.ZU.size > 0 : base_list.append("ZU")
         if self.ZP is not None and self.ZP.size > 0 : base_list.append("ZP")
         
         if self.DFIABG is not None: base_list.append("DFIABG")
         
         if self.DFIXYZ and any(val is not None and val.size > 0 for val in self.DFIXYZ.values()):
             base_list.append("DFIXYZ (elem ...)")
         if self.DJ and any(val is not None and val.size > 0 for val in self.DJ.values()):
             base_list.append("DJ (elem ...)")
         if self.MGE and any(val is not None and val.size > 0 for val in self.MGE.values()):
             base_list.append("MGE (elem ...)")
         if self.FE and any(val is not None and val.size > 0 for val in self.FE.values()):
             base_list.append("FE (elem ...)")

         if self.MG is not None: base_list.append("MG (Global Stiffness)")
         if self.F is not None: base_list.append("F (Global Load)")
         if self.U is not None: base_list.append("U (Displacements)")
         
         self.array_combo['values'] = base_list
         if prev_selection in base_list: self.selected_array_var.set(prev_selection)
         elif base_list: self.selected_array_var.set(base_list[0])
         else: self.selected_array_var.set("")
         self._update_array_params_visibility()
         print("Debug: Exiting _update_array_combo_list")


if __name__ == "__main__":
    print("Запуск програми МСЕ Аналізу...")
    app = FEMApp()
    app.mainloop()
