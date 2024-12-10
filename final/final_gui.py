import tkinter as tk
from tkinter import messagebox
from circuit_class import Circuit
import customtkinter as ctk
import skrf as rf
from tkinter import filedialog, messagebox  # Importamos messagebox desde tkinter
import numpy as np
import sys
import os
from Graficas import Vizualizer

#Función que selecciona entre cargar o no cargar el archivo s2p
def toggle_options():
    global nodos_label, nodos_entry, toggle_sel
    if s2p_file_chckbx.get():
        messagebox.showinfo("INFO", f"Esta funcion esta limitada a solo 2 puertos")
        toggle_sel = 1
        hide_info()
        nodos_label = ctk.CTkLabel(frame, text="Ingresa los nodos del s2p:")
        nodos_entry = ctk.CTkEntry(frame)
        nodos_label.grid(row=5, column=0, padx=10, pady=10)
        nodos_entry.grid(row=5, column=1, padx=10, pady=10)
        load_s2p_file()
    else:
        toggle_sel = 0

    #Ajustar tamaño automáticamente después de cambiar el contenido
    ventana.update_idletasks()

#Funcion para ocultar los datos de inf general
def hide_info():
    """Oculta las entradas adicionales."""
    init_freq_label.grid_remove()
    init_freq_entry.grid_remove()
    final_freq_label.grid_remove()
    final_freq_entry.grid_remove()
    steps_freq_label.grid_remove()
    steps_freq_entry.grid_remove()
    charac_imp_label.grid_remove()
    charac_imp_entry.grid_remove()
    submit_gnl_inf_button.grid_remove()

#Función para cargar archivo s2p
def load_s2p_file():
    
    global archivo_s2p
    
    # Usar filedialog para seleccionar el archivo
    archivo_s2p = filedialog.askopenfilename(filetypes=[("Archivos S2P", "*.s2p")])
    
    if archivo_s2p:
        messagebox.showinfo("Archivo Cargado", f"El archivo {archivo_s2p} se ha cargado correctamente.")
    else:
        messagebox.showerror("Error", "No se seleccionó ningún archivo.")
                           
    # Botón para enviar la información general
    submit_gnl_inf_button = ctk.CTkButton(frame, text="Enviar", command=submit_grl_info)
    submit_gnl_inf_button.grid(row=7, column=0, columnspan=2, pady=10)

#Función para leer un archivo s2p
def read_s2p(filename):
    """
    Lee un archivo .s2p y extrae la frecuencia, los parámetros S y la resistencia de referencia.
    
    Args:
        filename (str): Ruta del archivo .s2p.
    
    Returns:
        tuple: 
            - freq (numpy.ndarray): Vector de frecuencias.
            - s_params (numpy.ndarray): Matriz de parámetros S (complejos).
            - z_ref (float): Resistencia de referencia.
    """
    frequencies = []
    s_parameters = []
    z_ref = None  # Inicializa la resistencia de referencia

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Ignora comentarios y líneas vacías
                if line.startswith('!') or line.strip() == '':
                    continue
                
                # Procesa la línea de encabezado
                if line.startswith('#'):
                    header_parts = line.split()
                    try:
                        freq_type = (header_parts[1])
                        param_type = (header_parts[2])
                        result_type = (header_parts[3])
                        z_ref = float(header_parts[5])
                    except (IndexError, ValueError):
                        raise ValueError("El formato de la línea de encabezado no es válido.")
                    continue
                
                if freq_type == 'GHz':
                    freq_multiplier = 1e9
                elif freq_type == 'MHz':
                    freq_multiplier = 1e6
                elif freq_type == 'KHz':
                    freq_multiplier = 1e3
                elif freq_type == 'Hz':
                    freq_multiplier = 1
                else:
                    raise ValueError("Unidad de frecuencia no válida.")
                    
                # Procesa los datos de los parámetros S
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        freq = freq_multiplier*float(parts[0])
                        s11 = complex(float(parts[1]), float(parts[2]))
                        s21 = complex(float(parts[3]), float(parts[4]))
                        s12 = complex(float(parts[5]), float(parts[6]))
                        s22 = complex(float(parts[7]), float(parts[8]))
                    except ValueError:
                        raise ValueError("El formato de los datos de parámetros S no es válido.")
                    
                    #Matriz S para determinada frecuencia
                    matriz_s = np.array([[s11, s12], [s21, s22]], dtype=complex)
                    
                    #Guardar la frecuencia y la matriz
                    frequencies.append(freq)
                    s_parameters.append(matriz_s)

        # Verifica que se haya leído la resistencia de referencia
        if z_ref is None:
            raise ValueError("No se encontró la resistencia de referencia en el archivo.")
        
        steps = frequencies[1] - frequencies[0]
        return frequencies, steps, s_parameters, z_ref, freq_type, param_type, result_type
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {filename}")
    except Exception as e:
        raise RuntimeError(f"Ocurrió un error al procesar el archivo: {e}")

def es_numero(valor):
    try:
        float(valor)
        return True
    except ValueError:
        return False

# Función guarda la información general del circuito
def submit_grl_info():
    global box_list, i_freq, f_freq, s_freq, i_impedance, comp, input_nodes_list, s2p_freq, s2p_steps, s2p_params, s2p_z_ref, s2p_freq_type, s2p_param_type, s2p_result_type, nodos_s2p_lista 
        

    comp = int(component_number_entry.get())      # guarda número de los componentes
    input_nodes = input_nodes_entry.get()    # guarda los nodos de entrada
    input_nodes_list = input_nodes.strip().split()  
    
    if toggle_sel == 0:
        i_freq = init_freq_entry.get()           # guarda la frecuencia inicial
        f_freq = final_freq_entry.get()          # guarda la frecuencia final
        s_freq = steps_freq_entry.get()          # guarda los pasos que se darán
        i_impedance = charac_imp_entry.get()     # guarda el valor de la impedancia característica
        s2p_params = None
        nodos_s2p_lista = None
    elif toggle_sel == 1:
        s2p_freq, s2p_steps, s2p_params, s2p_z_ref, s2p_freq_type, s2p_param_type, s2p_result_type = read_s2p(archivo_s2p)
        i_freq = s2p_freq[0]
        f_freq = s2p_freq[-1]
        s_freq = s2p_steps
        i_impedance = s2p_z_ref
        nodos = nodos_entry.get()
        nodos_s2p_lista = nodos.split()
        if (len(nodos_s2p_lista) == 2 or len(nodos_s2p_lista) == 3) and all(i.isdigit() for i in nodos_s2p_lista):
            messagebox.showinfo("Información", f"Archivo S2P: {archivo_s2p}\nNodos: {nodos_s2p_lista}")
        else:
            messagebox.showerror("Error", "Por favor, ingresa tres nodos numéricos válidos separados por espacio.")
    else:
        messagebox.showerror("ERROR", "Algo salió mal")
        
    try:
        input_nodes_list = [int(i) for i in input_nodes_list]
    except ValueError:
        messagebox.showerror("ERROR", "Los nodos de entrada deben ser números enteros, no más de un espacio entre ellos")
        return

    
    # Verificar si todos los campos están completos, si no manda un error
    if toggle_sel == 0:
        if not all([comp, input_nodes, i_freq, i_impedance, f_freq, s_freq]):
            messagebox.showwarning("ERROR", "Campos incompletos")
            return False
        # Validar que las frecuencias y la impedancia sean números válidos
        if not all([es_numero(i_freq), es_numero(f_freq), es_numero(s_freq), es_numero(i_impedance)]):
            messagebox.showwarning("ERROR", "Las entradas deben ser números enteros")
            return
    elif toggle_sel == 1:
        if not all([input_nodes]):
            messagebox.showwarning("ERROR", "Campos incompletos!")
            print(input_nodes)
            return
        if not all([es_numero(i_freq), es_numero(f_freq), es_numero(s_freq), es_numero(i_impedance)]):
            messagebox.showwarning("ERROR", "Las entradas deben ser números enteros")
            return
    else:
        messagebox.showerror("ERROR", "Algo salió mal")   

    # Encabezado del frame
    header_label = ctk.CTkLabel(frame, text="REGISTRO DE COMPONENTES", font=("Arial", 14, "bold"))
    header_label.grid(row=10, column=0, columnspan=12, pady=10)

    # Crear las etiquetas y cajas de texto de acuerdo al número de componentes registrados
    box_list = []  # Limpiar la lista de cajas antes de agregar nuevas
    for i in range(comp):
        label = ctk.CTkLabel(frame, text=f"Componente {i+1}:")
        label.grid(row=i + 11, column=0, sticky="w", padx=5, pady=5)
        box = ctk.CTkEntry(frame)
        box.grid(row=i + 11, column=1, padx=5, pady=5)
        box_list.append(box)

    submit_comp_inf_button = ctk.CTkButton(frame, text="Simular", command=save_comp_inf)
    submit_comp_inf_button.grid(row=10, column=2, padx=10, pady=10)
    submit_comp_inf_button.configure(width=10, height=1)


# Función para guardar la información de los componentes
def save_comp_inf():
    components_info = []

    # Procesar cada entrada y dividirla en partes
    for i in box_list:
        texto = i.get()
        partes = texto.split()

        # Verificar si la entrada tiene 4 partes
        if partes[0] in ['R', 'L', 'C', 'S', 'O', 'T']:
            if len(partes) == 4:
                components_info.append([partes[0], float(partes[1]), int(partes[2]), int(partes[3])])
            elif len(partes) == 3:
                components_info.append([partes[0], float(partes[1]), int(partes[2])])
            else:
                messagebox.showerror("ERROR", "Descripción del componente inválida, Ejemplo: C 1e4 1 2")
        else:
            messagebox.showerror("ERROR", "Describir el tipo de componente (R, L, C, S, O, T)")

    # Mostrar la información guardada
    if components_info or toggle_sel:
        messagebox.showinfo("INFO", f"Se han guardado los siguientes componentes:\n{components_info}")

        circuit = Circuit(components=components_info,
                          input_nodes=input_nodes_list,
                          lower_freq_limit=float(i_freq),
                          upper_freq_limit=float(f_freq),
                          freq_step=float(s_freq),
                          z_charac=float(i_impedance),
                          s2p_device = s2p_params,
                          s2p_nodes = nodos_s2p_lista,
                          )
        try:
            matrix, frequencies, z_ref = circuit.run_simulation()
        except Exception as e:
            messagebox.showerror("ERROR", f"Ocurrió un error al simular el circuito: {e}")
        else:
            messagebox.showinfo("INFO", f"Se ha simulado el circuito correctamente")
            Vizualizer(matrix, frequencies, z_ref)
    else:
        messagebox.showwarning("Advertencia", "No se ha ingresado suficiente información.")
        return
    
def reset_script():
    python = sys.executable
    os.execl(python, python, *sys.argv)

# Configuración de estilo de CustomTkinter
ctk.set_appearance_mode("System")  # Modo de apariencia: "Light", "Dark", "System"
ctk.set_default_color_theme("blue")  # Tema de color: "blue", "dark-blue", "green"

# Principal Window
ventana = ctk.CTk()
ventana.title("Simulador de Circuitos - Ing Microondas I")
ventana.geometry("470x550")
ventana.resizable(True, True)  # Habilitar redimensionamiento

# Crear el marco principal
frame = ctk.CTkFrame(ventana)
frame.pack(padx=20, pady=20, fill="both", expand=True)  # Usar pack en el marco principal

# Encabezado del frame
general_info_label = ctk.CTkLabel(frame, text="INFORMACIÓN GENERAL DEL CIRCUITO", font=("Arial", 14, "bold"))
general_info_label.grid(row=0, column=0, columnspan=2, pady=10)

# Crear el checkbox
s2p_file_chckbx = ctk.BooleanVar()
chckbx_si = ctk.CTkCheckBox(frame, text="¿Desea cargar .s2p?", variable=s2p_file_chckbx, command=toggle_options)
chckbx_si.grid(row=1, column=1, sticky="w", pady=5)
toggle_sel = 0

component_number_label = ctk.CTkLabel(frame, text="  Número de componentes")
component_number_entry = ctk.CTkEntry(frame)

input_nodes_label = ctk.CTkLabel(frame, text="  Nodos donde se asignan\n los puertos")
input_nodes_entry = ctk.CTkEntry(frame)

init_freq_label = ctk.CTkLabel(frame, text="  Frecuencia inicial (Hz)")
init_freq_entry = ctk.CTkEntry(frame)

final_freq_label = ctk.CTkLabel(frame, text="  Frecuencia final (Hz)")
final_freq_entry = ctk.CTkEntry(frame)

steps_freq_label = ctk.CTkLabel(frame, text="  Pasos de frecuencia (Hz)")
steps_freq_entry = ctk.CTkEntry(frame)

charac_imp_label = ctk.CTkLabel(frame, text="  Impedancia característica (Ω)")
charac_imp_entry = ctk.CTkEntry(frame)

component_number_label.grid(row=3, column=0, sticky="w", pady=5)
component_number_entry.grid(row=3, column=1, padx=10, pady=5)

input_nodes_label.grid(row=4, column=0, sticky="w", pady=5)
input_nodes_entry.grid(row=4, column=1, padx=10, pady=5)

init_freq_label.grid(row=5, column=0, sticky="w", pady=5)
init_freq_entry.grid(row=5, column=1, padx=10, pady=5)

final_freq_label.grid(row=6, column=0, sticky="w", pady=5)
final_freq_entry.grid(row=6, column=1, padx=10, pady=5)

steps_freq_label.grid(row=7, column=0, sticky="w", pady=5)
steps_freq_entry.grid(row=7, column=1, padx=10, pady=5)

charac_imp_label.grid(row=8, column=0, sticky="w", pady=5)
charac_imp_entry.grid(row=8, column=1, padx=10, pady=5)

submit_gnl_inf_button = ctk.CTkButton(frame, text="Enviar", command=submit_grl_info)
submit_gnl_inf_button.grid(row=9, column=0, columnspan=2, pady=10)

reset_button = tk.Button(frame, text="Reset", command=reset_script)
reset_button.grid(row=0, column=2, padx=10, pady=10)

ventana.mainloop()
