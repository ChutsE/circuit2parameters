import numpy as np
import skrf as rf
import cmath
import matplotlib.pyplot as plt

def t_to_s_matrix(t_matrix):
    """
    Convert T-parameters matrix to S-parameters matrix.
    
    Parameters:
    t_matrix (numpy.ndarray): 2x2 T-parameters matrix
    
    Returns:
    numpy.ndarray: 2x2 S-parameters matrix
    """
    t11, t12, t21, t22 = t_matrix[0, 0], t_matrix[0, 1], t_matrix[1, 0], t_matrix[1, 1]
    
    s_matrix = np.zeros((2, 2), dtype=complex)
    s_matrix[0, 0] = t12
    s_matrix[0, 1] = np.linalg.det(t_matrix)
    s_matrix[1, 0] = 1/10
    s_matrix[1, 1] = -t21
    
    return s_matrix/t22

def s_to_t_matrix(s_matrix):
    """
    Convert S-parameters matrix to T-parameters matrix.
    
    Parameters:
    s_matrix (numpy.ndarray): 2x2 S-parameters matrix
    
    Returns:
    numpy.ndarray: 2x2 T-parameters matrix
    """
    s11, s12, s21, s22 = s_matrix[0, 0], s_matrix[0, 1], s_matrix[1, 0], s_matrix[1, 1]
    
    t_matrix = np.zeros((2, 2), dtype=complex)
    t_matrix[0, 1] = s11
    t_matrix[1, 0] = -s22 
    
    t_matrix[0, 0] = -np.linalg.det(s_matrix)  
    t_matrix[1, 1] = 1
    
    return t_matrix / s21

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
                
                # Procesa los datos de los parámetros S
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        freq = float(parts[0])
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

        return np.array(s_parameters), np.array(frequencies), z_ref
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {filename}")
    except Exception as e:
        raise RuntimeError(f"Ocurrió un error al procesar el archivo: {e}")

def resolver_ecuacion_cuadratica_compleja(a, b, c):
    """
    Solve the quadratic equation ax^2 + bx + c = 0 for complex coefficients.
    
    Parameters:
    a, b, c (complex): Coefficients of the quadratic equation
    
    Returns:
    tuple: Solutions x1 and x2
    """
    discriminant = cmath.sqrt(b**2 - 4*a*c)
    x1 = (-b + discriminant) / (2*a)
    x2 = (-b - discriminant) / (2*a)
    
    return x1, x2

def convertir_matriz(matriz):
    """
    Convert a given matrix to a new matrix and a scalar g.
    
    Parameters:
    matriz (numpy.ndarray): Input matrix
    
    Returns:
    tuple: New matrix and scalar g
    """
    new_mat = np.zeros((2, 2), dtype=complex)
    new_mat = matriz 
    
    return new_mat/matriz[1,1], matriz[1,1]

def menor_complejo(c1, c2):
    """
    Determine the smaller of two complex numbers based on their magnitudes.
    
    Parameters:
    c1, c2 (complex): Two complex numbers to compare
    
    Returns:
    complex: The complex number with the smaller magnitude
    """
    return c1 if abs(c1) < abs(c2) else c2

def mayor_complejo(c1, c2):
    """
    Determine the larger of two complex numbers based on their magnitudes.
    
    Parameters:
    c1, c2 (complex): Two complex numbers to compare
    
    Returns:
    complex: The complex number with the larger magnitude
    """
    return c1 if abs(c1) > abs(c2) else c2

def matriz_errorB(g, mat, a_c, b):
    d = mat[0, 0]
    e = mat[0, 1]
    f = mat[1, 0]
    c_a = 1 / a_c
    
    fi = ((a_c * f) - d) / (a_c - e)
    r_ro = (g * (a_c - e)) / (a_c - b)
    bt_alph = (e - b) / (d - (b * f))
    a_alph = (d - (b * f)) / (1 - (c_a * e))
    
    return fi, r_ro, bt_alph, a_alph

def calcular_a(mat, mat_open1, mat_open2, fi, bt_alph, a_c, b):
    d = mat[0, 0]
    e = mat[0, 1]
    f = mat[1, 0]
    
    w1 = mat_open1[0, 0]
    w2 = mat_open2[1, 1]
    
    c_a = 1 / a_c
    
    a1_num = (d - (b * f)) * (b - w1) * (1 + (bt_alph * w2))
    a1_den = (1 - (c_a * e)) * (fi + w2) * ((c_a * w1) - 1)
    
    a1 = cmath.sqrt(a1_num / a1_den) 
    a2 = -1 * cmath.sqrt(a1_num / a1_den)   
    
    return a1, a2

def calcular_aest(b, a_c, mat):
    w1 = mat[0, 0]
    c_a = 1 / a_c
    aest = (b - w1) / (c_a - 1)
    return aest

def calcular_c(a, a_c):
    return a / a_c

def calcular_alpha(mat, a, b, c):
    d = mat[0, 0]
    e = mat[0, 1]
    f = mat[1, 0]
    alph = (d - (b * f)) / (a - (c * e))
    return alph

def calcular_betha(mat, a, b, c):
    e = mat[0, 1]
    betha = (e - b) / (a - (c * e))
    return betha

def calcular_dut(a1, b, c, alph, betha, fi, r_ro, t_med):
    """
    Calculate the S-parameters of the DUT.
    
    Parameters:
    a1, b, c, alph, betha, fi, r_ro (complex): Coefficients
    t_med (numpy.ndarray): Measured T-parameters matrix
    
    Returns:
    numpy.ndarray: T-parameters matrix of the DUT
    """
    t11 = t_med[0, 0]
    t12 = t_med[0, 1]
    t21 = t_med[1, 0]
    t22 = t_med[1, 1]
    
    den = ((c * betha * t11) - (c * alph * t12) - (a1 * betha * t21) + (a1 * alph * t22))
    
    s12_num = r_ro * (a1 - b * c) * (alph - betha * fi) * (t11 - fi * t12 - b * t21 + b * fi * t22)
    s12_den = c * betha * t11 - c * alph * t12 - a1 * betha * t21 + a1 * alph * t22
    
    s11 = (r_ro * (a1 - b * c) * (alph - betha * fi) * (alph * t12 - betha * t11 + b * betha * t21 - b * alph * t22)) / den
    s22 = -(r_ro * (a1 - b * c) * (alph - betha * fi) * (c * fi * t12 - c * t11 + (a1 * (t21 - (fi * t22))))) / den
    s21 = (r_ro * (a1 - b * c) * (alph - betha * fi)) / den
    s12 = ((s12_num / s12_den) + s11 * s22) / s21
    
    return np.array([[s11, s12], [s21, s22]], dtype=complex)

def calcular_dut1(t_med, a1, b, a_c):
    t11 = t_med[0, 0]
    t12 = t_med[0, 1]
    t21 = t_med[1, 0]
    t22 = t_med[1, 1]
    
    s12_num = a_c*(t11-(b*t21))-(b*t22)+t12
    s12_den = a_c*((b*t21)+t22)-(b*t11)-t12
    
    s11 = (a_c*((b*t11)+t12-(b*(b*t21+t22))))/(a1*(a_c*((b*t21)+t22)-(b*t11)-t12))
    s21 = (a_c-b)/(a_c*((b*t21)+t22)-(b*t11)-t12)
    
    s22 = -(a1 * (((a_c ** 2) * t21) + (a_c * (t22 - t11)) - t12)) / (a_c * (a_c * (b * t21 + t22) - (b * t11) - t12))
    s12 = ((s12_num/s12_den)+(s11*s22))/(s21)
    #(((a_c*(t11-(b*t21))-b*t22+t12)/(a_c*(b*t21+t22))-b*t11-t12)+s11*s22)/(a1*(a_c*(b*t21+t22)-b*t11-t12))
    return np.array([[s11, s12], [s21, s22]], dtype=complex)
    
def guardar_s2p(frecuencias, parametros_S, archivo_s2p, z_ref):
    """
    Función para guardar parámetros S en formato Touchstone (.s2p).
    
    Args:
    - frecuencias (list): Lista de frecuencias en GHz.
    - parametros_S (list): Lista de arrays de parámetros S (cada array tiene 2x2 elementos).
    - archivo_s2p (str): Nombre del archivo de salida (.s2p).
    - z_ref (float): Resistencia de referencia.
    """
    
    # Asegúrate de que la lista de parámetros S y las frecuencias tengan la misma longitud
    if len(frecuencias) != len(parametros_S):
        raise ValueError("La longitud de las frecuencias debe ser igual a la longitud de los parámetros S.")
    
    # Abrir el archivo en modo de escritura
    with open(archivo_s2p, 'w') as file:
        # Escribir la cabecera del archivo Touchstone (.s2p)
        file.write("! Touchstone file saved by Python\n")
        file.write(f"# Hz  S RI R:{z_ref}\n")
        
        # Recorrer las frecuencias y parámetros S
        for i in range(len(frecuencias)):
            f = int(frecuencias[i])  # Frecuencia en Hz
            S = parametros_S[i]  # Matriz de parámetros S (2x2)
            
            # Extraer los valores reales e imaginarios de los parámetros S
            S11_real, S11_imag = S[0, 0].real, S[0, 0].imag
            S21_real, S21_imag = S[1, 0].real, S[1, 0].imag
            S12_real, S12_imag = S[0, 1].real, S[0, 1].imag
            S22_real, S22_imag = S[1, 1].real, S[1, 1].imag
            
            # Escribir los parámetros S para la frecuencia actual en el archivo
            file.write(f"{f:.6f}   {S11_real:.6f}   {S11_imag:.6f}   {S21_real:.6f}   {S21_imag:.6f}   "
                       f"{S12_real:.6f}   {S12_imag:.6f}   {S22_real:.6f}   {S22_imag:.6f}\n")
    
    print(f"Archivo {archivo_s2p} guardado exitosamente.")

def plot_magnitude_vs_frequency(frequencies, results1, results2, results3):
    """
    Plot the magnitude of S-parameters vs frequency.
    
    Parameters:
    frequencies (numpy.ndarray): Array of frequencies
    results (list): List of dictionaries containing frequency and S-parameters matrix
    """
    s11_magnitudes = [np.angle(result) for result in results1]
    s21_magnitudes = [np.angle(result) for result in results2]
    s22_magnitudes = [np.angle(result) for result in results3]
    
    plt.figure()
    plt.plot(frequencies, s11_magnitudes, label='a1')
    plt.plot(frequencies, s21_magnitudes, label='a2')
    plt.plot(frequencies, s22_magnitudes, label='a')

    plt.legend()
    plt.grid(True)
    plt.show()
    
# Example usage
s_line_file = 'Line.s2p'
s_thru_file = 'Thru.s2p'
s_openp1_file = 'Open_P1.s2p'
s_openp2_file = 'Open_P2.s2p'
s_med_file = 'ATF-38143_Vgs_-0.55_Vds_2.s2p'

s_line, frequencies, z_ref = read_s2p(s_line_file)
s_thru, _, _ = read_s2p(s_thru_file)
s_openp1, _, _ = read_s2p(s_openp1_file)
s_openp2, _, _ = read_s2p(s_openp2_file)
s_med, _, _ = read_s2p(s_med_file)

results = []
a1_plot = []
a2_plot = []
a_plot = []
b_plot = []
a_c_plot = []   

for i in range(len(s_line)):
    t_line = s_to_t_matrix(s_line[i]) #Convertir matriz S a matriz T del line
    t_thru = s_to_t_matrix(s_thru[i]) #Convertir matriz S a matriz T del thru
    
    t_thru_inv = np.linalg.inv(t_thru) #Inversa de la matriz T del thru
    
    t_tot = np.dot(t_line, t_thru_inv) #Multiplicación de las matrices T del line y la inversa del thru
    
    t_med = s_to_t_matrix(s_med[i]) #Convertir matriz S a matriz T del DUT

    #x^2*t21 + y*(t22-t11) - t12*z = 0
    x = t_tot[1, 0]
    y = t_tot[1, 1] - t_tot[0, 0]
    z = -t_tot[0, 1]
    
    #Resuelve con fórmula general la ecuación cuadrática
    x1, x2 = resolver_ecuacion_cuadratica_compleja(x, y, z)
    
    #El valor menor se pasa a b y el mayor a a_c
    b = menor_complejo(x1, x2)
    a_c = mayor_complejo(x1, x2) 
        
    #Convierte la matriz T1 del thru a una nueva matriz y un escalar g
    #                  |d e| 
    #   new_t_thru = g |f 1|
    new_t_thru, g = convertir_matriz(t_thru) 
    
    #Calcula fi, r22_ro22, beta/alpha y a*alpha
    fi, r_ro, bt_alph, a_alph = matriz_errorB(g, new_t_thru, a_c, b)
    w1 = s_openp1[i][0, 0]
    w2 = s_openp2[i][1, 1]
    #Calcula los dos posibles valores de a
    a1, a2 = calcular_a(new_t_thru, s_openp1[i], s_openp2[i], fi, bt_alph, a_c, b)

    aest = calcular_aest(b, a_c, s_openp1[i])
    
    a1_aux = aest - a1
    a2_aux = aest - a2
    
    if abs(a1_aux) < abs(a2_aux):
        a = a1
    else:
        a = a2  
    a1_plot.append(a1)
    a2_plot.append(a2)
    a_plot.append(a)
    
    #Calcula el valor de C
    c = calcular_c(a, a_c)
    
    #Calcula el valor de alpha y betha
    alph = calcular_alpha(new_t_thru, a, b, c)
    betha = calcular_betha(new_t_thru, a, b, c)
    
    Ta = np.array([[a, b], [c, 1]], dtype=complex)
    Tb = np.array([[alph, betha], [fi, 1]], dtype=complex)
    
    Ta_inv = np.linalg.inv(Ta)
    Tb_inv = np.linalg.inv(Tb)
    
    T_aux = np.dot(Ta_inv, t_med)
    
    T_dut = (1/r_ro) * np.dot(T_aux, Tb_inv)
    s_dut = t_to_s_matrix(T_dut)
    
    #Calcula los parámetros S del DUT
    #s_dut = calcular_dut(a, b, c, alph, betha, fi, r_ro, t_med)
    #s_dut = calcular_dut1(t_med, a, b, a_c)
    
    results.append({
        'frequency': frequencies[i],
        's_dut': s_dut
    })

# Save the results to a new s2p file
output_file = 'DUT9.s2p'
guardar_s2p(frequencies, [result['s_dut'] for result in results], output_file, z_ref)
print(a_alph)
print(a_c)
print(a)
print(alph)
print(b)
print(betha)
print(bt_alph)
print(c)
print(new_t_thru[0, 0])
print(new_t_thru[0, 1])
print(new_t_thru[1, 0])
print(g)
print(fi)
print(r_ro)
print(w1)
print(w2)

plot_magnitude_vs_frequency(frequencies, a1_plot, a2_plot, a_plot)
# Print the results
#for result in results:
#    print(f"Frequency: {result['frequency']}")
#    print("S-parameters DUT matrix:\n", result['s_dut'])

