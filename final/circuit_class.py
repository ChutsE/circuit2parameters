import numpy as np

class Circuit:
    
    def __init__(self, components: list, input_nodes: list, lower_freq_limit: float, upper_freq_limit: float, freq_step: float, z_charac: float, s2p_device: list = None, s2p_nodes: list = []):
        self._components = components
        self._input_nodes = input_nodes
        self._frecuency = lower_freq_limit
        self._upper_freq_limit = upper_freq_limit
        self._freq_step = freq_step
        self._z_charac = z_charac
        self._s2p_device = s2p_device
        self._s2p_nodes = s2p_nodes
        self._nodes_matrix = []
        self._circuit_matrix = None
        self._no_in_nodes = None
        self.z_matrix = None
        self.y_matrix = None
        self.abcd_matrix = None
        self.s_matrix = None
        self.__s2p_cnt = 0
    
    def components_divider(self):
        """Divide the components in the circuit."""
        self._s2p_nodes = [int(node) for node in self._s2p_nodes]

        total_nodes_in = self._input_nodes + self._s2p_nodes
        total_nodes_in = set(total_nodes_in)

        node_threshold_1 = self._s2p_nodes[0]
        node_threshold_2 = self._s2p_nodes[1]

        #identify the components in the malla_1 and malla_2
        self.malla_1 = set()
        self.malla_2 = set()
        for component_num, nodes in enumerate(self._components_nodes):
            for node in nodes:
                if node <= node_threshold_1:
                    self.malla_1.add(component_num)
                elif node >= node_threshold_2:
                    self.malla_2.add(component_num)

        
        self.in_nodes_1 = [node for node in total_nodes_in if node <= node_threshold_1]
        self.in_nodes_2 = [node for node in total_nodes_in if node >= node_threshold_2]


        self.components_values_malla_1 = []
        self.components_nodes_malla_1 = []
        for component_num in self.malla_1:
            self.components_values_malla_1.append(self._components_values[component_num])
            self.components_nodes_malla_1.append(self._components_nodes[component_num])
        
        self.components_values_malla_2 = []
        self.components_nodes_malla_2 = []
        for component_num in self.malla_2:
            self.components_values_malla_2.append(self._components_values[component_num])
            self.components_nodes_malla_2.append(self._components_nodes[component_num])

    def y2abcd(self):          
            y11 = self.y_matrix[0,0]
            y12 = self.y_matrix[0,1]
            y21 = self.y_matrix[1,0]
            y22 = self.y_matrix[1,1]
                
            abcd_mat = np.zeros((2,2), dtype=complex)
                
            abcd_mat[0,0] = -y22/y21
            abcd_mat[0,1] = -1/y21
            abcd_mat[1,0] = -(y11*y22-y12*y21)/y21
            abcd_mat[1,1] = -y11/y21
                
            self.abcd_matrix = abcd_mat

    def abcd2y(self):      
        A = self.abcd_matrix[0,0]
        B = self.abcd_matrix[0,1]
        C = self.abcd_matrix[1,0]
        D = self.abcd_matrix[1,1]
       
        y_mat = np.zeros((2,2), dtype=complex)
       
        y_mat[0,0] = D/B
        y_mat[0,1] = (B*C-A*D)/B
        y_mat[1,0] = -1/B
        y_mat[1,1] = A/B  

        self.y_matrix = y_mat
   
    def s2p_to_components(self):

        Za, Zb, Zc = self.__ABCD_2PI(self.__S2ABCD(self._s2p_device[self.__s2p_cnt]))

        if len(self._s2p_nodes) == 3:
            self._components.append(["Z", Za, int(self._s2p_nodes[2]), int(self._s2p_nodes[0])])
            self._components.append(["Z", Zb, int(self._s2p_nodes[0]), int(self._s2p_nodes[1])])
            self._components.append(["Z", Zc, int(self._s2p_nodes[1]), int(self._s2p_nodes[2])])
        elif len(self._s2p_nodes) == 2:
            self._components.append(["Z", Za, int(self._s2p_nodes[0])])
            self._components.append(["Z", Zb, int(self._s2p_nodes[0]), int(self._s2p_nodes[1])])
            self._components.append(["Z", Zc, int(self._s2p_nodes[1])])
        self.__s2p_cnt += 1

    def __ABCD_2T(self, param):
        [A, B, C, D] = param

        Zc = complex(1 / C)
        Za = complex((A/C) - 1/C)
        Zb = complex((D/C) - 1/C)

        return Za, Zb, Zc

    def __ABCD_2PI(self, param):

        [A, B, C, D] = param
        
        Yc = complex(1 / B)
        Ya = complex((D/B) - 1/B)
        Yb = complex((A/B) - 1/B)
        
        Za = 1/Ya
        Zb = 1/Yb
        Zc = 1/Yc

        return Za, Zb, Zc
    
    def __S2ABCD(self, s_matrix):

        s11 = s_matrix[0,0]
        s12 = s_matrix[0,1]
        s21 = s_matrix[1,0]
        s22 = s_matrix[1,1]
        
        abcd_mat = np.zeros((2,2), dtype=complex)
        
        denom = 2 * s21
            
        abcd_mat[0,0] = complex(((1+s11) * (1-s22) + (s12*s21)) / (denom))
        abcd_mat[0,1] = complex(self._z_charac * ((1+s11) * (1+s22) - (s12*s21)) / (denom))
        abcd_mat[1,0] = complex((1/self._z_charac) * ((1-s11) * (1-s22) - (s12*s21)) / (denom)) 
        abcd_mat[1,1] = complex(((1-s11) * (1+s22) + (s12*s21)) / (denom))
        
        return abcd_mat
    
    def impedance_calculator(self):
        """Convert input components to components_values and components_nodes."""
        self._components_nodes = []
        self._components_values = []
        for component in self._components:
            type_, value, *nodes = component
            if type_ in ["R", "Z"]:
                self._components_values.append(value)
            elif type_ == "C":
                self._components_values.append(-1j / (2 * np.pi * self._frecuency * value))
            elif type_ == "L":
                self._components_values.append(1j * 2 * np.pi * self._frecuency * value)
            elif type_ == "S":
                relative_permitivity = 4.6 # Relative permitivity of the substrate FR4
                Beta = 2 * np.pi * self._frecuency * np.sqrt(relative_permitivity)/ 3e8
                self._components_values.append(1j * self._z_charac * np.tan(Beta * value))
            elif type_ == "O":
                relative_permitivity = 4.6 # Relative permitivity of the substrate FR4
                Beta = 2 * np.pi * self._frecuency * np.sqrt(relative_permitivity)/ 3e8
                self._components_values.append(-1j * self._z_charac / np.tan(Beta * value))
                
            self._components_nodes.append(sorted(nodes))

    def serial_paralel_reduction(self):
        """Find the equivalent circuit for a circuit."""
        parallel_components_set = self.__paralel_branch_finder()
        serial_components_set = self.__serial_branch_finder()

        while parallel_components_set or serial_components_set:
            if parallel_components_set:
                self.__parallel_sum(parallel_components_set)
                parallel_components_set = self.__paralel_branch_finder()
                serial_components_set = self.__serial_branch_finder()
            if serial_components_set:
                self.__serial_sum(serial_components_set)
                parallel_components_set = self.__paralel_branch_finder()
                serial_components_set = self.__serial_branch_finder()

    def __paralel_branch_finder(self) -> list:
        """Find parallel branches in a circuit."""

        components_frequency = {}

        for i, component_nodes in enumerate(self._components_nodes):
            component_tuple = tuple(component_nodes)
            if component_tuple in components_frequency:
                components_frequency[component_tuple].append(i)
            else:
                components_frequency[component_tuple] = [i]

        return [indices for indices in components_frequency.values() if len(indices) > 1]

    def __serial_branch_finder(self) -> list:
        """Find serial branches in a circuit."""
        
        nodes_frequency = {}
        for component_nodes in self._components_nodes:
            for node in component_nodes:
                nodes_frequency[node] = nodes_frequency.get(node, 0) + 1
        #todos los nodos en serie
        serial_nodes = [num for num, count in nodes_frequency.items() if count == 2]

        #elimina los nodos de entrada
        serial_nodes = [i for i in serial_nodes if i not in self._input_nodes]
        serial_components_set = []
        for serial_node in serial_nodes:
            serial_components = []
            for component, component_nodes in enumerate(self._components_nodes):
                if serial_node in component_nodes:
                    serial_components.append(component)
            serial_components_set.append(serial_components)

        unified = []
        for pair in serial_components_set:
            found = None
            for group in unified:
                if any(item in group for item in pair):
                    found = group
                    break
            if found:
                found.update(pair)
            else:
                unified.append(set(pair))

        return [list(group) for group in unified]

    def __serial_sum(self, serial_components_set: list):
        """Sum the serial components in a circuit."""

        components_to_delete = []
        for components in serial_components_set:
            sum_value = sum(self._components_values[component] for component in components)
            nodes_join = [node for component in components for node in self._components_nodes[component]]
            new_component_nodes = [x for x in nodes_join if nodes_join.count(x) == 1]

            self._components_nodes[components[-1]] = new_component_nodes
            self._components_values[components[-1]] = sum_value
            components_to_delete.extend(components[:-1])

        self._components_values = [v for i, v in enumerate(self._components_values) if i not in components_to_delete]
        self._components_nodes = [n for i, n in enumerate(self._components_nodes) if i not in components_to_delete]

    def __parallel_sum(self, parallel_components_set: list):
        """Sum the parallel components in a circuit."""

        components_to_delete = []
        for components in parallel_components_set:
            sum_value = sum(1 / self._components_values[component] for component in components)
            self._components_values[components[-1]] = 1 / sum_value
            components_to_delete.extend(components[:-1])

        self._components_values = [v for i, v in enumerate(self._components_values) if i not in components_to_delete]
        self._components_nodes = [n for i, n in enumerate(self._components_nodes) if i not in components_to_delete]

    def components_to_node(self):
        """Convert the components to nodes."""
        nodes = []
        max_node_num = max(max(component) for component in self._components_nodes) + 1

        for node_num in range(max_node_num):
            node = [component_num for component_num, component in enumerate(self._components_nodes) if node_num in component]
            if node_num in self._input_nodes:
                node.append(f"In_{node_num}")

            nodes.append(node)

        self._nodes_matrix = [node for node in nodes if node]

    def get_circuit_matrix(self):
        """Calculate the circuit matrix for a circuit."""

        circuit_matrix_len = len(self._nodes_matrix)
        self._circuit_matrix = np.zeros((circuit_matrix_len, circuit_matrix_len), dtype=complex)
        
        for j in range(circuit_matrix_len):
            for i in range(circuit_matrix_len):
                if i == j:
                    acc = complex(0, 0)
                    for component_name in self._nodes_matrix[j]:
                        if not isinstance(component_name, str):
                            acc += 1 / self._components_values[component_name]
                    self._circuit_matrix[j, i] = acc

        self._in_nodes = []
        for j, node in enumerate(self._nodes_matrix):
            for component_name in node:
                if isinstance(component_name, str):
                    self._in_nodes.append(j)
                    continue
                node_num = self.__finder(j, component_name, self._nodes_matrix)
                if node_num > -1:
                    self._circuit_matrix[j, node_num] = -1 / self._components_values[component_name]
    
    def __finder(self, row: int, component_name: str, nodes: np.ndarray) -> int:
        """This function find the component pair and returns the node number where it was allocated.

        Args:
            row (int): Current row of circuit matrix .
            nodes (np.ndarray): nodes matrix.
            component_name (str): component name want to find it.

        Returns:
            int: Node number where component pair was found, -1 if didn't find it.
        """
        for i, node in enumerate(nodes):
            if (i != row) and (component_name in node):
                return i
        return -1

    def get_y_matrix(self):
        """Calculate the Y matrix for a circuit."""
        
        total_nodes = set(range(len(self._circuit_matrix)))
        self._no_in_nodes = list(total_nodes - set(self._in_nodes)) 
        
        while len(self._no_in_nodes) > 0:
            self.__matrix_reduction()
            self._no_in_nodes = [n - 1 for n in self._no_in_nodes[1:]]
        
        self.y_matrix = self._circuit_matrix

    def __matrix_reduction(self):
        """Reduce a circuit matrix by eliminating the row and column corresponding to the given node."""

        node = self._no_in_nodes[0]

        y, x = self._circuit_matrix.shape
        pivote_value = self._circuit_matrix[node, node]
        new_matrix = np.zeros((y - 1, x - 1), dtype=complex)
        new_row = 0
        for j in range(y):
            if j == node:
                continue
            new_col = 0
            for i in range(x):
                if i == node:
                    continue
                new_matrix[new_row, new_col] = (self._circuit_matrix[j, i] 
                                                - self._circuit_matrix[node, i] 
                                                * self._circuit_matrix[j, node] 
                                                / pivote_value)
                new_col += 1
            new_row += 1

        self._circuit_matrix = new_matrix
    
    def y2z(self):
        """Convert Z matrix to Y matrix."""
        det = np.linalg.det(self.y_matrix)
        if det:
            self.z_matrix = np.linalg.inv(self.y_matrix)
        else:
            self.z_matrix = None

    def z2abcd(self):
        """Convert Z matrix to ABCD matrix."""
        if len(self.z_matrix) == 2:
            det_mat = np.linalg.det(self.z_matrix)
            C = 1 / self.z_matrix[0][0]
            D = self.z_matrix[1][1] / self.z_matrix[1][0]
            A = self.z_matrix[0][0] / self.z_matrix[1][0]
            B = det_mat / self.z_matrix[1][0]    
            self.abcd_matrix = np.array([[A, B], [C, D]], dtype=complex)

    def z2s(self):
        """Convert Z matrix to S matrix."""
        if self.z_matrix is None:
            self.s_matrix = None
        else:
            z0_unitary_matrix = self._z_charac * np.eye(len(self.z_matrix), dtype=complex)
            self.s_matrix =  np.dot(np.linalg.inv(self.z_matrix + z0_unitary_matrix), self.z_matrix - z0_unitary_matrix)

        """
        if len(self.z_matrix) == 2:
            s_11 = ((self.z_matrix[0][0] - self._z_charac) * (self.z_matrix[1][1] + self._z_charac) - 
                    (self.z_matrix[0][1] * self.z_matrix[1][0])) / ((self.z_matrix[0][0] + self._z_charac) * 
                    (self.z_matrix[1][1] + self._z_charac) - (self.z_matrix[0][1] * self.z_matrix[1][0]))
            s_12 = (2 * self.z_matrix[0][1] * self._z_charac) / ((self.z_matrix[0][0] + self._z_charac) * 
                    (self.z_matrix[1][1] + self._z_charac) - (self.z_matrix[0][1] * self.z_matrix[1][0]))
            s_21 = (2 * self.z_matrix[1][0] * self._z_charac) / ((self.z_matrix[0][0] + self._z_charac) * 
                    (self.z_matrix[1][1] + self._z_charac) - (self.z_matrix[0][1] * self.z_matrix[1][0]))
            s_22 = ((self.z_matrix[0][0] + self._z_charac) * (self.z_matrix[1][1] - self._z_charac) - 
                    (self.z_matrix[0][1] * self.z_matrix[1][0])) / ((self.z_matrix[0][0] + self._z_charac) * 
                    (self.z_matrix[1][1] + self._z_charac) - (self.z_matrix[0][1] * self.z_matrix[1][0]))
            self.s_matrix = np.array([[s_11, s_12], [s_21, s_22]], dtype=complex)
        """

    def run_simulation(self):
        """Run the circuit simulation."""
        frequencies = []
        matrix = {}
        matrix["Y"] = []
        matrix["Z"] = []
        matrix["ABCD"] = []
        matrix["S"] = []
        
        cnt = 0
        static_input_nodes = self._input_nodes

        while self._frecuency <= self._upper_freq_limit:

            frequencies.append(self._frecuency)
            self.impedance_calculator()

            if self._s2p_device:
                self._input_nodes = static_input_nodes

                self.components_divider()

                self._components_values = self.components_values_malla_1
                self._components_nodes = self.components_nodes_malla_1
                self._input_nodes = self.in_nodes_1

                #print("components_nodes", self._components_nodes)
                #print("components_values", self._components_values)
                #print("input_nodes", self._input_nodes)

                
                if len(self._components_nodes) > 0:
                    self.serial_paralel_reduction()
                    self.components_to_node()
                    #print("nodes_matrix", self._nodes_matrix)
                    self.get_circuit_matrix()
                    #print("circuit_matrix", self._circuit_matrix)
                    self.get_y_matrix()
                    #print("no_in_nodes", self._no_in_nodes)
                    self.y2abcd()
                    ABCD_1 = self.abcd_matrix
                else:
                    ABCD_1 = np.eye(2, dtype=complex)
                

                self._components_values = self.components_values_malla_2
                self._components_nodes = self.components_nodes_malla_2
                self._input_nodes = self.in_nodes_2

                #print("components_nodes", self._components_nodes)
                #print("components_values", self._components_values)
                #print("input_nodes", self._input_nodes)

                if len(self._components_nodes) > 0:
                    self.serial_paralel_reduction()
                    self.components_to_node()
                    #print("nodes_matrix", self._nodes_matrix)
                    self.get_circuit_matrix()
                    #print("circuit_matrix", self._circuit_matrix)
                    self.get_y_matrix()
                    #print("no_in_nodes", self._no_in_nodes)
                    self.y2abcd()
                    ABCD_2 = self.abcd_matrix
                else:                
                    ABCD_2 = np.eye(2, dtype=complex)   
                    

                ABCD_S2P = self.__S2ABCD(self._s2p_device[cnt])
                cnt += 1


                ABCD_aux = np.dot(ABCD_1, ABCD_S2P)
                self.abcd_matrix = np.dot(ABCD_aux, ABCD_2)

                self.abcd2y()
                matrix["Y"].append(self.y_matrix)
                self.y2z()
                matrix["Z"].append(self.z_matrix)
                self.z2abcd()
                matrix["ABCD"].append(self.abcd_matrix)
                self.z2s()
                matrix["S"].append(self.s_matrix)



            else:
                self.serial_paralel_reduction()
                self.components_to_node()
                self.get_circuit_matrix()
                self.get_y_matrix()
                matrix["Y"].append(self.y_matrix)
                self.y2z()
                matrix["Z"].append(self.z_matrix)
                self.y2abcd()
                matrix["ABCD"].append(self.abcd_matrix)
                self.z2s()
                matrix["S"].append(self.s_matrix)

            self._frecuency += self._freq_step

        return matrix, frequencies, self._z_charac
    
if __name__ == "__main__":
    
    input_nodes = [0,7]

    components = [
        ["L", 0.00045, 0, 1],
        ["R", 10000, 1, 2],
        ["C", 0.01, 0],
        ["R", 1000, 0, 3],
        ["L", 0.001, 2, 3],
        ["C", 0.0001, 2, 4],
        ["R", 9800, 3, 4],
        ["C", 0.01, 3, 4],
        ["L", 0.01, 4],
        ["R", 1500, 4],
        ["R", 150, 4],
        ["L", 1, 4, 5],
        ["R", 1123, 5, 6],
        ["C", 0.00007, 6, 7],
        ["L", 0.1, 7, 4],
        ["R", 10000, 7]
    ]

    lower_freq_limit = 1e3 
    upper_freq_limit = 1e6  
    freq_step = 1e3  

    circuit = Circuit(components, input_nodes, lower_freq_limit, upper_freq_limit, freq_step, 50)
    
    circuit.run_simulation()