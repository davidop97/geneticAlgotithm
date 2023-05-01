import numpy as np, matplotlib.pyplot as plt, random 
from sklearn.metrics import mean_squared_error

class AG:
    #Constructor
    def __init__(self, dimensions , mutation_rate, n_individuals, n_selection, n_generation, n_iterations, errot_tol, verbose = True):
        self.dimensions = dimensions 
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_selection = n_selection * n_individuals
        self.n_generation = n_generation 
        self.n_iterations = n_iterations
        self.error_tol = errot_tol
        self.verbose = verbose
    #Crea la posible solución aleatoria
    def create_individual(self):
        min_range = -512
        max_range = 512
        individual = [round(random.uniform(min_range, max_range), 2)for i in range(self.dimensions)]
        return individual
    #Crea la población de acuerdo al número indicado, en este caso 20
    def create_population(self):
        population = [self.create_individual() for i in range(self.n_individuals)]
        return population
    #Evalua la función objetivo por cada elemento de la población
    def create_fn_objective(self, population):
        fObj = []
        for i in range(len(population)):
            x = population[i][0]
            y = population[i][1]
            Z = round(-(y + 47) * np.sin(np.sqrt(np.abs(y + x/2 + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47)))), 2)
            fObj.append(Z)
        return fObj
    #Crea la función fitnees para la selección
    def create_fn_fitness(self, fObj): #UNA CURVA QUE DEPENDE DEL MÁXIMO DEL MÍNIMO)
        fitness = [round(1 / (1 + f), 5) for f in fObj]
        return fitness
    #Función para la selección
    def selection(self, fitness, population):
        total = sum(fitness)
        probability = [p / total for p in fitness]
        prob_accumulated = [sum(probability[:i+1]) for i in range(len(probability))]
        R = [random.uniform(0,1) for i in range(len(fitness))]
        indices = []
        i = 0 
        while len(indices) < self.n_selection:
            for j, prob in enumerate(prob_accumulated):
                if R[i] < prob:
                    if j not in indices: #Evita que se repitan padres
                        indices.append(j)
                        i += 1
                        break
        parents = [population[index] for index in indices]

        return parents
    #Funcion que se encarga de codificar los padres 
    def encode_parents(self, parents):
        encoded_parents = []
        for parent in parents:
            encoded_parent = []
            for value in parent:
                if value >= 0:
                    sign = 0
                else:
                    sign = 1
                    value = abs(value)

                integer_part = int(value)
                decimal_part = int(round((value - integer_part), 2) * 100)

                integer_part_binary = format(integer_part, '010b')
                decimal_part_binary = format(decimal_part, '06b')

                binary_value = integer_part_binary + decimal_part_binary + str(sign)
                encoded_parent.append(binary_value)

            encoded_parents.append(encoded_parent)

        return encoded_parents
    #Cruzamiento con aleatoriedad de padres, después de su selección en la ruleta
    def crossover_one_point(self, encoded_parents):
        population_size = self.n_generation
        children = []
        while len(children) < population_size:
            # seleccionar dos padres aleatoriamente
            p1, p2 = random.sample(encoded_parents, 2)
            
            # seleccionar un punto de corte aleatorio
            cut_point = random.randint(1, len(p1[1])-2)
            
            # realizar el cruce a un punto
            p1x = p1[0]
            p1y = p1[1]
            p2x = p2[0]
            p2y = p2[1]
            c1 = [p1x[:cut_point] + p2x[cut_point:], p1y[:cut_point] + p2y[cut_point:]]
            c2 = [p2x[:cut_point] + p1x[cut_point:],p2y[:cut_point] + p1y[cut_point:]]
            
            # añadir hijos a la lista de hijos
            children.append(c1)
            if len(children) < population_size:
                children.append(c2)
        
        return children
    #Mutacion a un bit
    def mutation(self, children):
        chromosomes = len(children)
        genes = len(children[0])
        total_genes = chromosomes * genes
        mutation_rate = self.mutation_rate
        mutations = round(total_genes * mutation_rate)
        mutation_indices = random.sample(range(total_genes), mutations)

        #Lista que contiene a todos los hijos
        genes_list = [gene for child in children for gene in child]

        for i in mutation_indices:
            gene = genes_list[i]
            random_index = random.randint(0, len(gene)-1)
            gene = gene[:random_index] + gene[random_index].replace(gene[random_index], str(int(not int(gene[random_index])))) + gene[random_index+1:]
            genes_list[i] = gene

            mutated_children = [genes_list[i:i+genes] for i in range(0, len(genes_list), genes)]
            
        return mutated_children
    
    #Función de decodificación
    def decode_children(self, mutated_children):
        decoded_children = []
        for child in mutated_children:
            decoded_child = []
            for gene in child:
                integer_part = int(gene[:9], 2)
                decimal_part = int(gene[9:16], 2) / 100
                sign = -1 if gene[16] == '1' else 1
                decoded_value = sign * (integer_part + decimal_part)
                decoded_child.append(decoded_value)
            decoded_children.append(decoded_child)
        
        return decoded_children
    
    
    def genetic_algorithm(self):
        population = self.create_population()
        fObj = self.create_fn_objective(population)
        fitness = self.create_fn_fitness(fObj)
        parents = self.selection(fitness, population)
        encoded_parents = self.encode_parents(parents)
        children = self.crossover_one_point(encoded_parents)
        mutated_children = self.mutation(children)
        decoded_children = self.decode_children(mutated_children)

        mse_values = []
        for i in range(self.n_iterations):
            fitness_old = fitness
            population = decoded_children
            fObj = self.create_fn_objective(population)
            fitness = self.create_fn_fitness(fObj)
            mse = mean_squared_error(fitness, fitness_old)
            mse_values.append(mse)

            if(mse_values[i] > self.error_tol):
                parents = self.selection(fitness, population)
                encoded_parents = self.encode_parents(parents)
                children = self.crossover_one_point(encoded_parents)
                mutated_children = self.mutation(children)
                decoded_children = self.decode_children(mutated_children)
            else: 
                break
        plt.plot(range(len(mse_values)), mse_values)
        plt.xlabel('Iteraciones')
        plt.ylabel('Error')
        plt.title('Evolución del error a través de las iteraciones')
        plt.show()

        return mse_values, parents


        
def main():
    model = AG(dimensions = 2, mutation_rate=0.2, n_individuals=20, n_selection=0.3, n_generation=20, n_iterations=50, errot_tol=1e-03)
    test = model.genetic_algorithm()
    print(test)

if __name__ == '__main__':
    main()
   