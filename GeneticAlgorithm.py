import time
import matplotlib.pyplot as plt
from numpy.random import randint, rand
import pandas as pd
import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        file_name=None,
        maximize=True,
        population_size=100,
        tournament_size=4,
        crossover_prob=0.9,
        mutation_prob=0.02,
        num_mutations=1,
        elitism=False,
        max_time=60,
        max_gen=2000000000,
        january=None,
        initial_capital=10000,
        operational_cost=1,
    ):
        self.file_name = file_name
        self.maximize = maximize
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_mutations = num_mutations
        self.elitism = elitism
        self.max_time = max_time
        self.max_gen = max_gen
        self.january = (
            pd.read_csv(self.file_name) if self.file_name else self.generate_dataset()
        ).drop(columns=["Date"])
        self.initial_capital = initial_capital
        self.operational_cost = operational_cost
        self.population = list()

    @staticmethod
    def generate_dataset(
        num_stocks=500, start="2024-01-01", end="2024-01-31", freq="D"
    ):
        # Create a date range for all days in January
        date_range = pd.date_range(start=start, end=end, freq=freq)
        data = []
        for date in date_range:
            row = [date] + list(np.random.randint(100, 501, size=num_stocks))
            data.append(row)
        columns = ["Date"] + [str(i) for i in range(1, num_stocks + 1)]
        df = pd.DataFrame(data, columns=columns)
        return df

    def generate_chromosome_randomized_with_restrictions(self, n_assets=500, n_days=31):
        valid = False
        while not valid:
            chromosome = np.zeros(
                (n_assets, n_days), dtype=int
            )  # Initialize chromosome with 0
            total_capital = self.initial_capital  # Initial capital
            stocks_held = np.zeros(n_assets, dtype=int)  # Stocks held for each asset

            for day in range(n_days):
                # Generate a random order of assets to process
                random_asset_order = np.random.permutation(n_assets)

                for asset in random_asset_order:
                    # Current asset price on the given day
                    asset_price = self.january.iloc[day, asset]

                    # Probability of each action (hold, buy, sell)
                    action_prob = np.random.rand()

                    # Hold action (33% probability)
                    if action_prob < 1 / 3:
                        continue  # Do nothing
                    # Buy action (33% probability)
                    elif action_prob < 2 / 3:
                        if (
                            total_capital >= asset_price
                        ):  # Only buy if we have enough capital
                            max_buy = total_capital // asset_price  # Maximum we can buy
                            # max_buy = max(1, int(np.log(max_buy)))  # Ensure minimum of 1

                            quantity = np.random.randint(
                                1, max_buy + 1
                            )  # Random buy quantity
                            chromosome[asset, day] = quantity  # Record buy action
                            stocks_held[asset] += quantity  # Update held stocks
                            total_capital -= quantity * asset_price  # Deduct capital
                    # Sell action (33% probability)
                    else:
                        if stocks_held[asset] > 0:  # Only sell if we own stocks
                            max_sell = stocks_held[
                                asset
                            ]  # Cannot sell more than we own
                            quantity = np.random.randint(
                                1, max_sell + 1
                            )  # Random sell quantity
                            chromosome[asset, day] = -quantity  # Record sell action
                            stocks_held[asset] -= quantity  # Update held stocks
                            total_capital += (
                                quantity * asset_price
                            )  # Add capital from sale

            # Ensure all remaining stocks are sold on the last day
            for asset in range(n_assets):
                if stocks_held[asset] > 0:  # If there are remaining stocks
                    last_day_price = self.january.iloc[
                        n_days - 1, asset
                    ]  # Price on the last day

                    chromosome[asset, n_days - 1] -= stocks_held[
                        asset
                    ]  # Sell all remaining stocks
                    # print(f"chromosome[asset, n_days - 1] : {chromosome[asset, n_days - 1]} = -stocks_held[asset] : {-stocks_held[asset]} ")
                    total_capital += (
                        stocks_held[asset] * last_day_price
                    )  # Update total capital
                    stocks_held[asset] = 0  # Clear remaining stocks

            # Validate the chromosome
            valid = self.check_chromosome(chromosome)  # Validate chromosome

        return chromosome

    @staticmethod  # Not used yet...
    def check_all_stocks_sold(chromosome, end_date_index=30):
        """
        Checks if all stocks have been sold by the last day of the period.

        Parameters:
        - chromosome: numpy array with decisions to buy/sell/hold.
        - end_date_index: Column Index of the last day of the period (default is 30, cause there are 31 days in January).

        Returns:
        - bool: True if no stocks are left, False otherwise.
        """
        stocks_held = np.sum(chromosome[:, end_date_index], axis=0)
        all_sold = np.all(stocks_held == 0)

        return all_sold

    @staticmethod
    def aux_check_chromosome_has_previous_stocks(chromosome):
        for stock in range(chromosome.shape[0]):
            value = 0
            for day in range(
                chromosome.shape[1] - 1
            ):  # -1 because we do not want to check the last day!
                value += chromosome[stock][day]
                if value < 0:
                    # print(stock,day)
                    return False
        return True

    def check_chromosome(self, chromosome):
        total = 10000
        day_sold = 0
        if np.any(chromosome[:, 0] < 0):  # Can´t sell in the first day
            print("Can´t sell in the first day ")
            return False
        if not GeneticAlgorithm.aux_check_chromosome_has_previous_stocks(
            chromosome
        ):  # before selling stocks we must own them
            print("Before selling stocks we must own them")
            return False
        # if not check_all_stocks_sold(chromosome):
        #     print("All stocks must have been sold by the last day of the period.")
        #     return False

        for active_index in range(chromosome.shape[1]):  # Iterate through columns(days)
            day_sold, day_spent = 0, 0

            if np.any(chromosome[:, active_index] < 1):  # If it decides to sell
                negative_indexes = np.where(chromosome[:, active_index] < 0)[0]
                for index in negative_indexes:
                    day_sold += (
                        chromosome[index][active_index]
                        * self.january.iloc[active_index, index]
                    )  # Date column is taken into account so +1
            total_before = total
            total -= day_spent + day_sold

            if np.any(chromosome[:, active_index] > 1):  # If it decides to buy
                positive_indexes = np.where(chromosome[:, active_index] > 0)[0]
                for index in positive_indexes:
                    day_spent += (
                        chromosome[index][active_index]
                        * self.january.iloc[active_index, index]
                    )  # Date column is taken into account so +1

                if (
                    day_spent > total
                ):  # Cannot spend more than you have. We first buy, then we can sell.
                    print(f"Cannot spend more than you have.")
                    return False

            if total < 0:
                print(f"total<0 : {total} = {total_before} - {day_spent} + {day_sold}")
                return False
            # print(f"Day spent : {day_spent} > total : {total}")
            # print(f"TOTAL: {total},DAY_SOLD: {abs(day_sold)}, DAY_SPENT : {-day_spent}")
        return True

    def get_capital(self, chromosome, asset_index, day_index):
        """
        Calcula el capital restante y las acciones poseídas hasta un día y activo específicos.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - january: DataFrame, precios de los activos por día.
        - asset_index: int, índice del activo hasta el cual se evalúa.
        - day_index: int, índice del día hasta el cual se evalúa.
        - initial_capital: int, capital inicial disponible (por defecto, 10000).

        Returns:
        - capital: float, capital restante hasta el día y activo dados.
        - stocks_held: np.ndarray, acciones poseídas hasta el día y activo dados.
        """
        # Inicializar capital y stocks poseídos
        capital = self.initial_capital
        n_assets = chromosome.shape[0]
        stocks_held = np.zeros(n_assets, dtype=int)

        # Evaluar las decisiones hasta el día y activo seleccionados
        for d in range(day_index + 1):  # Recorre días hasta el día dado
            for a in range(asset_index + 1):  # Recorre activos hasta el índice dado
                if chromosome[a, d] > 0:  # Compra
                    cost = chromosome[a, d] * self.january.iloc[d, a]
                    capital -= cost
                    stocks_held[a] += chromosome[a, d]
                elif chromosome[a, d] < 0:  # Venta
                    revenue = -chromosome[a, d] * self.january.iloc[d, a]
                    capital += revenue
                    stocks_held[a] += chromosome[a, d]  # Resta la venta

        return capital, stocks_held

    def filter_and_choose_random(self, last_day, asset, max_quantity):
        """
        Filtra el DataFrame `january` para seleccionar valores mayores a un día y activo dados,
        y con precios menores que una cantidad específica. Finalmente, escoge uno al azar.

        Parameters:
        - january: pd.DataFrame, precios de los activos (filas: días, columnas: activos).
        - last_day: int, índice del último día considerado.
        - asset: int, índice del activo.
        - max_quantity: float, cantidad máxima para filtrar.

        Returns:
        - (selected_day, selected_asset, price): Tupla con el día, activo y precio seleccionado.
        Devuelve (None, None, None) si no hay resultados que cumplan las condiciones.
        """
        # Filtrar el DataFrame para días y activos posteriores al dado
        filtered_df = self.january.iloc[last_day:-1, asset]

        # Filtrar valores por la cantidad máxima
        filtered_df = filtered_df[filtered_df <= max_quantity]

        # Verificar si hay elementos disponibles después del filtrado
        if filtered_df.empty:
            return None, None, None

        # Escoger un índice aleatorio entre los valores que cumplen las condiciones
        selected_index = np.random.choice(filtered_df.index)

        # Devolver el día, el activo y el precio
        return selected_index, asset, self.january.iloc[selected_index, asset]

    @staticmethod
    def find_lasts_movements(chromosome, day_index, asset_index):
        """
        Encuentra la última y próxima compra y venta realizada en cualquier activo antes de un día específico.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - day_index: int, índice del día.
        - asset_index int, índice del stock.

        Returns:
        -
        """
        results = list()
        asset_found = False
        day_found = False
        # Recorrer días y activos en orden inverso
        for asset in range(asset_index - 1, -1, -1):
            for day in range(day_index - 1, -1, -1):
                if chromosome[asset, day] > 0:  # Si se encuentra una compra
                    last_buy = [asset, day, chromosome[asset, day]]
                    results.append(last_buy)
                    asset_found = True
                    break
            if not asset_found:
                return []
            else:
                break

        # Recorrer días y activos en orden lógico
        for day_ind in range(day + 1, chromosome.shape[1], 1):
            if chromosome[asset, day_ind] < 0:  # Si se encuentra una venta
                results.append([asset, day_ind, chromosome[asset, day_ind]])
                day_found = True
                break

            if not day_found:
                return []

        return results

    @staticmethod
    def generate_mutation_indices(chromosome, num_mutations):
        """
        Genera índices de mutación asegurando que siempre cumplan con los criterios
        de encontrar una última compra válida y una venta válida.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - num_mutations: int, número de mutaciones a generar.

        Returns:
        - list de tuplas (asset_index, day_index) con índices válidos.
        """
        n_assets, n_days = chromosome.shape
        selected_indices = []

        while len(selected_indices) < num_mutations:
            # Generar índices aleatorios
            asset_index = np.random.randint(0, n_assets)
            day_index = np.random.randint(0, n_days)

            try:
                # Comprobar si el índice cumple con los criterios
                movements = GeneticAlgorithm.find_lasts_movements(
                    chromosome, day_index, asset_index
                )
                # Validar si encontró tanto una compra como una venta
                last_buy = any(m[2] > 0 for m in movements)
                next_sale = any(m[2] < 0 for m in movements)

                if last_buy and next_sale:
                    selected_indices.append((asset_index, day_index))
            except:
                continue

        return selected_indices

    # TODO : Implement the probability condition
    def mutate_selected_points(self, chromosome):
        """
        Realiza mutaciones en puntos seleccionados de un cromosoma respetando las restricciones.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - january: DataFrame, precios de los activos por día.
        - initial_capital: int, capital inicial disponible (por defecto, 10000).

        Returns:
        - mutated_chromosome: np.ndarray, cromosoma mutado.
        """
        mutated_chromosome = chromosome.copy()
        n_assets, n_days = chromosome.shape
        selected_indices = GeneticAlgorithm.generate_mutation_indices(
            chromosome, num_mutations=self.num_mutations
        )  # [(np.random.randint(0, n_assets), np.random.randint(0, n_days)) for _ in range(num_mutations)]
        print(selected_indices)

        for asset_index, day_index in selected_indices:
            # Obtener el capital y las acciones disponibles hasta este momento
            capital, stocks_held = self.get_capital(
                mutated_chromosome, asset_index, day_index
            )

            # Precio del activo en el día actual
            asset_price = self.january.iloc[day_index, asset_index]

            movements = GeneticAlgorithm.find_lasts_movements(
                chromosome, day_index, asset_index
            )
            last_asset, last_day, quantity = movements[0]  # Buys
            print(f"Buys : {movements[0]}")
            print(f"Sells : {movements[1]}")
            substract = np.random.randint(1, quantity + 1)
            mutated_chromosome[last_asset][last_day] = (
                quantity - substract
            )  # Now we must make sure we sell "substract amount" fewer participations of that stock!

            last_sell, last_sell_day, sell_quantity = movements[1]  # Sells
            mutated_chromosome[last_sell][last_sell_day] = (
                sell_quantity + substract
            )  # now we should have substract*january.iloc[last_day, last_asset] USD more

            extra_usd_earned = substract * self.january.iloc[last_day, last_asset]

            selected_day, selected_asset, price = self.filter_and_choose_random(
                last_day, last_asset, extra_usd_earned
            )
            max_buy = extra_usd_earned // price
            actives_bought = np.random.randint(
                1, max_buy + 1
            )  # we should have some USD margin. We must now make sure to sell them before january ends.
            mutated_chromosome[selected_asset][selected_day] = actives_bought

            day_to_sell = np.random.choice(
                [day for day in range(selected_day, 30, 1)]
            )  # We select a random day to sell, but not the last one.
            mutated_chromosome[selected_asset][day_to_sell] -= actives_bought

        return mutated_chromosome

    def calculate_weights_from_chromosome(self, cromosoma):
        """
        Calculate portfolio weights based on the chromosome (buy/sell quantities).

        Parameters:
        - cromosoma: np.array, a matrix of buy/sell decisions (positive for buy, negative for sell).

        Returns:
        - weights: np.array, portfolio weights (proportions of capital allocated to each asset).
        """
        # Initialize the net flows (purchases/sales) for each asset
        net_flows = np.zeros(cromosoma.shape[0])  # One entry for each asset

        # Calculate net flows for each asset based on the chromosome
        for asset in range(cromosoma.shape[0]):
            for day in range(cromosoma.shape[1]):
                if cromosoma[asset, day] > 0:  # Buy action
                    net_flows[asset] -= (
                        cromosoma[asset, day] * self.january.iloc[day, asset]
                    )  # Outflow (buy)
                elif cromosoma[asset, day] < 0:  # Sell action
                    net_flows[asset] += (
                        abs(cromosoma[asset, day]) * self.january.iloc[day, asset]
                    )  # Inflow (sell)

        # Normalize to get portfolio weights (capital allocation for each asset)
        total_flows = np.sum(net_flows)
        weights = net_flows / total_flows

        return weights

    @staticmethod
    def calculate_portfolio_return(weights, expected_returns):
        """
        Calculate the expected return of the portfolio.

        Parameters:
        - weights: np.array, portfolio weights (proportions of capital allocated to each asset).
        - expected_returns: np.array, expected returns for each asset. expected_returns = january.mean(axis=0).to_numpy()

        Returns:
        - portfolio_return: float, expected return of the portfolio.
        """
        return np.dot(weights, expected_returns)

    @staticmethod
    def calculate_portfolio_risk(weights, returns_df):
        """
        Calculate the portfolio risk (variance) using the covariance matrix of returns.

        Parameters:
        - weights: np.array, portfolio weights (proportions of capital allocated to each asset).
        - returns_df: pd.DataFrame, DataFrame of historical asset returns over time.

        Returns:
        - risk: float, portfolio variance.
        """
        covariance_matrix = returns_df.cov()  # Covariance matrix of asset returns
        risk = np.dot(
            weights.T, np.dot(covariance_matrix, weights)
        )  # Portfolio variance
        return risk

    def CARA(chromosome, expected_returns, returns_df, gamma):
        """
        Calculate the Certainty-Equivalent Risk Aversion (CARA) objective function for a portfolio.

        Parameters:
        - chromosome: np.array, a matrix representing the buy/sell decisions for each asset and day.
        - expected_returns: np.array, expected returns for each asset.
        - returns_df: pd.DataFrame, DataFrame of asset returns over time.
        - gamma: float, risk aversion coefficient.

        Returns:
        - cara_value: float, the CARA objective value for the portfolio.
        """
        # 1. Calculate portfolio weights from chromosome (buy/sell quantities)
        weights = self.calculate_weights_from_chromosome(chromosome)

        # 2. Calculate portfolio return (E[R_p])
        portfolio_return = GeneticAlgorithm.calculate_portfolio_return(
            weights, expected_returns
        )  # expected_returns(500,), average daily predictions. expected_returns = january.mean(axis=0).to_numpy()  # Vector de tamaño (500,)

        # 3. Calculate portfolio risk (variance)
        portfolio_risk = GeneticAlgorithm.calculate_portfolio_risk(weights, returns_df)

        # 4. Calculate CARA objective function
        cara_value = portfolio_return - (gamma / 2) * portfolio_risk

        return cara_value

    def is_better_than(self, a, b):
        return a > b if self.maximize else a < b

    def selection(self, population, fitness):
        chosen = randint(len(population))
        for i in randint(0, len(population), self.tournament_size - 1):
            if self.is_better_than(fitness[i], fitness[chosen]):
                chosen = i
        return population[chosen]

    # def crossover(self, p1, p2):
    #     c1, c2 = p1.copy(), p2.copy()
    #     if rand() < self.crossover_prob:
    #         point = randint(1, len(p1) - 2)
    #         c1 = p1[:point] + p2[point:]
    #         c2 = p2[:point] + p1[point:]
    #     return [c1, c2]

    def crossover(self, p1, p2):
        pass

    def best_average_worst(self, fitness):
        best_chromosome = 0
        best_fitness = fitness[0]
        worst_fitness = fitness[0]
        average_fitness = sum(fitness) / self.population_size
        for i in range(len(fitness)):
            if self.is_better_than(fitness[i], best_fitness):
                best_chromosome = i
                best_fitness = fitness[i]
            if self.is_better_than(worst_fitness, fitness[i]):
                worst_fitness = fitness[i]
        return best_chromosome, best_fitness, average_fitness, worst_fitness

    def genetic_algorithm(self, expected_returns, returns_df, gamma):
        generation = 0
        evolution = []
        population = [
            self.generate_chromosome_randomized_with_restrictions()
            for _ in range(self.population_size)  # Ensured valid chromosome.
        ]
        fitness = [
            self.CARA(c, expected_returns, returns_df, gamma) for c in population
        ]
        best_chromosome, best_fitness, average_fitness, worst_fitness = (
            self.best_average_worst(fitness)
        )
        print(
            f">Generation {generation}: Worst: {worst_fitness:.3f} Average: {average_fitness:.3f} Best: {best_fitness:.3f}"
        )
        evolution.append([best_fitness, average_fitness, worst_fitness])
        start_time = time.time()

        while generation < self.max_gen and time.time() - start_time < self.max_time:
            generation += 1
            selected = [
                self.selection(population, fitness) for _ in range(self.population_size)
            ]
            offspring = []
            if self.elitism:
                offspring.append(population[best_chromosome])
            for i in range(0, self.population_size, 2):
                p1, p2 = selected[i], selected[(i + 1) % len(selected)]
                for c in self.crossover(p1, p2):
                    self.mutate_selected_points(c)
                    if len(offspring) < self.population_size:
                        offspring.append(c)
            population = offspring
            fitness = [
                self.CARA(c, expected_returns, returns_df, gamma) for c in population
            ]
            best_chromosome, best_fitness, average_fitness, worst_fitness = (
                self.best_average_worst(fitness)
            )
            print(
                f">Generation {generation}: Worst: {worst_fitness:.3f} Average: {average_fitness:.3f} Best: {best_fitness:.3f}"
            )
            evolution.append([best_fitness, average_fitness, worst_fitness])

        return population[best_chromosome], fitness[best_chromosome], evolution

    def execute(self):
        best_chromosome, best_fitness, evolution = self.genetic_algorithm(
            expected_returns, returns_df, gamma
        )
        print(
            f"Execution completed! Best solution: f({best_chromosome}) = {best_fitness:.3f}"
        )

        # TODO
        # resulting_usd = calculate_result_usd_function(best_chromosome)
        # print(
        #     f"Resulting USD: {max_capacity:.2f}, Stocks bought: {stocks_bought:.2f}, Stocks sold: {stocks_sold:.2f}"
        # )

        plt.plot(range(len(evolution)), [x[0] for x in evolution], label="Best")
        plt.plot(range(len(evolution)), [x[1] for x in evolution], label="Average")
        plt.plot(range(len(evolution)), [x[2] for x in evolution], label="Worst")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Evolution")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    GA = GeneticAlgorithm()
    df = GA.generate_dataset()
    df.to_csv("data/data.csv", index=None)
    c = GA.generate_chromosome_randomized_with_restrictions()
    print(f"Normal chromosome : {c}")
    c_muted = GA.mutate_selected_points(c)
    print(f"Muted chromosome : {c_muted}")
