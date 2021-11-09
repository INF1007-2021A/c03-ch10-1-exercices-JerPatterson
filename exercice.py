#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    radius = np.sqrt(np.sum(cartesian_coordinates**2))
    angle = np.arctan(cartesian_coordinates[1]/cartesian_coordinates[0])

    return np.array([radius, angle])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def graph_specific_intervall(intervall: tuple) -> None:
    x = np.linspace(intervall[0], intervall[1], 250)
    y = x**2 * np.sin(1 / x**2) + x

    plt.plot(x, y, "-b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Grapique de la fonction entre {intervall}")
    plt.grid()
    plt.show()


def montecarlo_approximation(iterations: int) -> float:
    number_points_inside = 0

    for _ in range(iterations):
        random_point = np.random.rand(2)
        if np.sqrt(np.sum(random_point**2)) <= 1:
            plt.scatter(random_point[0], random_point[1], c="m")
            number_points_inside +=1
        else:
            plt.scatter(random_point[0], random_point[1], c="b")
        
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Calcul de π par la méthode de Monte-Carlo")
    plt.show()

    return number_points_inside / iterations
    

def integrate_specific_interval(intervall):
    pass


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())
    print(f"Conversion en coordonnées polaire {coordinate_conversion(np.array([1, 1]))}")
    print(f"L'indice de la valeur est {find_closest_index(np.array([1,3,5,7,-2, 20, 98]), -4)}")
    graph_specific_intervall([-1, 1])
    print(f" La valeur de pi/4 approximée est {montecarlo_approximation(1000)}")