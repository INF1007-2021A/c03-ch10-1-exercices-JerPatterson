#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sympy



def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    list_coordinates = []

    for index in range(np.shape(cartesian_coordinates)[0]):
        radius = np.sqrt(np.sum(cartesian_coordinates[index]**2))
        angle = np.arctan2(cartesian_coordinates[index][1],cartesian_coordinates[index][0])
        list_coordinates.append([radius, angle])

    return np.asarray(list_coordinates)


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def graphics_default(title: str=" ", grid: bool=True, show_graph: bool=True) -> None:
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    if grid == True:
        plt.grid()


def graph_specific_intervall(intervall: list) -> None:
    x = np.linspace(intervall[0], intervall[1], 250)
    y = x**2 * np.sin(1 / x**2) + x

    plt.plot(x, y, "-b")
    graphics_default(f"Grapique de la fonction entre {intervall}")
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
        
    graphics_default("Calcul de π par la méthode de Monte-Carlo", False)

    return number_points_inside / iterations
    

def integrate_function() -> None:
    x = sympy.Symbol('x')
    f = sympy.exp(-x**2)
    f_integrated = sympy.integrate(f, x)
    sympy.plot(f_integrated)

    f = lambda x: np.exp(-x**2)

    return scipy.integrate.quad(f, -np.inf, np.inf)[0]
    


if __name__ == '__main__':
    print(linear_values())
    print(f"Conversion en coordonnées polaire {coordinate_conversion(np.array([[0, 0]]))}")
    print(f"L'indice de la valeur est {find_closest_index(np.array([1,3,5,7,-2, 20, 98]), -4)}")
    graph_specific_intervall([-1, 1])
    print(f" La valeur de pi/4 approximée est {montecarlo_approximation(1000)}")
    print(integrate_function())