{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Задача нелинейный метод найменьших квадратов"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Математический метод, применяемый для решения различных задач, основанный на минимизации суммы квадратов отклонений некоторых функций от искомых переменных. Он может использоваться для «решения» переопределенных систем уравнений (когда количество уравнений превышает количество неизвестных), для поиска решения в случае обычных (не переопределенных) нелинейных систем уравнений, для аппроксимации точечных значений некоторой функции. МНК является одним из базовых методов регрессионного анализа для оценки неизвестных параметров регрессионных моделей по выборочным данным"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Импортируем библиотеку:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Она отвечает за отрисовку графика, мы назовем ее plt, также импортируем:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib as mpl"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Нужно импортировать библиотеку numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Назовем ее np, с помощью неё будем аппроксимировать данные методом наименьших квадратов."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ниже представленны входные значения точек x и y исходящих из условия задачи для дальнейшего решение метода наимешьних квадратов."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=[2.5134, 2.0443, 1.6684, 1.3664, 1.1232, 0.9269, 0.7679, 0.6389, 0.5338, 0.4479, 0.3776, 0.3197, 0.2720, 0.2325, 0.1997, 0.1723, 0.1493, 0.1301, 0.1138, 0.1,0.0883,0.0783,0.0698,0.0624] \n",
    "y=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,\n",
    "   0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1,1.05,1.1,1.15]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Создадим функцию mnkGP:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mnkGP(x,y):"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "В функции создадим переменную d, она будет отвечать за степень полинома."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "d= 0.99"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Дальше с помощью np.polyfit(),аппроксимируем данные методом наименьших квадратов."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "beta = np.polyfit(x,y,d)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Выведем на экран коэффициент:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Коэффициент -- a %s  '%round(beta[0],4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "beta0 = beta"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Отрисуем исходный график:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(x, y, 'o', label='Original data', markersize=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Сгенерируем данные на основе значений из списка x."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xx = np.linspace(x[23], x[0], 24)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Отрисуем график:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(x, beta0*xx)\n",
    "plt.legend()\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Дальше мы вызовим нашу функцию и передадим ей значения x и y:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mnkGP(x,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Запустим наш скрипт на Python и посмотрим вывод на графике:"
   ]
  },
  {
   "attachments": {
    "Figure_1.png": {
     "image/png": "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzdeXhU5fnG8e9MNpJAFsIOMSSCIAIWiUXLJkIwBatSFJfWnyhasYsKxCpaW2grCK2xWqkL1ooLKIKoIFsUESOI4IIYVpMAYRMCgSQkJJPM/P44TkjMQkJmzmz357rmSjJzcubJm0jvvu8572NxOBwORERERCRgWD1dgIiIiIiYSwFQREREJMAoAIqIiIgEGAVAERERkQCjACgiIiISYBQARURERAKMAqCIiIhIgFEAFBEREQkwCoAiIiIiAUYBUERERCTAKACKiIiIBBgFQBEREZEAowAoIiIiEmAUAEVEREQCjAKgiIiISIBRABQREREJMAqAIiIiIgFGAVBEREQkwCgAioiIiAQYBUARERGRAKMAKCIiIhJgFABFREREAowCoIiIiEiAUQAUERERCTAKgCIiIiIBRgFQREREJMAoAIqIiIgEGAVAERERkQCjACgiIiISYBQARURERAKMAqCIiIhIgFEAFBEREQkwCoAiIiIiAUYBUERERCTAKACKiIiIBBgFQBEREZEAowAoIiIiEmAUAEVEREQCjAKgiIiISIBRABQREREJMAqAIiIiIgFGAVBEREQkwCgAioiIiAQYBUARERGRAKMAKCIiIhJgFABFREREAowCoIiIiEiAUQAUERERCTAKgCIiIiIBRgFQREREJMAoAIqIiIgEGAVAERERkQCjACgiIiISYBQARURERAKMAqCIiIhIgFEAFBEREQkwwZ4uwJfZ7XYOHjxIq1atsFgsni5HREREGsHhcFBUVESnTp2wWgNzLkwBsBkOHjxIfHy8p8sQERGRc5CXl0eXLl08XYZHKAA2Q6tWrQDjDygqKqrBY202G6tXr2bkyJGEhISYUV5A03ibS+NtLo23uTTe5jJjvAsLC4mPj6/63/FApADYDM5l36ioqEYFwIiICKKiovQPiAk03ubSeJtL420ujbe5zBzvQL58KzAXvkVEREQCmAKgiIiISIBRABQREREJMLoGUERE/EJl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3bqRkJDAhAkTmDBhAnv37uWdd96hZcuWOByOqn0E65uBu+uuu7j11lsJCgoiOTmZ1atXs379ehISEgAYPnw4f/vb37Db7ZSWlvLSSy+xe/ducnNzq45xOBw17gI+W01qBdg0fjkDOHfuXIYOHUp2dvZZ/yA+/vhj/v73v5Ofn9+kDSrdKjgUwlsbnxd/79laREQCwA3J8ayZcgW/GZLE6L6d+M2QJNZMuYIbkuPd9p4jR45k8+bN9O3bl3/+859MmDCB9957jz/96U+8//77NbZY6dKlS9UsXnUWi4WePXvSvn17AKZOncoNN9zAn//8ZxYsWMC//vUv4EwA7NatG126GDOa7dq1o2vXrjX+d/Kyyy4jLCyM4OBgFi9eTGVlJb///e95//33mT17NiEhIezbt482bdqQlJRUIzxWd8stt/DCCy/wwQcf8Lvf/Y59+/YxceLEquOffPJJBgwYwCOPPMLTTz/N+PHjGTlyZNWNLc76evbsWfX12WqSprE46pv39mEPP/wwcXFxpKWlsXbtWoYOHVrnceXl5Vx88cW0bt2a9evXs3379hp/bGdTWFhIdHQ0J0+ePOs0vc1mY/ny5YwaNapx+yb953I4sg1uXWLsByhN0uTxlmbReJtL413T6dOnyc3NJTExsd4lyeaw2+0UFhYSFRVV7x273mrDhg0MHjyYsrKys24o7S1cNd4N/V005X+//ZVv/SU30owZM7j55psB6v1/JwAzZ85kz549VbfHexXdCCIiIs20fPlyevXq5TPhT8zjt9cAfv+9sXTarl27Ol/ftWsXM2fOZNKkSVXXG3gV540gRdoKRkREGufzzz+vWg7dtm0bs2fP9s5JDvE4vw+AHTt2rPP1u+++m9jYWB5++OFGN7MuKyur0UDbeVGw8zb8hjhfP9txTtaINgQBlYWHsDfye+SMpo63NI/G21wa75psNhsOhwO73V61QbErOa+Ucr6HN3v55Zd5/fXXKSkpoUuXLjz44IPcd999Xl93da4ab7vdjsPhwGaz1ZoB1X87fhwACwoKiI6OJjQ0tNZrTz75JGvXruXVV1+lVatWjQ6AM2fOZPr06bWeX716dYNLzdU1tmVN0pHj9AEO7f6aL2zLG/U9UptaBJlL420ujbchODiYDh06UFxcXHWHqztU337EW82YMaPWjF9xcbGHqmme5o53eXk5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+fk9DmOttxDCcRnYrqamBqWlpfz8ngxufgjFa2RlZYnIyEixY8cOsWfPHjF37lwRExMjrly5Ipqbmx1PQV6+fFnExMSItLQ08ec//1m89NJLIjIyUmRnZ7v5CDzLWOt95MgRodfrxW9+8xuxf/9+8dxzz4nAwECRmZkp+vv73XwUniUjI0MAEACEVqsVH3zwgfj+++/F2bNnhRBCVFdXC4PBIJYsWSL27Nkjnn/+eREQECDWrVvn5pl7phvVe9euXSIoKEhs27ZNvPnmm+JXv/qV0Ov1YvXq1W6eueeZN2+e+P3vfy+6u7uFxWIRV69eFQMDA0IIIaxWqzh58qQQQoje3l6RmpoqkpKSxMsvvyx27dolZs6cKWbPni1sNps7D8GjjLXe33zzjdBoNGLdunVi//79YuvWrSIyMlIkJyeL9vZ2dx6CV2AAnCQWi0U89thjIjQ0VAQEBIisrCxRWVkphBBi586dIj093TH29OnTYunSpcJgMIjw8HDx5JNPCovF4q6pe6Tx1Pvll18WSUlJQq/XC5PJJDZs2CCsVqu7pu6xzp07J8rKykRZWZmoqKgQNptNrF+/XuTm5jrGlJSUiHvuuUf4+fmJ2NhYsXnzZtHd3e3GWXuuG9W7r69P/O53vxMmk0no9XqRmJgo8vPzRW9vr5tn7nlmz54tDAaD0Gg0jtAdGBgojh49Kg4ePChMJpPjfVxXVyceeughERQUJIxGo1i5cqW4cOGCm4/As4yn3m+99ZZITU0Ver1eREVFiSeeeEJcvHjRzUfgHTRCCOG+849ERETuVVtbi/Pnzzv1abVaLFiwAAaDwU2z8l6s962BAZCIiIhIZfgQCBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqQwDIBEREZHKMAASERERqcz/AyhDBkk6HQ0EAAAAAElFTkSuQmCC"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Figure_1.png](attachment:Figure_1.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Вывод"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Мы смогли решить систему уравнений с помощью метода найменьших квадратов и вывести график на экран. Результаты не отличаются высокой точностью, но при этом являются достаточно объективными"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
