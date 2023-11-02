import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def H(y_i, target_name_0, target_name_1):
    return 1 - (len(y_i[np.where(y_i == target_name_0)]) / len(y_i))**2 - \
           (len(y_i[np.where(y_i == target_name_1)]) / len(y_i))**2

def Q(R, R_l, y_l, R_r, y_r, target_name_0, target_name_1):
    return -len(R_l) / len(R) * H(y_l, target_name_0, target_name_1) - \
           len(R_r) / len(R) * H(y_r, target_name_0, target_name_1)

def find_best_split(feature_vector, target_vector):
    # приведем значения таргета к 0 и 1
    unique_target = np.unique(target_vector)
    target_name_0 = unique_target[0]
    target_name_1 = unique_target[1]

    thresholds = [] # отсортированный по возрастанию вектор со всеми возможными порогами
    ginis = [] # вектор со значениями критерия Джини для каждого из порогов
    threshold_best = 0 # оптимальный порог (число)
    gini_best = None # оптимальное значение критерия Джини (число)

    # уникальные значения фич, среди которых будем искать порог
    unique_vals = np.unique(feature_vector)
    unique_vals = np.sort(unique_vals)

    for i in range(len(unique_vals) - 1):
        # пороговое значение
        threshold = (unique_vals[i+1] + unique_vals[i]) / 2

        # разобьем вектор фич и вектор таргетов по пороговому значению
        X_l = feature_vector[np.where(feature_vector <= threshold)[0]]
        y_l = target_vector[np.where(feature_vector <= threshold)[0]]
        X_r = feature_vector[np.where(feature_vector > threshold)[0]]
        y_r = target_vector[np.where(feature_vector > threshold)[0]]

        # посчитаем значение критерия гини для данного разбиения по формуле из ноутбука
        gini = Q(feature_vector, X_l, y_l, X_r, y_r, target_name_0, target_name_1)

        thresholds.append(threshold)
        ginis.append(gini)

        # обновим наилучшее значения для критерия Гини
        if gini_best is None or gini < gini_best:
            gini_best = gini
            threshold_best = threshold

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._root_node = {} #  переименовал _tree, потому что так логичнее, а то я не понимал вначале, что это за список
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):  # Ошибка
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if len(sub_y) < self._min_samples_split:  # Вместо того, чтобы учесть минимальное число элементов, необходимых для разбиения, они делают непойми что
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # В задании написано, что нужно вернуть число, без [0][0] возвращает массив
            return

        # if len(sub_y) < self._min_samples_leaf * 2:  # Минимальное число листьев они вообще не учитывают, но чтобы его учесть нормально, нужно слишком много думать
        #     node["type"] = "terminal"
        #     node["class"] = Counter(sub_y).most_common(1)[0][0]
        #     return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):  # Ошибка
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
                feature_vector = np.array(feature_vector, dtype=float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # ошибка
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[0])))  # ошибка
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))  # Офицеры... Погоны нацепили, **** жрут. **** **** ****
            else:
                raise ValueError
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is not None and (gini_best is None or gini > gini_best):  # ошибка связанная с моей реализацией поиска splita
                split = feature_vector < threshold
                # if len(sub_y[split]) == 0:
                #     continue

                feature_best = feature
                gini_best = gini

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": # ошибка
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature_split = node["feature_split"]
        split_type = self._feature_types[feature_split]

        if split_type == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif split_type == "real":
            if float(x[feature_split]) < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._root_node)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._root_node))
        return np.array(predicted)

# X = np.array([['b', 5.0], ['a', 2.0], ['b', 3.0], ['a', 4.0], ['c', 1.0]])  # очень люблю динамическую типизацию данных, самая удобная вещь на свете, так упрощает жизнь
# y = np.array([1, 1, 1, 0, 1])
#
# dt = DecisionTree(['categorical', 'real'])
# dt.fit(X, y)
#print(dt.predict(X))

# shrooms = pd.read_csv('agaricus-lepiota.data')
#
# le = LabelEncoder()
# for column in shrooms:
#     le.fit(shrooms[column])
#     shrooms[column] = le.transform(shrooms[column])
#
# X = shrooms.copy().drop(columns=['p']).to_numpy()
# y = shrooms['p'].copy().to_numpy()
#
# feature_types = np.array(['categorical' for _ in range(23)])
#
# dt = DecisionTree(feature_types)
#
# dt.fit(X, y)
