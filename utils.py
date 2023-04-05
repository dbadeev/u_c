import pandas as pd
import numpy as np

import sys
from nltk.stem import PorterStemmer


def stemline(line: str) -> str:
	'''
	:param line: Строка слов
	:return: Строка слов после стемминга
	'''

	ps = PorterStemmer()
	token_words = line.split()
	stem_sentence = []
	for word in token_words:
		stem_sentence.append(ps.stem(word))
		stem_sentence.append(" ")
	return "".join(stem_sentence)


def text_without_stopwords(text: str, stoplist: list) -> str:
	'''
	text приводится к нижнему регистру, удаляются стоп-слова из списка stoplist
	:param text: Исходный текст
	:param stoplist: Список стоп-слов
	:return: Текст без стоп-слов
	'''
	text = text.lower()
	text_split = text.split(' ')
	# print(text_split)
	res = [word for word in text_split if word not in stoplist]
	# print(res)
	return ' '.join(res)


def bin_arr(arr_: np.ndarray) -> np.ndarray:
	"""
	Преобразование двумерного массива с вероятностями в предсказания (выбирается индекс с максимальным значением для каждой строки)
	:arr_: Входной массив с вероятностями
	:return: Массив предсказаний
	"""
	res = np.array([])
	num_col = len(arr_[0])
	num_lines = 0
	for line in arr_:
		max_ = np.amax(line)
		line = np.where(line == max_, 1, 0)
		res = np.append(res, line)
		num_lines += 1
	res.shape = (num_lines, num_col)
	return res