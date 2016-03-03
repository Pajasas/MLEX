#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from math import sqrt

def max_(a,b):
	'''46 - 1'''
	if(a>b):
		return a
	return b

def max_of_three(a,b,c):
	'''46 - 2'''
	return max_(max_(a,b),c)

def sum(l):
	'''46 - 6a'''
	s = 0
	for x in l:
		s += x
	return s

def multiply(l):
	'''46 - 6b'''
	m = 1
	for x in l:
		m *= x
	return m

def reverse(s):
	'''46 - 7'''
	return s[::-1]

def is_palindrome(s):
	'''46 - 8'''
	return reverse(s) == s

def is_member(x, l):
	'''46 - 9'''
	for y in l:
		if x == y:
			return True
	return False

def generate_n_chars(n, c):
	'''46 - 11'''
	return "".join(c for x in xrange(n))

def histogram(l):
	'''46 - 12 - returns string instead of printing it (that would be harder to test).'''
	return "".join(generate_n_chars(x, '*') + "\n" for x in l)

def max_in_list(l):
	'''46 - 13'''
	m = l.pop()
	for x in l:
		m = max_(m,x)
	return m

def strings_to_len(l):
	'''46 - 14'''
	return [len(x) for x in l]


class Complex():
	'''Simple class for complex numbers'''
	def __init__(self, a, b):
		self.a = a
		self.b = b

	def absolute(self):
		return sqrt(self.a*self.a + self.b*self.b)

	def conjugate(self):
		return Complex(self.a, -self.b)

	def __eq__(self, other):
		return (isinstance(other, self.__class__)
			and self.__dict__ == other.__dict__)

class TestAll(unittest.TestCase):
	string1 = 'test'

	def test_max_(self):
		self.assertEqual(max_(5,6),6)
		self.assertEqual(max_(7,6),7)

	def test_max_of_three(self):
		self.assertEqual(max_of_three(1,2,3),3)
		self.assertEqual(max_of_three(1,4,3),4)
		self.assertEqual(max_of_three(1,5,3),5)

	def test_sum_and_multiply(self):
		self.assertEqual(sum([1,2,3,4]),10)
		self.assertEqual(multiply([1,2,3,4]),24)

	def test_reverse(self):
		self.assertEqual(reverse('abc'),'cba')
		self.assertEqual(reverse('123456'),'654321')

	def test_is_palindrome(self):
		self.assertTrue(is_palindrome('radar'))
		self.assertTrue(is_palindrome('abcba'))
		self.assertFalse(is_palindrome('abcbab'))

	def test_is_member(self):
		self.assertTrue(is_member('a', ['a', 'b', 'c', 'd']))
		self.assertTrue(is_member(7, [2, 3, 5, 7]))
		self.assertFalse(is_member('qwerty', ['555', 'asdfgh']))

	def test_generate_n_chars(self):
		self.assertEqual(generate_n_chars(5,'x'),'xxxxx')
		self.assertEqual(generate_n_chars(0,'!'),'')

	def test_histogram(self):
		self.assertEqual(histogram([]),'')
		self.assertEqual(histogram([1,0,5]),"*\n\n*****\n")

	def test_max_in_list(self):
		self.assertEqual(max_in_list([5,6]),6)
		self.assertEqual(max_in_list([7,6,5,4,3,9999]),9999)

	def test_strings_to_len(self):
		self.assertEqual(strings_to_len(['aa','','aaaaaaa']),[2,0,7])

	def test_simple_absolute(self):
		self.assertEqual(Complex(3,4).absolute(),5)

	def test_simple_conjugate(self):
		self.assertEqual(Complex(3,4).conjugate(),Complex(3,-4))


if __name__ == '__main__':
	unittest.main()
