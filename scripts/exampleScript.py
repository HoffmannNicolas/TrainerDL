
import os
import sys
# Adds higher directory to python modules path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from examplePackage.exampleClass import ExampleClass

exampleObject = ExampleClass()
print(exampleObject)
