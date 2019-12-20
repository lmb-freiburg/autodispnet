#!/usr/bin/env python3
from autodispnet.controller.base_controller import BaseController
import os

class Controller(BaseController):
    base_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
	controller = Controller()
	controller.run()

