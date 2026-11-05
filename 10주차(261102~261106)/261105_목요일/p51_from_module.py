from machine.car import drive
from machine.tv import watch

drive()
watch()

print("===================")

from machine import car, tv

car.drive()
tv.watch()

print("============= test 폴더 =============")

from machine.test import test_car, test_tv

test_car.drive()
test_tv.watch()
