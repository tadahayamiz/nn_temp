# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

main file

@author: tadahaya
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lower", default=False)
args = parser.parse_args()

def main():
    hello = "Hello World!"
    if args.lower:
        hello = hello.lower()
    print(hello)

def main2():
    goodbye = "Goodbye World!"
    if args.lower:
        goodbye = goodbye.lower()
    print(goodbye)

# not necessary when entry_points are used
if __name__ == "__main__":
    main()