import os
import json
import time
import subprocess
import datetime
import signal
import argparse
import logging
import pandas as pd

def process(file_path, start, end):
    '''
    frame = pd.read_json(file_path, lines=True)
    files = list(frame['file_name'])
    timeout = 5
    '''
    i = start
    timeout = 5
    files = os.listdir(file_path)
    print(len(files))
    if end > len(files):
        end = len(files)
    while i < end:
        slicer = "bash ./slicer.sh " + file_path + "  " + str(files[i]) + "  1 " + "parsed/" + str(files[i])
        start0 = datetime.datetime.now()
        process1 = subprocess.Popen(slicer, shell = True)
        while process1.poll() is None:
            time.sleep(0.2)
            end0 = datetime.datetime.now()
            if (end0-start0).seconds > timeout:
                os.kill(process1.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='funtions dic.', default='../devign_dataset')
    parser.add_argument('--start', help='start functions number to parsed', type=int, default=0)
    parser.add_argument('--end', help='end functions number to parsed', type=int, default=4500)
    args = parser.parse_args()
    file_path = args.file_path
    start = args.start
    end = args.end
    process(file_path, start, end)

