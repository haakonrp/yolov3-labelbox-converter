import os
import csv
import json
import logging
import requests
from PIL import Image
from sklearn.model_selection import train_test_split

# logger settings
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# create output dirs
logging.info('Create output dirs')
os.makedirs('output/data/img', exist_ok=True)

logging.info('Read CSV')
with open('labels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)

    # create train and test files
    train = open('output/data/train.txt', 'w+')
    test = open('output/data/test.txt', 'w+')

    # init csv array
    csvArray = []

    logging.info('Conversion START')
    for row in readCSV:
        # img names
        img_path = row[2]
        img_name = img_path.rpartition('/')[2]
        img_name_txt = img_name.split('.', 1)[0] + '.txt'

        # add name to csv array
        csvArray.append(img_name)

        # create label files
        labelFile = open('output/data/img/' + img_name_txt, 'w+')

        # fetch img to get width/height
        response = requests.get(img_path, stream=True, timeout=10.0)
        response.raw.decode_content = True
        img_width, img_height = Image.open(response.raw).size

        if row[3] != 'Skip':
            jsonObj = json.loads(row[3])
            for label in jsonObj['Sheep']:
                objID = -1
                # object class id
                # white = 0, grey = 1, black = 2
                if label['sheep_color'] == 'white':
                    objID = 0
                if label['sheep_color'] == 'grey':
                    objID = 1
                if label['sheep_color'] == 'black':
                    objID = 2

                # bounding box center x/y
                x = ((label['geometry'][0]['x'] + label['geometry'][2]['x']) / 2) / img_width
                y = ((label['geometry'][0]['y'] + label['geometry'][2]['y']) / 2) / img_height

                # bounding box width/height
                width = abs(label['geometry'][0]['x'] - label['geometry'][2]['x']) / img_width
                height = abs(label['geometry'][0]['y'] - label['geometry'][2]['y']) / img_height

                # write label files
                labelStr = str(objID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height)
                labelFile.write(labelStr + '\n')
        labelFile.close()
    logging.info('Conversion DONE')

    # split train and test data
    logging.info('Split train/test data START')
    trainArray, testArray = train_test_split(csvArray, test_size=0.2)

    # write train file
    for name in trainArray:
        train.write('data/img/' + name + '\n')

    # write test file
    for name in testArray:
        test.write('data/img/' + name + '\n')

    train.close()
    test.close()
    logging.info('Split train/test data DONE')
