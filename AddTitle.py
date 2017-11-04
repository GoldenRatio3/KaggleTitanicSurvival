import csv
import sys

with open(sys.argv[1],'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	with open(sys.argv[2], 'w') as outfile:
		outfile.write(','.join(reader.next()) + ',Title\n')
		for row in reader:
			title = row[3].split(',')[1].split('.')[0].strip()
			row[3] = '"' + row[3] + '"'
			outfile.write(','.join(row) + ',' + title + '\n')
