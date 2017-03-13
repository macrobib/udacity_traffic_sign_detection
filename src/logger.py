"""Module to log to a csv file."""
import csv
import time
from datetime import datetime

class logger:
    """Logger class"""

    def __init__(self, filename=None, dirname=None):
        if dirname == None:
            now = datetime.now()
            # Directory for tensorboard logs.
            self.dirname = "../log" + now.strftime("%Y%m%d-%H%M%S") + "/"
        else:
            self.dirname = dirname
        self.filename = filename
        self.fileid = None

    def __enter__(self):
        self.fileid = open(self.filename, 'rb')
        self.writer = csv.writer(self.fileid, delimiter=';')

    def __exit__(self):
        self.fileid.close() 
    
    def log(self, data):
        """Log to csv"""
        now = time.strftime('%H:%M:%S')
        self.writer.writerow([data, now])

    def variable_summaries(self, var, name):
        """Save the variable scope"""
        #https://gist.github.com/Wei1234c/141b4316389826dd0b741e8f293c8e49
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max' + name, tf.reduce_max(var))
            tf.scalar_summary('min' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def get_file_name(self):
        """Return csv file name."""
        return self.filename

    def get_column_data(self, col):
        """Get data corresponding to given column"""
        col_read = []
        with open(self.filename, 'rb') as file:
            reader = csv.reader(file, delimiter=';')
            col_read = [row[col] for row in reader]
        return col_read
    

        
