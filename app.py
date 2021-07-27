from flask import Flask, render_template, url_for, request
import tensorflow as tf
import scipy
import scipy.misc
import imageio
# from scipy.misc.pilutil import imread
# from scipy.misc import imread, imresize
from PIL import Image
# for matrix math
import numpy as np
import pandas as pd
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os
from load import *
import base64
from tensorflow.keras.models import Sequential
from keras.models import load_model

app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model = load_model('model.h5')


# graph = tf.get_default_graph()

class Event:
    def __init__(self,
                 duration,
                 protocol_type,
                 service,
                 flag,
                 src_bytes,
                 dst_bytes,
                 land,
                 wrong_fragment,
                 urgent,
                 hot,
                 num_failed_logins,
                 logged_in,
                 num_compromised,
                 root_shell,
                 su_attempted,
                 num_root,
                 num_file_creations,
                 num_shells,
                 num_access_files,
                 is_host_login,
                 is_guest_login,
                 count,
                 srv_count,
                 serror_rate,
                 srv_serror_rate,
                 rerror_rate,
                 srv_rerror_rate,
                 same_srv_rate,
                 diff_srv_rate,
                 srv_diff_host_rate,
                 dst_host_count,
                 dst_host_srv_count,
                 dst_host_same_srv_rate,
                 dst_host_diff_srv_rate,
                 dst_host_same_src_port_rate,
                 dst_host_srv_diff_host_rate,
                 dst_host_serror_rate,
                 dst_host_srv_serror_rate,
                 dst_host_rerror_rate,
                 dst_host_srv_rerror_rate
                 ):
        self.duration = duration
        self.protocol_type = protocol_type
        self.service = service
        self.flag = flag
        self.src_bytes = src_bytes
        self.dst_bytes = dst_bytes
        self.land = land
        self.wrong_fragment = wrong_fragment
        self.urgent = urgent
        self.hot = hot
        self.num_failed_logins = num_failed_logins
        self.logged_in = logged_in
        self.num_compromised = num_compromised
        self.root_shell = root_shell
        self.su_attempted = su_attempted
        self.num_root = num_root
        self.num_file_creations = num_file_creations
        self.num_shells = num_shells
        self.num_access_files = num_access_files
        self.is_host_login = is_host_login
        self.is_guest_login = is_guest_login
        self.count = count
        self.srv_count = srv_count
        self.serror_rate = serror_rate
        self.srv_serror_rate = srv_serror_rate
        self.rerror_rate = rerror_rate
        self.srv_rerror_rate = srv_rerror_rate
        self.same_srv_rate = same_srv_rate
        self.diff_srv_rate = diff_srv_rate
        self.srv_diff_host_rate = srv_diff_host_rate
        self.dst_host_count = dst_host_count
        self.dst_host_srv_count = dst_host_srv_count
        self.dst_host_same_srv_rate = dst_host_same_srv_rate
        self.dst_host_diff_srv_rate = dst_host_diff_srv_rate
        self.dst_host_same_src_port_rate = dst_host_same_src_port_rate
        self.dst_host_srv_diff_host_rate = dst_host_srv_diff_host_rate
        self.dst_host_serror_rate = dst_host_serror_rate
        self.dst_host_srv_serror_rate = dst_host_srv_serror_rate
        self.dst_host_rerror_rate = dst_host_rerror_rate
        self.dst_host_srv_rerror_rate = dst_host_srv_rerror_rate


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['get', 'post'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    # encode it into a suitable format
    if request.method == 'POST':
        duration = request.form.get('duration')  # запрос к данным формы
        protocol_type = request.form.get('protocol_type')
        service = request.form.get('service')
        flag = request.form.get('flag')
        src_bytes = request.form.get('src_bytes')
        dst_bytes = request.form.get('dst_bytes')
        land = request.form.get('land')
        wrong_fragment = request.form.get('wrong_fragment')
        urgent = request.form.get('urgent')
        hot = request.form.get('hot')
        num_failed_logins = request.form.get('num_failed_logins')
        logged_in = request.form.get('logged_in')
        num_compromised = request.form.get('num_compromised')
        root_shell = request.form.get('root_shell')
        su_attempted = request.form.get('su_attempted')
        num_root = request.form.get('num_root')
        num_file_creations = request.form.get('num_file_creations')
        num_shells = request.form.get('num_shells')
        num_access_files = request.form.get('num_access_files')
        is_host_login = request.form.get('is_host_login')
        is_guest_login = request.form.get('is_guest_login')
        count = request.form.get('count')
        srv_count = request.form.get('srv_count')
        serror_rate = request.form.get('serror_rate')
        srv_serror_rate = request.form.get('srv_serror_rate')
        rerror_rate = request.form.get('rerror_rate')
        srv_rerror_rate = request.form.get('srv_rerror_rate')
        same_srv_rate = request.form.get('same_srv_rate')
        diff_srv_rate = request.form.get('diff_srv_rate')
        srv_diff_host_rate = request.form.get('srv_diff_host_rate')
        dst_host_count = request.form.get('dst_host_count')
        dst_host_srv_count = request.form.get('dst_host_srv_count')
        dst_host_same_srv_rate = request.form.get('dst_host_same_srv_rate')
        dst_host_diff_srv_rate = request.form.get('dst_host_diff_srv_rate')
        dst_host_same_src_port_rate = request.form.get('dst_host_same_src_port_rate')
        dst_host_srv_diff_host_rate = request.form.get('dst_host_srv_diff_host_rate')
        dst_host_serror_rate = request.form.get('dst_host_serror_rate')
        dst_host_srv_serror_rate = request.form.get('dst_host_srv_serror_rate')
        dst_host_rerror_rate = request.form.get('dst_host_rerror_rate')
        dst_host_srv_rerror_rate = request.form.get('dst_host_srv_rerror_rate')
        event = Event(duration,
                      protocol_type,
                      service,
                      flag,
                      src_bytes,
                      dst_bytes,
                      land,
                      wrong_fragment,
                      urgent,
                      hot,
                      num_failed_logins,
                      logged_in,
                      num_compromised,
                      root_shell,
                      su_attempted,
                      num_root,
                      num_file_creations,
                      num_shells,
                      num_access_files,
                      is_host_login,
                      is_guest_login,
                      count,
                      srv_count,
                      serror_rate,
                      srv_serror_rate,
                      rerror_rate,
                      srv_rerror_rate,
                      same_srv_rate,
                      diff_srv_rate,
                      srv_diff_host_rate,
                      dst_host_count,
                      dst_host_srv_count,
                      dst_host_same_srv_rate,
                      dst_host_diff_srv_rate,
                      dst_host_same_src_port_rate,
                      dst_host_srv_diff_host_rate,
                      dst_host_serror_rate,
                      dst_host_srv_serror_rate,
                      dst_host_rerror_rate,
                      dst_host_srv_rerror_rate
                      )
        x = [[duration,
              src_bytes,
              dst_bytes,
              land,
              wrong_fragment,
              urgent,
              hot,
              num_failed_logins,
              logged_in,
              num_compromised,
              root_shell,
              su_attempted,
              num_root,
              num_file_creations,
              num_shells,
              num_access_files,
              is_host_login,
              is_guest_login,
              count,
              srv_count,
              serror_rate,
              srv_serror_rate,
              rerror_rate,
              srv_rerror_rate,
              same_srv_rate,
              diff_srv_rate,
              srv_diff_host_rate,
              dst_host_count,
              dst_host_srv_count,
              dst_host_same_srv_rate,
              dst_host_diff_srv_rate,
              dst_host_same_src_port_rate,
              dst_host_srv_diff_host_rate,
              dst_host_serror_rate,
              dst_host_srv_serror_rate,
              dst_host_rerror_rate,
              dst_host_srv_rerror_rate]]
        x = np.asarray(x).astype('float32')


    out = model.predict(x)
    # print(out)
    # print(np.argmax(out, axis=1))
    response = np.argmax(out, axis=1)
    return render_template('index.html', pred=response[0])


if __name__ == "__main__":
    app.run(debug=True)
