import urllib.request
import urllib.parse
from os import path, sep, chdir, getcwd, environ, remove
from time import sleep
from sys import exc_info
import tempfile
import shutil
import contextlib
import flask
from lxml.html import fromstring, tostring
import cv2 as cv
from re import findall
from urllib.parse import urlparse
from hashlib import sha1
from collections import OrderedDict
import requests
import numpy as np

app = flask.Flask(__name__)
BASE_DIRECTORY = path.dirname(path.abspath(__file__))


def make_filename(url, extension="jpg"):
    return sha1(url.encode()).hexdigest() + "." + extension


def add_glasses(filename, face_info, glass_color):
    try:
        eye_cascade = cv.CascadeClassifier(BASE_DIRECTORY + sep + 'XML/haarcascade_eye.xml')
        # Credit: Adapted from example in OpenCV Documentation, Face Detection using Haar Cascades
        #         License: https://opencv.org/license.html
        #                  Copyright (C) 2000-2018, Intel Corporation, all rights reserved.
        #                  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
        #                  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
        #                  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
        file_name_full = BASE_DIRECTORY + sep + 'static' + sep + filename
        img = cv.imread(file_name_full)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for face in face_info:
            w, h, x, y = face['w'], face['h'], face['x'], face['y']
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) is 0: #no valid eyes or glassed eyes found
                eyes = np.array([[x - w * 3 // 5, y - h * 1 // 7, w * 7 // 8, h * 1 // 5]])

            for (ex, ey, ew, eh) in eyes:
                #Left Arm
                cv.line(roi_color, (1 * ex // 5, ey - eh // 3), (ex, ey + eh // 2), glass_color, 2)

                #Right Arm
                cv.line(roi_color, (9 * ex // 5 + ew, ey - eh // 3), (ex + ew, ey + eh // 2), glass_color, 2)

                #Left
                cv.rectangle(roi_color, (ex, ey), (ex + 2 * ew // 5, ey + eh), glass_color, 2)

                #Connector
                cv.rectangle(roi_color, (ex + 2 * ew // 5, ey + eh // 3), (ex + 3 * ew // 5,ey + eh // 3), glass_color, 2)

                #Right
                cv.rectangle(roi_color, (ex + 3 * ew // 5, ey), (ex + ew, ey + eh), glass_color, 2)

        cv.imwrite(file_name_full, img)

    except (Exception, Warning) as e:
        exception_type, exception_obj, exception_tb = exc_info()
        exception_filename = path.split(exception_tb.tb_frame.f_code.co_filename)[1]
        print(type(e), e, exception_type, exception_filename, exception_tb.tb_lineno)


def add_mustache(filename, face_info):
    try:
        nose_cascade = cv.CascadeClassifier(BASE_DIRECTORY + sep + 'XML/haarcascade_mcs_nose.xml')
        # Credit: Adapted from example in OpenCV Documentation, Face Detection using Haar Cascades
        #         License: https://opencv.org/license.html
        #                  Copyright (C) 2000-2018, Intel Corporation, all rights reserved.
        #                  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
        #                  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
        #                  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
        file_name_full = BASE_DIRECTORY + sep + 'static' + sep + filename
        img = cv.imread(file_name_full)
        mustache = cv.imread(BASE_DIRECTORY + sep + 'adhoc/mustache.png')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for face in face_info:
            w, h, x, y = face['w'], face['h'], face['x'], face['y']
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            nose = nose_cascade.detectMultiScale(roi_gray)
            # Credit: Kunal Gupta, example on Github
            #         License: MIT License Copyright (c) 2017 Kunal Gupta
            #                  https://raw.githubusercontent.com/kunalgupta777/Moustache-Adder-/master/LICENSE
            mst_width = int(w*0.4166666) + 1
            mst_height = int(h*0.142857) + 1
            mustache = cv.resize(mustache,(mst_width, mst_height))
            for i in range(int(0.62857142857*h), int(0.62857142857*h) + mst_height):
                for j in range(int(0.29166666666*w), int(0.29166666666*w) + mst_width):
                    for k in range(3):
                        if mustache[i-int(0.62857142857*h)][j-int(0.29166666666*w)][k] < 235:
                            img[y+i][x+j][k] = mustache[i - int(0.62857142857*h)][j-int(0.29166666666 * w)][k]

        cv.imwrite(file_name_full, img)

    except (Exception, Warning) as e:
        exception_type, exception_obj, exception_tb = exc_info()
        exception_filename = path.split(exception_tb.tb_frame.f_code.co_filename)[1]
        print(type(e), e, exception_type, exception_filename, exception_tb.tb_lineno)


def add_cartoon(filename):
    try:
        # Credit: Adapted from example in OpenCV Documentation, Face Detection using Haar Cascades
        #         License: https://opencv.org/license.html
        #                  Copyright (C) 2000-2018, Intel Corporation, all rights reserved.
        #                  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
        #                  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
        #                  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
        file_name_full = BASE_DIRECTORY + sep + 'static' + sep + filename
        img = cv.imread(file_name_full)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
        color = cv.bilateralFilter(img, 9, 9, 7)
        cartoon = cv.bitwise_and(color, color, mask=edges)
        cv.imwrite(file_name_full, cartoon)

    except (Exception, Warning) as e:
        exception_type, exception_obj, exception_tb = exc_info()
        exception_filename = path.split(exception_tb.tb_frame.f_code.co_filename)[1]
        print(type(e), e, exception_type, exception_filename, exception_tb.tb_lineno)


def get_image_info(filename):
    # Credit: Numpy Documentation, adapted from example in numpy.ndarray.shape
    #       License: Copyright (c) 2005, NumPy Developers
    #                https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html
    #                https://docs.scipy.org/doc/numpy-1.10.0/license.html
    img = cv.imread(filename)
    file_dimensions = {}
    if img is not None:
        file_dimensions['h'], file_dimensions['w'] = img.shape[:2]
        face_cascade = cv.CascadeClassifier(BASE_DIRECTORY + sep + 'XML/haarcascade_frontalface_default.xml')
        # Credit: Adapted from example in OpenCV Documentation, Face Detection using Haar Cascades
        #         License: https://opencv.org/license.html
        #                  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        file_dimensions["faces"] = list()
        list_of_faces = []
        for (x,y,w,h) in faces:
            list_of_faces.append({'w': w, 'h': h, 'x': x, 'y': y})
  
        file_dimensions["faces"] = sorted(list_of_faces)
        return file_dimensions
    else:
        file_dimensions["faces"] = list()
        return file_dimensions


@contextlib.contextmanager
def cd(new_directory, cleanup=lambda: True):
    # Credit: Christopher Dunn, adapted from example of Temp Directory
    #           License: StackOverflow Creative Commons
    #           https://stackoverflow.com/questions/3223604, very loosely based off of example
    previous_directory = getcwd()
    chdir(path.expanduser(new_directory))
    try:
        yield

    finally:
        chdir(previous_directory)
        cleanup()


@contextlib.contextmanager
def pushd_temp_dir(base_dir=None, prefix="tmp.hpo."):
    # Credit: Alexander J. Quinn, Adapted from example in 364 Project Document
    #         License: Purdue University (C)
    #                  https://goo.gl/dk8u5S, Used with permission
    directory_path = tempfile.mkdtemp(prefix=prefix, dir=base_dir)

    def cleanup():
        shutil.rmtree(directory_path)
        
    with cd(directory_path, cleanup):
        yield directory_path


def find_profile_photo_filename(filename_to_etree):
    for image_name, node in filename_to_etree.items():
        size = get_image_info(getcwd() + sep + image_name)
        if not size['faces']:
            pass
        else:
            return image_name


def copy_profile_photo_to_static(etree):
    with fetch_images(etree) as filename_to_node:
        try:
            filename = find_profile_photo_filename(filename_to_node)
            if filename is not None:
                shutil.move(getcwd() + sep + filename, BASE_DIRECTORY + sep + 'static' + sep + filename)
                face_info = get_image_info(BASE_DIRECTORY + sep + 'static' + sep + filename)
                if flask.request.args.get(key='glass_color') is not None:
                    glass_color = tuple(
                            int(flask.request.args.get(key='glass_color').lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]
                else:
                    glass_color = (0, 0, 0)
                add_glasses(filename, face_info['faces'], glass_color)
                if flask.request.args.get(key='mustache') is not None:
                    add_mustache(filename, face_info['faces'])
                if flask.request.args.get(key='cartoon') is not None:
                    add_cartoon(filename)
                for element, attribute, link, pos in etree.iterlinks():
                    if link == filename_to_node[filename].attrib['src']:
                        new_src = link.replace(filename_to_node[filename].attrib['src'],
                                               flask.url_for('static', filename=filename))
                        element.set('src', new_src)
                return (BASE_DIRECTORY + sep + 'static' + sep + filename)

        except (Exception, Warning) as error:
            exception_type, exception_obj, exception_tb = exc_info()
            exception_filename = path.split(exception_tb.tb_frame.f_code.co_filename)[1]
            print(type(error), error, exception_type, exception_filename, exception_tb.tb_lineno)


@contextlib.contextmanager
def fetch_images(html_root):
    with pushd_temp_dir("data"):
        try:
            dir_path = getcwd()
            total_filename_to_node = OrderedDict()
            for node in html_root.iter():
                if node.tag == "img":
                    try:
                        image_name = make_filename(node.attrib['src'])
                        urllib.request.urlretrieve(node.attrib['src'], image_name)
                        total_filename_to_node[image_name] = node

                    except (Exception, Warning) as e:
                        print(total_filename_to_node)
                        print("Could not get source of image:" + node.attrib['src'] + " named as " + image_name)
                        exception_type, exception_obj, exception_tb = exc_info()
                        exception_filename = path.split(exception_tb.tb_frame.f_code.co_filename)[1]
                        print(type(e), e, exception_filename, exception_tb.tb_lineno)

            yield total_filename_to_node

        except FileNotFoundError as error:
            exception_type, exception_obj, exception_tb = exc_info()
            exception_filename = path.split(exception_tb.tb_frame.f_code.co_filename)[1]
            print(getcwd(), type(error), error, exception_type, exception_filename, exception_tb.tb_lineno)


@app.route("/view")
def view_page(url = "", retries = 0):
    try:
        url = flask.request.args.get(key='url')

        if url.strip() is "":
            flask.flash("URL is empty, please fill in a value before optometrizing")
            return flask.redirect(flask.url_for('root_page'))

        request_check = urlparse(url)
        req = requests.get(url)

        if retries > 20:
            raise requests.exceptions.ConnectionError

        if request_check.netloc == '' or req.status_code != 200:
            flask.flash("URL is broken or invalid")
            return flask.redirect(flask.url_for('root_page'))

        if len(findall(r'(facebook|twitter|instagram|myspace|reddit|vine|tinder|linkedin|tumblr|google)', url)) is not 0:
            flask.flash("Cannot Navigate to social media sites like " + '{uri.netloc}/'.format(uri=request_check).replace('www.', '').replace('.com', '').replace('/', '').capitalize())
            return flask.redirect(flask.url_for('root_page'))

        else:
            # Credit: Adapted from example in Python 3.4 Documentation, urllib.request
            #         License: PSFL https://www.python.org/download/releases/3.4.1/license/
            #                  https://docs.python.org/3.4/library/urllib.request.html
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'PurdueUniversityClassProject/1.0 (bharadwr@purdue.edu https://goo.gl/dk8u5S)')
            r = urllib.request.urlopen(request)
            html_code = r.read()
            etree = fromstring(html_code)
            etree.make_links_absolute(url)
            file_name = copy_profile_photo_to_static(etree)
            return tostring(etree)

    except requests.exceptions.ConnectionError:
        flask.flash("Connection error: Max number of retries exceeded")
        return flask.redirect(flask.url_for('root_page'))

    except requests.exceptions.MissingSchema:
        flask.flash("Invalid URL: Improper schema supplied")
        return flask.redirect(flask.url_for('root_page'))

    except FileNotFoundError:
        sleep(1)
        return view_page(url=url, retries=(retries + 1))

    except (Exception, Warning) as error:
        flask.flash(str(type(error)) + ", " + str(error))
        return flask.redirect(flask.url_for('root_page'))


# Bonus ECE Directory Credit
@app.route("/ece")
def view_directory():
    try:
        url='https://engineering.purdue.edu/ECE/People/Faculty'
        request = urllib.request.Request(url)
        # Credit: Adapted from example in Python 3.4 Documentation, urllib.request
        #         License: PSFL https://www.python.org/download/releases/3.4.1/license/
        #                  https://docs.python.org/3.4/library/urllib.request.html
        request.add_header('User-Agent', 'PurdueUniversityClassProject/1.0 (bharadwr@purdue.edu https://goo.gl/dk8u5S)')
        r = urllib.request.urlopen(request)
        html_code = r.read()
        etree = fromstring(html_code)
        etree.make_links_absolute(url)
        document = tostring(etree).decode('utf-8')
        profile_dict = OrderedDict()
        for link, first_name, last_name in findall(
                r'\"(https:\/\/engineering\.purdue\.edu\/ECE\/People\/ptProfile\?resource_id\=\d+)\">(.*)<strong>(.*)<\/strong>',
                document):
            sub_link = urllib.parse.quote_plus(link)
            if sub_link is not None:
                profile_dict[sub_link] = last_name + ', ' + first_name

        return flask.render_template('ece.html', profile_dict=profile_dict)

    except (Exception, Warning) as error:
        flask.flash(str(type(error)) + ", " + str(error))
        return flask.redirect(flask.url_for('root_page'))


@app.route("/")
def root_page():
    return flask.render_template('root.html')


if __name__ == "__main__":
    port = environ.get("ECE364_HTTP_PORT", 8000)
    app.secret_key = sha1(port.encode()).hexdigest().encode()
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host="127.0.0.1", port=port, use_reloader=True, use_evalex=False, debug=True, use_debugger=False)
