import io
import os.path

import cv2
import numpy as np
import PySimpleGUI as sg
import pywt
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

import preprocessing

def event_loop(window):
    while True:
        event, values = window.read()

        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        if event == '-Q FOLDER-':
            folder = values['-Q FOLDER-']
            file_list = os.listdir(folder)
            file_names = []
            for file_name in file_list:
                file_path = os.path.join(folder, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith('.jpg'):
                    file_names.append(file_name)
            window['-FILE LIST-'].update(file_names)

        elif event == '-FILE LIST-':
            file_name = os.path.join(values['-Q FOLDER-'], values['-FILE LIST-'][0])
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocessing.crop_centre_square(img)
            window['-IMAGE-'].update(data=get_img_data(img, first=True))


def get_layout():

    db_row = [
            sg.Text('Database Folder', size=(12,1)),
            sg.In(size=(25, 1), enable_events=True, key="-D FOLDER-"),
            sg.FolderBrowse(),
    ]

    search_row = [
            sg.Text('Query Folder', size=(12,1)),
            sg.In(size=(25, 1), enable_events=True, key="-Q FOLDER-"),
            sg.FolderBrowse(),
    ]

    list_box_elem = sg.Listbox(
            values=[], 
            enable_events=True, 
            size=(47, 10), 
            key="-FILE LIST-",
    )

    query_column = [
        db_row,
        search_row,
        [list_box_elem],
        [sg.Image(key="-IMAGE-")],
    ]

    search_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
    ]

    layout = [
        [
            sg.Column(query_column),
            sg.VSeperator(),
            sg.Column(search_viewer_column),
        ]
    ]
    
    return layout


def get_img_data(img, maxsize=(345,345), first=False):
    img = Image.fromarray(img)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

    
def display_image(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_image(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def plot_coeffs(coeffs):
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    plt.imshow(arr, cmap=plt.cm.gray)
    plt.show()


def plot(query_img, imgs):
    time.sleep(5)
    b,g,r = cv2.split(query_img)
    query_img = cv2.merge([r,g,b])

    # below could be stuffing us up...
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=4, nrows=(len(imgs)//4)+5, figure=fig)
    query_ax = fig.add_subplot(spec[0:2,1:3])
    query_ax.imshow(query_img)
    plt.xticks([]), plt.yticks([])
    
    for i in range(3,(len(imgs)//4) + 6):
        for j in range(4):
            if not imgs:
                break
            b,g,r = cv2.split(imgs.pop(0))
            img = cv2.merge([r,g,b])
            db_ax = fig.add_subplot(spec[i,j])
            db_ax.imshow(img)
            plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':
    layout = get_layout()
    window = sg.Window("WBIIS Viewer", layout).Finalize()
    window.Maximize()
    event_loop(window)














