""" The GUI module allows the user to interface with the preprocessing and
wavelet modules.

This is the application's entry point and encapsulates all visual components
of the program.
"""

import gc
import io
import os.path
import random

import cv2
import matplotlib
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

import preprocessing
import wavelet_index_search


def event_loop(window, orig_db_folder):
    """Waits and dispatches events triggored by the user.

    Args:
        window: the window on which all application contents is displayed.
        orig_db_folder: refers to the folder containing the collection of images
            to be displayed. This can be separate from the main database folder as
            long as the file names of corresponding images are the same. This is
            done if you want to display higher res versions of the images.
    """

    params = {
            'orig_db_folder': orig_db_folder,
            'file_names': None,
            'kd_tree': None,
            'db_vectors': None,
            'closest': None,
            'canvas': None,
            'fig': None
    }

    while True:
        event, values = window.read()

        if event in ('Exit', sg.WIN_CLOSED):
            break

        if event == '-Q FOLDER-':
            update_query_file_list(window, event, values)

        elif event == '-FILE LIST-':
            show_selected_query_image(window, event, values)

        elif event in ('-F LOAD-', '-L LOAD-'):
            file_names, kd_tree, db_vectors = load_database(window, event, values)
            params['file_names'] = file_names
            params['kd_tree'] = kd_tree
            params['db_vectors'] = db_vectors

        elif event in ('-F QUERY-', '-S QUERY-'):
            closest, canvas, fig = query(window, event, values, params)
            params['closest'] = closest
            params['canvas'] = canvas
            params['fig'] = fig

        elif event == '-SAVE-':
            save_images(window, event, values, params)


def update_query_file_list(window, event, values):
    """Updates file list box element to display available JPEG files."""
    folder = values['-Q FOLDER-']
    file_list = os.listdir(folder)
    file_names = []
    for file_name in file_list:
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith('.jpg'):
            file_names.append(file_name)
    window['-FILE LIST-'].update(file_names)


def show_selected_query_image(window, event, values):
    """Displays image selected from file list box element."""
    file_path = os.path.join(values['-Q FOLDER-'], values['-FILE LIST-'][0])
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing.crop_centre_square(img)
    window['-IMAGE-'].update(data=get_img_data(img, first=True))


def load_database(window, event, values):
    """Loads the database feature vectors and kd-tree for images from specified folder."""
    num_images = values['-D SIZE-']
    db_folder = values['-D FOLDER-']
    if num_images == '' or db_folder == '':
        window['-STR LOAD-'].update(
                value='Invalid database path or database size',
                visible=True,
        )
        return None, None, None
    num_images = int(num_images)
    file_names = random.sample(os.listdir(db_folder), num_images)
    file_paths = list(map(lambda f: os.path.join(db_folder, f), file_names))

    if event == '-F LOAD-':
        db_imgs = preprocessing.multi_preprocess(file_paths)
    else:
        db_imgs = preprocessing.lazy_preprocess(file_paths)
    feature_vectors = wavelet_index_search.init_feature_vectors(db_imgs, level=3)
    gc.collect()
    kd_tree = wavelet_index_search.init_kd_tree(feature_vectors)
    window['-STR LOAD-'].update(value='Complete!', visible=True)
    return file_names, kd_tree, feature_vectors


def query(window, event, values, params):
    "Queries image for semantically similar images from the loaded database."""
    orig_db_folder = params['orig_db_folder']
    file_names = params['file_names']
    kd_tree = params['kd_tree']
    db_vectors = params['db_vectors']

    query_folder = values['-Q FOLDER-']
    query_size = values['-Q SIZE-']
    file_list = values['-FILE LIST-']
    if not all((orig_db_folder, file_names, kd_tree, db_vectors)):
        return None, None, None
    if query_folder == '' or query_size == '' or not file_list:
        return None, None, None
    query_path = os.path.join(query_folder, file_list[0])
    query_size = int(query_size)
    query_img = preprocessing.process_image(query_path, cropped=False, scaled=False)
    vector = wavelet_index_search.construct_feature_vector(query_img, level=3)
    if event == '-F QUERY-':
        closest = wavelet_index_search.fast_query(vector, kd_tree, query_size)
    else:
        closest = wavelet_index_search.slow_query(vector, db_vectors, query_size)
    indices = [i for (i, _) in closest]
    get_orig = lambda i: cv2.imread(os.path.join(orig_db_folder, file_names[i]),
            cv2.IMREAD_COLOR)
    orig_imgs = list(map(get_orig, indices))
    cropped = list(map(preprocessing.crop_centre_square, orig_imgs))

    canvas = params['canvas']
    fig = params['fig']
    if fig is None:
        matplotlib.use('TkAgg')
        fig = plot(cropped)
        canvas = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    else:
        plot(cropped, fig)
        canvas.draw()
    return closest, canvas, fig


def save_images(window, event, values, params):
    """Saves the results of a query."""
    orig_db_folder = params['orig_db_folder']
    save_folder = values['-S FOLDER-']
    file_names = params['file_names']
    closest = params['closest']
    for i, _ in closest:
        orig_path = os.path.join(orig_db_folder, file_names[i])
        save_path = os.path.join(save_folder, file_names[i])
        orig_img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, orig_img)


def get_img_data(img, maxsize=(345,345), first=False):
    """Formats img so that it can be viewed using PySimpleGUI."""
    img = Image.fromarray(img)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)


def draw_figure(canvas, figure):
    """Draws Matplotlib figure onto the given canvas."""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def plot(imgs, fig=None):
    """Plots the result of the query into a grid limited to four columns."""
    if fig:
        fig.clear()
    else:
        fig = plt.figure(
                figsize=(100,100),
                constrained_layout=True,
                facecolor='#65778d',
        )
    ncols = 4
    nrows = len(imgs)//4 + 1
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    for i in range(nrows):
        for j in range(ncols):
            if not imgs:
                break
            b,g,r = cv2.split(imgs.pop(0))
            img = cv2.merge([r,g,b])
            db_ax = fig.add_subplot(spec[i,j])
            db_ax.imshow(img)
            plt.xticks([]); plt.yticks([])
    return fig


def get_layout():
    """Defines the GUI layout of all components in the application."""
    search_row = [
            sg.Text('Query Folder', size=(12,1)),
            sg.In(size=(25, 1), enable_events=True, key='-Q FOLDER-'),
            sg.FolderBrowse(),
    ]

    list_box_elem = sg.Listbox(
            values=[],
            enable_events=True,
            size=(47, 6),
            key='-FILE LIST-',
    )

    db_row = [
            sg.Text('Database Folder', size=(12,1)),
            sg.In(size=(25, 1), enable_events=True, key='-D FOLDER-'),
            sg.FolderBrowse(),
    ]

    load_row = [
            sg.Text('Database Size', size=(12,1)),
            sg.In(size=(6,1), key='-D SIZE-'),
            sg.Button('Fast Load', size=(10,1), key='-F LOAD-'),
            sg.Button('Lazy Load', size=(10,1), key='-L LOAD-'),
    ]

    display_load_elem = sg.Text(
            '',
            size=(43,1),
            visible=False,
            justification='center',
            key='-STR LOAD-'
    )

    query_row = [
            sg.Text('Query Size', size=(12,1)),
            sg.In(size=(6,1), key='-Q SIZE-'),
            sg.Button('Fast Query', size=(10,1), key='-F QUERY-'),
            sg.Button('Slow Query', size=(10,1), key='-S QUERY-'),
    ]

    save_row = [
            sg.Text('Save Folder', size=(12,1)),
            sg.In(size=(25, 1), enable_events=True, key='-S FOLDER-'),
            sg.FolderBrowse(),
    ]

    save_button = [
            sg.Button('Save Images', size=(43,1), key='-SAVE-'),
    ]

    query_column = [
        search_row,
        [list_box_elem],
        [sg.Image(key='-IMAGE-')],
        db_row,
        load_row,
        [display_load_elem],
        query_row,
        save_row,
        save_button,
    ]

    search_viewer_column = [
        [sg.Canvas(key='-CANVAS-')],
    ]

    layout = [
        [
            sg.Column(query_column),
            sg.VSeperator(),
            sg.Column(search_viewer_column),
        ]
    ]

    return layout


if __name__ == '__main__':
    layout = get_layout()
    window = sg.Window('WBIIS Viewer', layout).Finalize()
    window.Maximize()
    orig_db_folder = 'resources/db/'
    event_loop(window, orig_db_folder)
